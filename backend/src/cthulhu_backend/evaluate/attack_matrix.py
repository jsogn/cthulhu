"""对抗原语效果矩阵：同一片段跑多档变换，用判重代理栈 + VMAF 双目标打分。

研究口径：输入应为短片段（≤60s）。变换在 540p 色彩帧上进行，输出产物与
同分辨率参考对比，衡量「判重风险下降」与「画质损失」的权衡。
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from cthulhu_backend import parallel
from cthulhu_backend.evaluate import dedup_harness, metrics
from cthulhu_backend.media import ffmpeg
from cthulhu_backend.transform import audio as audio_transform
from cthulhu_backend.transform import shots, strategies
from cthulhu_backend.transform import video as video_transform

_FF = ffmpeg.FFMPEG_BIN


def _mux_audio(video_path: str, audio: np.ndarray, sample_rate: int, out_path: str) -> None:
    wav_path = video_path + ".wav"
    wavfile.write(wav_path, sample_rate, (np.clip(audio, -1, 1) * 32767).astype(np.int16))
    subprocess.run(
        [
            _FF, "-y", "-v", "error",
            "-i", video_path, "-i", wav_path,
            "-map", "0:v", "-map", "1:a",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "-shortest",
            out_path,
        ],
        check=True,
        capture_output=True,
    )


def _gray(frames: np.ndarray) -> np.ndarray:
    return 0.299 * frames[..., 0] + 0.587 * frames[..., 1] + 0.114 * frames[..., 2]


def build_variant(
    reference: str,
    out_path: str,
    *,
    reference_frames: np.ndarray | None = None,
    seed: int,
    shot_retime: tuple[float, float] | None = None,
    cut_margin: int = 0,
    regrade: float = 0.0,
    recrop: float = 0.0,
    rotate_deg: float = 0.0,
    midband: float = 0.0,
    audio_tempo: float = 1.0,
    audio_pitch: float = 1.0,
    audio_eq_db: float = 0.0,
    audio_noise: float = 0.0,
) -> str:
    """对 540p 参考片段应用一组原语并输出带音轨的 MP4。"""
    rng = np.random.default_rng(seed)
    if reference_frames is not None:
        frames = reference_frames
    else:
        frames, _ = ffmpeg.decode_video(
            reference, grayscale=False, vf="scale=540:960:flags=lanczos", out_dtype="float32"
        )
    gray = _gray(frames)
    boundaries = shots.detect_cuts(gray)

    if shot_retime is not None:
        frames = video_transform.per_shot_retime(
            frames, boundaries, rng, min_factor=shot_retime[0], max_factor=shot_retime[1]
        )
    if cut_margin > 0:
        frames = video_transform.cut_jitter(frames, boundaries, rng, margin=cut_margin)
    if recrop > 0:
        frames = video_transform.recrop(frames, crop_frac=recrop)
    if regrade > 0:
        frames = video_transform.regrade(frames, rng, strength=regrade / 4, brightness=regrade / 2)
    if rotate_deg > 0:
        frames = strategies.rotate_de_sync(frames, list(range(len(frames))), rotate_deg)
    if midband > 0:
        frames = video_transform.midband_perturb(frames, strength=midband, rng=rng)

    ffmpeg.encode_video(frames, out_path + ".video.mp4", fps=30.0, crf=23)
    decoded = ffmpeg.decode_audio(reference, max_seconds=60)
    if decoded is not None:
        signal, sample_rate = decoded
        signal = audio_transform.remix_strong(
            signal,
            sample_rate,
            rng,
            tempo=audio_tempo,
            pitch_ratio=audio_pitch,
            eq_db=audio_eq_db,
            noise_floor=audio_noise,
        )
        _mux_audio(out_path + ".video.mp4", signal, sample_rate, out_path)
    else:
        subprocess.run(["mv", out_path + ".video.mp4", out_path], check=True)
    return out_path


def _quality(ref_path: str, variant_path: str, ref_frames: np.ndarray | None = None) -> dict:
    """变速会造成帧错位，VMAF 不可用；改用时间对齐的 SSIM 与叙事连续性。"""
    ref = ref_frames if ref_frames is not None else ffmpeg.decode_video(ref_path, out_dtype="float32")[0]
    cand, _ = ffmpeg.decode_video(variant_path, out_dtype="float32")
    if ref.shape[1:] != cand.shape[1:]:
        return {"ssim_aligned": None, "order_cosine": None}
    step = max(1, len(cand) // 60)
    indices = list(range(0, len(cand), step))[:60]
    scores = []
    for i in indices:
        j = min(len(ref) - 1, round(i * len(ref) / len(cand)))
        lo, hi = max(0, j - 8), min(len(ref), j + 9)
        scores.append(max(metrics.ssim(ref[k], cand[i]) for k in range(lo, hi)))
    return {
        "ssim_aligned": round(float(np.mean(scores)), 4),
        "order_cosine": round(metrics.adjacent_cosine(cand), 4),
    }


PRESETS = {
    "日常": {
        "shot_retime": (0.95, 1.07),
        "cut_margin": 3,
        "regrade": 0.10,
        "recrop": 0.02,
        "rotate_deg": 0.25,
        "midband": 0.1,
        "audio_tempo": 1.05,
        "audio_pitch": 0.98,
        "audio_eq_db": 5.0,
        "audio_noise": 0.004,
    },
    "轻+": {
        "shot_retime": (0.98, 1.04),
        "cut_margin": 2,
        "regrade": 0.08,
        "recrop": 0.015,
        "audio_tempo": 1.04,
        "audio_pitch": 0.985,
        "audio_eq_db": 4.0,
        "audio_noise": 0.003,
    },
    "强": {
        "shot_retime": (0.95, 1.08),
        "cut_margin": 4,
        "regrade": 0.12,
        "recrop": 0.03,
        "rotate_deg": 0.8,
        "midband": 0.5,
        "audio_tempo": 1.06,
        "audio_pitch": 0.97,
        "audio_eq_db": 7.0,
        "audio_noise": 0.006,
    },
}


def run_matrix(clip: str, out_dir: str, seed: int = 0, reuse: bool = False) -> dict:
    work = Path(out_dir)
    work.mkdir(parents=True, exist_ok=True)
    reference = str(work / "ref540.mp4")
    subprocess.run(
        [
            _FF, "-y", "-v", "error", "-i", clip,
            "-vf", "scale=540:960:flags=lanczos",
            "-c:v", "libx264", "-crf", "16", "-preset", "fast",
            "-c:a", "aac", "-b:a", "128k",
            reference,
        ],
        check=True,
        capture_output=True,
    )
    # 参考片只解码一次：全部变体共享同一份只读帧（变换均返回新数组，无原地写）。
    ref_rgb, _ = ffmpeg.decode_video(reference, grayscale=False, out_dtype="float32")
    ref_gray = _gray(ref_rgb)
    built: list[tuple[str, str]] = []
    for name, options in PRESETS.items():
        variant = str(work / f"variant-{name}.mp4")
        if not (reuse and Path(variant).exists()):
            build_variant(reference, variant, reference_frames=ref_rgb, seed=seed, **options)
        built.append((name, variant))

    def _score_one(item: tuple[str, str]) -> dict:
        name, variant = item
        dedup = dedup_harness.compare(reference, variant)
        quality = _quality(reference, variant, ref_frames=ref_gray)
        return {
            "preset": name,
            "duplicate_risk": dedup["duplicate_risk"],
            "risk_level": dedup["risk_level"],
            "distances": dedup["distances"],
            **quality,
        }

    rows = parallel.map_items(_score_one, built)
    report = {"reference": reference, "seed": seed, "rows": rows}
    (work / "attack-matrix.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report
