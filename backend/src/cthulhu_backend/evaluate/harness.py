"""攻击评估闭环：嵌入 → 攻击 → 提取 → BER/PSNR/SSIM。"""

from __future__ import annotations

import os
import tempfile

import numpy as np

from cthulhu_backend import samples, services
from cthulhu_backend.attacks import dct as dct_attacks
from cthulhu_backend.attacks import geometric, spatial, temporal
from cthulhu_backend.evaluate import metrics
from cthulhu_backend.media import ffmpeg
from cthulhu_backend.similarity import embedding
from cthulhu_backend.transform import audio as audio_transform
from cthulhu_backend.transform import shots
from cthulhu_backend.transform import video as video_transform
from cthulhu_backend.watermark import common, detect, dwt, echo, lsb, qim, qim_rep, ss

VIDEO_EMBEDDERS = {
    "lsb": (lsb.embed, lsb.extract),
    "ss": (ss.embed, ss.extract),
    "qim": (qim.embed, qim.extract),
    "qim-rep": (qim_rep.embed, qim_rep.extract),
    "dwt": (dwt.embed, dwt.extract),
}

VIDEO_ATTACKS = {
    "median": spatial.median,
    "gaussian": spatial.gaussian,
    "wiener": spatial.wiener_denoise,
    "requant": spatial.requant_pixels,
    "requant-dct": lambda frames: np.stack([dct_attacks.requant_dct(f) for f in frames]),
    "lsb-randomize": spatial.randomize_lsb,
    "geometric": geometric.crop_rotate_rescale,
    "temporal": temporal.drop_duplicate,
}

AUDIO_ATTACKS = {
    "speed": lambda sig, rng: temporal.speed_change_audio(sig, 0.99),
    "noise": lambda sig, rng: np.clip(sig + 0.01 * rng.standard_normal(len(sig)), -1, 1),
}


def run_video_harness(
    method: str,
    frames: np.ndarray,
    bits: list[int],
    attacks: list[str],
    seed: int = 0,
    **kwargs: object,
) -> dict:
    embed_fn, extract_fn = VIDEO_EMBEDDERS[method]
    seeded = method in {"lsb", "ss", "dwt"}
    watermarked = np.stack(
        [
            embed_fn(frame, bits, seed=seed, **kwargs) if seeded else embed_fn(frame, bits, **kwargs)
            for frame in frames
        ]
    )
    results = {}
    for attack in attacks:
        attacked = VIDEO_ATTACKS[attack](watermarked)
        out_bits = []
        for frame in attacked:
            out_bits.extend(
                extract_fn(frame, len(bits), seed=seed, **kwargs)
                if seeded
                else extract_fn(frame, len(bits), **kwargs)
            )
        ref = (common.SYNC + bits) * len(frames)
        results[attack] = {
            "ber": metrics.ber(ref, out_bits),
            "psnr_db": metrics.psnr(frames, attacked),
            "ssim": metrics.ssim(frames, attacked),
        }
    return {"method": method, "attacks": results}


def run_audio_harness(
    signal: np.ndarray,
    bits: list[int],
    attacks: list[str],
    sample_rate: int = 16000,
    seed: int = 0,
    segment: float = 0.05,
) -> dict:
    watermarked = echo.embed(signal, bits, sample_rate, segment=segment)
    results = {}
    for attack in attacks:
        rng = np.random.default_rng(seed)
        attacked = AUDIO_ATTACKS[attack](watermarked, rng)
        out_bits = echo.extract(attacked, len(bits), sample_rate, segment=segment)
        results[attack] = {
            "ber": metrics.ber(common.SYNC + bits, out_bits),
            "psnr_db": metrics.psnr(signal, attacked),
        }
    return {"method": "echo", "attacks": results}


def run_detection_video_harness(
    methods: list[str],
    frames: np.ndarray,
    bits: list[int],
    attacks: list[str],
    seed: int = 0,
    **kwargs: object,
) -> dict:
    """盲检测评估：干净基线 vs 水印 vs 攻击后，各方案置信度对比。"""
    clean_scores = detect.video_scores(frames)
    report: dict = {"clean": clean_scores}
    for method in methods:
        embed_fn = VIDEO_EMBEDDERS[method][0]
        seeded = method in {"lsb", "ss", "dwt"}
        watermarked = np.stack(
            [
                embed_fn(frame, bits, seed=seed, **kwargs)
                if seeded
                else embed_fn(frame, bits, **kwargs)
                for frame in frames
            ]
        )
        watermarked_scores = detect.video_scores(watermarked)
        attacked_scores = {
            attack: detect.video_scores(VIDEO_ATTACKS[attack](watermarked))[method]
            for attack in attacks
        }
        report[method] = {
            "watermarked": watermarked_scores[method],
            "attacked": attacked_scores,
        }
    return report


def run_detection_audio_harness(
    signal: np.ndarray,
    bits: list[int],
    attacks: list[str],
    sample_rate: int = 16000,
    seed: int = 0,
    segment: float = 0.25,
) -> dict:
    """音频盲检测评估：干净 vs 回声水印 vs 攻击后置信度。"""
    clean_score = detect.audio_scores(signal, sample_rate)["echo"]
    watermarked = echo.embed(signal, bits, sample_rate, segment=segment)
    watermarked_score = detect.audio_scores(watermarked, sample_rate)["echo"]
    attacked_scores = {}
    for attack in attacks:
        rng = np.random.default_rng(seed)
        attacked = AUDIO_ATTACKS[attack](watermarked, rng)
        attacked_scores[attack] = detect.audio_scores(attacked, sample_rate)["echo"]
    return {
        "clean": clean_score,
        "watermarked": watermarked_score,
        "attacked": attacked_scores,
    }


def _codec_roundtrip(frames: np.ndarray, fps: float, crf: int) -> np.ndarray:
    """模拟真实分发链路的编码-解码往返。"""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "roundtrip.mp4")
        ffmpeg.encode_video(frames, path, fps=fps, crf=crf)
        decoded, _ = ffmpeg.decode_video(path)
        return decoded


def _stats(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": round(float(arr.mean()), 4),
        "std": round(float(arr.std()), 4),
        "values": [round(float(v), 4) for v in arr],
    }


def run_compressed_detection_baseline(
    methods: list[str],
    width: int,
    height: int,
    frames_n: int,
    payload_bits: int,
    seeds: list[int],
    attacks: list[str],
    fps: float = 30.0,
    crf: int = 23,
) -> dict:
    """压缩域差分基线：干净 / 水印 / 攻击后均经编码往返后检测。

    目的：量化「有损压缩对像素域检测器的污染」与「水印在真实链路上的
    生存力」，为差分判定提供可复现阈值依据。
    """
    report: dict = {"config": {
        "methods": methods,
        "resolution": f"{width}x{height}",
        "frames": frames_n,
        "payload_bits": payload_bits,
        "seeds": seeds,
        "attacks": attacks,
        "fps": fps,
        "crf": crf,
    }, "methods": {}}
    for method in methods:
        clean_values: list[float] = []
        watermarked_values: list[float] = []
        attacked_values: dict[str, list[float]] = {attack: [] for attack in attacks}
        embed_fn = VIDEO_EMBEDDERS[method][0]
        for seed in seeds:
            clean = samples.make_video_frames(frames_n, width, height, seed=seed)
            bits = common.payload_bits(seed + 1, payload_bits)
            clean_coded = _codec_roundtrip(clean, fps, crf)
            clean_values.append(detect.video_scores(clean_coded)[method])

            if method == "ss":
                watermarked = np.stack(
                    [embed_fn(frame, bits, seed=seed, alpha=0.03) for frame in clean],
                )
            elif method in {"lsb", "dwt"}:
                watermarked = np.stack(
                    [embed_fn(frame, bits, seed=seed) for frame in clean],
                )
            else:
                watermarked = np.stack([embed_fn(frame, bits) for frame in clean])
            watermarked_coded = _codec_roundtrip(watermarked, fps, crf)
            watermarked_values.append(detect.video_scores(watermarked_coded)[method])

            for attack in attacks:
                attacked = VIDEO_ATTACKS[attack](watermarked_coded)
                attacked_coded = _codec_roundtrip(attacked, fps, crf)
                attacked_values[attack].append(detect.video_scores(attacked_coded)[method])

        watermarked_stats = _stats(watermarked_values)
        clean_stats = _stats(clean_values)
        diff_mean = round(watermarked_stats["mean"] - clean_stats["mean"], 4)
        attacked_report: dict[str, dict] = {}
        for attack, values in attacked_values.items():
            attacked_stats = _stats(values)
            attacked_diff = attacked_stats["mean"] - clean_stats["mean"]
            residual_ratio = (
                round(attacked_diff / diff_mean, 4) if abs(diff_mean) > 1e-9 else None
            )
            attacked_report[attack] = {
                **attacked_stats,
                "diff_mean": round(attacked_diff, 4),
                "residual_ratio": residual_ratio,
            }
        report["methods"][method] = {
            "clean": clean_stats,
            "watermarked": watermarked_stats,
            "diff_mean": diff_mean,
            "attacked": attacked_report,
        }
    return report


def run_desensitize_harness(
    frames: np.ndarray,
    configs: dict[str, dict],
    seed: int = 0,
) -> dict:
    """内容脱敏评估：各参数组合对内容/运动相似度与画质的影响。

    psnr_db 为未时间对齐的帧序差，变速/重排会造成帧错位而拉低分数，
    仅作同一口径下的横向对比，不代表逐帧视觉画质。
    """
    report: dict = {}
    for name, config in configs.items():
        rng = np.random.default_rng(seed)
        out = frames
        if config.get("reorder"):
            out = video_transform.reorder_shots(out, shots.detect_cuts(out), rng)
        speed = config.get("speed", 1.0)
        if speed != 1.0:
            out = video_transform.retime(out, speed)
        recrop = config.get("recrop", 0.0)
        if recrop > 0:
            out = video_transform.recrop(out, recrop)
        if config.get("regrade"):
            out = video_transform.regrade(
                out,
                rng,
                strength=config.get("regrade_strength", 0.03),
                brightness=config.get("regrade_brightness", 0.02),
            )
        if config.get("sharpen"):
            out = video_transform.sharpen(out, amount=0.25, radius=1.2, threshold=0.01)
        if config.get("denoise"):
            out = spatial.wiener_denoise(out, size=5)
        length = min(len(frames), len(out))
        similarity = embedding.similarity_report(frames[:length], out[:length])
        report[name] = {
            "frames": len(out),
            "content_cosine": round(similarity["content_cosine"], 4),
            "motion_cosine": round(similarity["motion_cosine"], 4),
            "temporal_consistency": round(metrics.adjacent_cosine(out), 4),
            "order_disruption": round(metrics.order_disruption(frames, out), 4),
            "aligned_psnr_db": round(metrics.temporal_aligned_psnr(frames, out), 2),
            "psnr_db": round(metrics.psnr(frames[:length], out[:length]), 2),
        }
    return report


def _extract_ber(frames: np.ndarray, extract_fn, bits: list[int], segment: bool = False) -> float:
    if segment:
        return metrics.ber(common.SYNC + bits, extract_fn(frames, len(bits)))
    out_bits: list[int] = []
    for frame in frames:
        out_bits.extend(extract_fn(frame))
    reference = (common.SYNC + bits) * len(frames)
    return metrics.ber(reference, out_bits)


def run_cleanse_matrix(
    frames: np.ndarray,
    variants: dict[str, dict],
    levels: dict[str, dict],
    bits: list[int] | None = None,
    seed: int = 0,
    fps: float = 30.0,
    crf: int = 23,
) -> dict:
    """通杀验收矩阵：各水印变体经编码与各档清洗后的已知水印误码率。

    variants 每个条目需提供 embed(frame, bits) 与 extract(frame)。
    levels 每个条目为 run_desensitize 的关键字参数（不含 path/output）。
    """
    bits = bits or common.payload_bits(seed + 1, 64)
    report: dict = {}
    with tempfile.TemporaryDirectory() as tmp:
        for variant_name, variant in variants.items():
            segment = variant.get("segment", False)
            if segment:
                watermarked = variant["embed"](frames, bits)
            else:
                watermarked = np.stack([variant["embed"](frame, bits) for frame in frames])
            wm_path = os.path.join(tmp, f"{variant_name}-wm.mp4")
            ffmpeg.encode_video(watermarked, wm_path, fps=fps, crf=crf)
            coded, _ = ffmpeg.decode_video(wm_path)
            row: dict = {
                "encoded_ber": round(_extract_ber(coded, variant["extract"], bits, segment), 4),
                "levels": {},
            }
            for level_name, params in levels.items():
                out_path = os.path.join(tmp, f"{variant_name}-{level_name}.mp4")
                services.run_desensitize(wm_path, out_path, **params)
                cleaned, _ = ffmpeg.decode_video(out_path)
                row["levels"][level_name] = round(
                    _extract_ber(cleaned, variant["extract"], bits, segment), 4,
                )
            report[variant_name] = row
    return report


def run_audio_cleanse(
    signal: np.ndarray,
    bits: list[int],
    sample_rate: int = 16000,
    segment: float = 0.05,
    speed_factor: float = 0.97,
    seed: int = 0,
) -> dict:
    """音频通杀验收：回声水印经变速重混后的误码率。"""
    watermarked = echo.embed(signal, bits, sample_rate, segment=segment)
    rng = np.random.default_rng(seed)
    remixed = audio_transform.remix(watermarked, sample_rate, rng, speed_factor=speed_factor)

    def extract_ber(target: np.ndarray) -> float:
        out = echo.extract(target, len(bits), sample_rate, segment=segment)
        return metrics.ber(common.SYNC + bits, out)

    return {
        "before_ber": round(extract_ber(watermarked), 4),
        "after_ber": round(extract_ber(remixed), 4),
    }
