"""内容级变换效果矩阵：哪些变换能推动 CLIP 语义向量，代价多少。

直接在解码帧上测量，不做编码往返：CLIP 距离、pHash 距离、对齐 SSIM、
叙事连续性。目标是在不动叙事的前提下找到把 CLIP 相似度打下去的手段。
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np

from cthulhu_backend.evaluate import dedup_harness, metrics
from cthulhu_backend.fingerprint import deep
from cthulhu_backend.media import ffmpeg
from cthulhu_backend.transform import content, shots


def _gray(frames: np.ndarray) -> np.ndarray:
    return 0.299 * frames[..., 0] + 0.587 * frames[..., 1] + 0.114 * frames[..., 2]


def _aligned_ssim(ref: np.ndarray, cand: np.ndarray) -> float:
    n = min(len(ref), len(cand))
    indices = np.linspace(0, n - 1, 16).astype(int)
    scores = []
    for i in indices:
        j = min(len(ref) - 1, round(i * len(ref) / len(cand)))
        lo, hi = max(0, j - 6), min(len(ref), j + 7)
        scores.append(max(metrics.ssim(ref[k], cand[i]) for k in range(lo, hi)))
    return float(np.mean(scores))


def evaluate(clip: str, out_dir: str) -> dict:
    frames, info = ffmpeg.decode_video(
        clip, grayscale=False, vf="scale=540:960:flags=lanczos", out_dtype="float32"
    )
    gray = _gray(frames)
    boundaries = shots.detect_cuts(gray)
    fps = info["fps"]
    rng = np.random.default_rng(0)
    base_emb = deep.video_embedding(frames, max_frames=16)

    head = int(1.5 * fps)
    variants: dict[str, Callable[[], np.ndarray]] = {
        "电影调色": lambda: content.film_grade(frames, rng),
        "角标贴片": lambda: content.corner_mark(frames, "Cthulhu · 短剧", seed=0),
        "暗角颗粒": lambda: content.vignette_grain(frames, rng),
        "切点溶解": lambda: content.shot_dissolve(frames, boundaries, rng, length=4),
        "逐镜头重构": lambda: content.per_shot_reframe(frames, boundaries, rng),
        "片头片尾裁剪": lambda: content.trim_ends(frames, head=head, tail=head),
        "组合（调色+重构+溶解）": lambda: content.shot_dissolve(
            content.per_shot_reframe(content.film_grade(frames, rng), boundaries, rng),
            boundaries,
            rng,
            length=3,
        ),
        "字幕对比重塑": lambda: content.subtitle_restyle(frames, rng, "contrast"),
        "字幕色偏": lambda: content.subtitle_restyle(frames, rng, "tint"),
        "字幕锐化": lambda: content.subtitle_restyle(frames, rng, "sharpen"),
        "字幕微移位": lambda: content.subtitle_restyle(frames, rng, "shift"),
        "字幕组合": lambda: content.subtitle_restyle(
            content.subtitle_restyle(frames, rng, "contrast"), rng, "shift"
        ),
        "字幕+全组合": lambda: content.subtitle_restyle(
            content.shot_dissolve(
                content.per_shot_reframe(content.film_grade(frames, rng), boundaries, rng),
                boundaries,
                rng,
                length=3,
            ),
            rng,
            "contrast",
        ),
    }

    rows = []
    for name, build in variants.items():
        out = build()
        gray_out = _gray(out)
        emb = deep.video_embedding(out, max_frames=16)
        clip_distance = 1.0 - deep.cosine(base_emb, emb)
        phash_distance = dedup_harness._frame_dists(gray[:40], gray_out[:40])["phash"]
        rows.append(
            {
                "transform": name,
                "clip_similarity": round(1.0 - clip_distance, 4),
                "clip_distance": round(clip_distance, 4),
                "phash_distance": round(phash_distance, 4),
                "ssim_aligned": round(_aligned_ssim(gray, gray_out), 4),
                "order_cosine": round(metrics.adjacent_cosine(gray_out), 4),
            }
        )
    report = {"clip": clip, "fps": fps, "rows": rows}
    out = Path(out_dir) / f"content-matrix-{Path(clip).stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
