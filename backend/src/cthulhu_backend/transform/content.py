"""内容级变换：在不动叙事顺序的前提下推动语义级（CLIP）指纹。

与像素/几何/时间轴原语不同，这些变换改变画面“内容构成”：
片头片尾裁剪、强电影调色、角标贴片、暗角颗粒、切点溶解、逐镜头重构。
全部保持镜头顺序与剧情连续，是否启用由观感约束决定。
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, zoom

from cthulhu_backend.transform.shots import split_shots


def trim_ends(frames: np.ndarray, head: int, tail: int) -> np.ndarray:
    """裁剪片头片尾帧（片头片尾常与平台判重最相似，且不参与叙事）。"""
    if head + tail >= len(frames):
        return frames.copy()
    return frames[head : len(frames) - tail if tail else None]


def film_grade(
    frames: np.ndarray,
    rng: np.random.Generator,
    temp: float = 0.12,
    tint: float = 0.08,
    contrast: float = 1.08,
) -> np.ndarray:
    """强电影调色：白平衡偏移 + 色调偏移 + 对比度，彩色帧才生效。"""
    arr = np.asarray(frames, dtype=np.float32)
    if arr.ndim != 4:
        return arr
    r_gain = 1.0 + float(rng.uniform(-temp, temp))
    b_gain = 1.0 - float(rng.uniform(-temp, temp))
    g_shift = float(rng.uniform(-tint, tint))
    gains = np.array([r_gain, 1.0, b_gain], dtype=np.float32)
    shifted = arr * gains + np.array([0.0, g_shift, 0.0], dtype=np.float32)
    mean = shifted.mean(axis=(1, 2), keepdims=True)
    out = (shifted - mean) * contrast + mean
    return np.clip(out, 0.0, 1.0)


def corner_mark(
    frames: np.ndarray,
    text: str,
    seed: int = 0,
    alpha: int = 140,
    size_frac: float = 0.055,
) -> np.ndarray:
    """右下角半透明文字角标（如账号/剧名），改变构图内容。"""
    from PIL import Image, ImageDraw, ImageFont

    rng = np.random.default_rng(seed)
    h, w = frames.shape[1:3]
    box_w = int(w * 0.3)
    box_h = int(h * size_frac)
    y0 = h - box_h - int(h * 0.02)
    x0 = w - box_w - int(w * 0.02)
    font = ImageFont.load_default(size=max(12, box_h - 12))

    def draw(frame: np.ndarray) -> np.ndarray:
        image = Image.fromarray(
            (np.clip(frame, 0, 1) * 255).round().astype(np.uint8), mode="RGB"
        ).convert("RGBA")
        draw_layer = ImageDraw.Draw(image, "RGBA")
        draw_layer.rectangle([x0, y0, x0 + box_w, y0 + box_h], fill=(0, 0, 0, 110))
        draw_layer.text(
            (x0 + 10, y0 + (box_h - 16) // 2),
            text,
            fill=(255, 255, 255, alpha),
            font=font,
        )
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0

    from cthulhu_backend.transform.parallel import map_frames

    return map_frames(draw, frames)


def vignette_grain(
    frames: np.ndarray,
    rng: np.random.Generator,
    vignette: float = 0.12,
    grain: float = 0.006,
) -> np.ndarray:
    """暗角 + 细颗粒：给画面统一质感，同时扰动局部统计。"""
    arr = np.asarray(frames, dtype=np.float32)
    h, w = arr.shape[1:3]
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
    mask = 1.0 - vignette * np.clip(r, 0, 1)[None, ..., None] if arr.ndim == 4 else 1.0 - vignette * np.clip(r, 0, 1)
    noise = rng.normal(0.0, grain, arr.shape).astype(np.float32)
    return np.clip(arr * mask + noise, 0.0, 1.0)


def shot_dissolve(
    frames: np.ndarray,
    boundaries: list[int],
    rng: np.random.Generator,
    length: int = 4,
) -> np.ndarray:
    """在切点处做短溶解：保留剧情顺序，打散硬切结构。"""
    out = np.asarray(frames, dtype=np.float32).copy()
    for boundary in boundaries[1:-1]:
        span = min(length, boundary, len(out) - boundary)
        for t in range(1, span + 1):
            alpha = t / (span + 1)
            out[boundary - t] = (1 - alpha) * out[boundary - t] + alpha * out[boundary + t]
    return np.clip(out, 0.0, 1.0)


def per_shot_reframe(
    frames: np.ndarray,
    boundaries: list[int],
    rng: np.random.Generator,
    min_frac: float = 0.02,
    max_frac: float = 0.06,
) -> np.ndarray:
    """逐镜头随机取景窗：每镜头裁剪比例不同，再缩回原尺寸。"""
    from cthulhu_backend.transform.video import recrop

    parts = []
    for shot in split_shots(frames, boundaries):
        crop = float(rng.uniform(min_frac, max_frac))
        parts.append(recrop(shot, crop_frac=crop))
    return np.concatenate(parts, axis=0)
