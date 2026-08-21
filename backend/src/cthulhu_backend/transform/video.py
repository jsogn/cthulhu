"""画面层内容脱敏：分镜重排、变速、重新构图、重调光与贴纸。"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
from scipy.fftpack import dctn, idctn
from scipy.ndimage import gaussian_filter, zoom

from cthulhu_backend.watermark.qim import MID_BAND


def reorder_shots(frames: np.ndarray, boundaries: list[int], rng: np.random.Generator) -> np.ndarray:
    shots = [frames[a:b] for a, b in pairwise(boundaries)]
    order = rng.permutation(len(shots))
    return np.concatenate([shots[i] for i in order], axis=0)


def retime(frames: np.ndarray, factor: float) -> np.ndarray:
    """变速：factor>1 加速（抽帧），factor<1 减速（重复帧）。"""
    if factor <= 0:
        raise ValueError("factor 必须为正")
    count = max(1, round(len(frames) / factor))
    indices = np.minimum(len(frames) - 1, np.floor(np.arange(count) * factor).astype(int))
    return np.asarray(frames)[indices]


def recrop(frames: np.ndarray, crop_frac: float = 0.04) -> np.ndarray:
    """重新构图：四周裁剪后缩回原分辨率。"""
    h, w = frames.shape[1:3]
    cy0, cy1 = int(h * crop_frac), h - int(h * crop_frac)
    cx0, cx1 = int(w * crop_frac), w - int(w * crop_frac)

    def resize(frame: np.ndarray) -> np.ndarray:
        cropped = frame[cy0:cy1, cx0:cx1]
        factors = (h / cropped.shape[0], w / cropped.shape[1])
        if cropped.ndim == 3:
            factors = factors + (1.0,)
        rescaled = zoom(cropped, factors, order=1)
        return np.clip(rescaled, 0, 1)

    from cthulhu_backend.transform.parallel import map_frames

    return map_frames(resize, frames)


def regrade(
    frames: np.ndarray,
    rng: np.random.Generator,
    strength: float = 0.1,
    brightness: float = 0.04,
) -> np.ndarray:
    """重调光：逐帧伽马与亮度微调，strength 控制伽马范围 (1±strength)。"""
    gamma_range = (1.0 - strength, 1.0 + strength)
    gammas = rng.uniform(*gamma_range, size=len(frames))
    deltas = rng.uniform(-brightness, brightness, size=len(frames))
    return regrade_with_params(frames, gammas, deltas)


def regrade_with_params(
    frames: np.ndarray,
    gammas: np.ndarray,
    deltas: np.ndarray,
) -> np.ndarray:
    """按预先抽好的逐帧参数重调光（流式分块时保持全局确定性）。"""
    # 批量广播：灰度 (F,H,W) 与彩色 (F,H,W,3) 通用，避免逐帧 Python 循环。
    tail = (1,) * (frames.ndim - 1)
    return np.clip(
        frames ** gammas.reshape((-1,) + tail) + deltas.reshape((-1,) + tail),
        0.0,
        1.0,
    )


def overlay_banner(
    frames: np.ndarray,
    text: str,
    seed: int = 0,
    margin_frac: float = 0.04,
) -> np.ndarray:
    """叠加半透明贴纸条（ASCII 文本 + 底色块），模拟二次加工痕迹。"""
    from PIL import Image, ImageDraw, ImageFont

    rng = np.random.default_rng(seed)
    h, w = frames.shape[1:3]
    color = frames.ndim == 4
    margin = int(w * margin_frac)
    box_h = int(h * 0.12)
    y0 = int(rng.uniform(margin, max(margin + 1, h - box_h - margin)))
    font = ImageFont.load_default(size=max(12, box_h - 10))

    def draw_frame(frame: np.ndarray) -> np.ndarray:
        base = (np.clip(frame, 0, 1) * 255).round().astype(np.uint8)
        if color:
            image = Image.fromarray(base, mode="RGB").convert("RGBA")
        else:
            image = Image.fromarray(base, mode="L").convert("RGBA")
        draw = ImageDraw.Draw(image, "RGBA")
        draw.rectangle([margin, y0, w - margin, y0 + box_h], fill=(0, 0, 0, 120))
        draw.text((margin + 12, y0 + (box_h - 16) // 2), text, fill=(255, 255, 255, 220), font=font)
        if color:
            return np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
        return np.asarray(image.convert("L"), dtype=np.float64) / 255.0

    from cthulhu_backend.transform.parallel import map_frames

    return map_frames(draw_frame, frames)


def sharpen(
    frames: np.ndarray,
    amount: float = 0.25,
    radius: float = 1.2,
    threshold: float = 0.01,
) -> np.ndarray:
    """USM 非锐化掩模：抵消清洗带来的轻微模糊（PRD 3.2.1 锐度补偿）。"""
    def unsharp(frame: np.ndarray) -> np.ndarray:
        sigma = (radius, radius) if frame.ndim == 2 else (radius, radius, 0.0)
        blurred = gaussian_filter(frame, sigma=sigma)
        mask = frame - blurred
        mask[np.abs(mask) < threshold] = 0.0
        return np.clip(frame + amount * mask, 0, 1)

    from cthulhu_backend.transform.parallel import map_frames

    return map_frames(unsharp, frames)


def color_restore(
    frames: np.ndarray,
    ref_mean: float | np.ndarray,
    ref_std: float | np.ndarray,
) -> np.ndarray:
    """亮度还原：把处理后帧的逐通道均值校准回参考统计（PRD 3.2.1 色彩还原）。

    只校正均值、不做逐帧方差归一：全局方差包含镜头间差异，逐帧强行对齐
    会造成对比度扩张与 0/255 裁剪，反而让画面整体变暗。
    """
    ref_mean = np.asarray(ref_mean)
    del ref_std
    means = frames.mean(axis=(1, 2), keepdims=True)
    corrected = frames + (ref_mean - means)
    return np.clip(corrected, 0.0, 1.0)


def midband_perturb(
    frames: np.ndarray,
    strength: float = 0.4,
    seed: int = 0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """中频系数随机扰动：破坏二次嵌入基准（PRD 3.1.2 抗二次检测增强）。"""
    if rng is None:
        rng = np.random.default_rng(seed)

    def perturb_channel(frame: np.ndarray) -> np.ndarray:
        frame8 = frame * 255.0
        h, w = frame8.shape
        padded = np.pad(frame8, ((0, -h % 8), (0, -w % 8)), mode="edge")
        blocks = padded.reshape(padded.shape[0] // 8, 8, padded.shape[1] // 8, 8)
        coeffs = dctn(blocks, axes=(1, 3), norm="ortho")
        bh, bw = coeffs.shape[0], coeffs.shape[2]
        flat = coeffs.transpose(0, 2, 1, 3).reshape(bh, bw, 64)
        flat[:, :, MID_BAND] += rng.standard_normal((bh, bw, len(MID_BAND))) * strength
        coeffs = flat.reshape(bh, bw, 8, 8).transpose(0, 2, 1, 3)
        recon = idctn(coeffs, axes=(1, 3), norm="ortho").reshape(padded.shape)
        return np.clip(recon[:h, :w] / 255.0, 0, 1)

    out = []
    for frame in frames:
        if frame.ndim == 3:
            out.append(np.stack([perturb_channel(frame[..., c]) for c in range(3)], axis=-1))
        else:
            out.append(perturb_channel(frame))
    return np.asarray(out)
