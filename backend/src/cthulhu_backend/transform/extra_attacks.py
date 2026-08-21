"""底层对抗原语：把研究 harness 里已有的攻击能力接入清洗管线。

所有原语均为可选、可组合、确定性（同一 seed 下结果一致），灰度/彩色
帧通用。攻击顺序：几何（镜像/平移抖动）→ 空域（中值/噪声/像素重量化）
→ 频域（DCT 重量化）→ 色度量化 → 时域（抽帧复制）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.fftpack import dctn, idctn
from scipy.ndimage import median_filter


@dataclass
class AssaultParams:
    """底层攻击组合参数；零值表示关闭对应原语。"""

    mirror: bool = False
    jitter: float = 0.0
    median: int = 0
    noise: float = 0.0
    requant: int = 0
    dct_step: float = 0.0
    chroma_levels: int = 0
    drop_every: int = 0

    @property
    def enabled(self) -> bool:
        return any(
            (
                self.mirror,
                self.jitter > 0,
                self.median > 0,
                self.noise > 0,
                self.requant > 0,
                self.dct_step > 0,
                self.chroma_levels > 0,
                self.drop_every > 0,
            )
        )


def _spatial_kernel(shape: tuple[int, ...], size: int) -> tuple[int, ...]:
    """把灰度 (H,W) 的滤波尺寸推广到彩色 (H,W,3)（色通道不做滤波）。"""
    if len(shape) == 2:
        return (size, size)
    return (size, size, 1)


def median(frames: np.ndarray, size: int) -> np.ndarray:
    if size <= 0:
        return frames
    # PIL 的 MedianFilter 是 C 实现，比 scipy 快一个数量级；浮点帧先转 uint8。
    if frames.dtype == np.uint8:
        from PIL import Image, ImageFilter

        from cthulhu_backend.transform.parallel import map_frames

        def filt(frame: np.ndarray) -> np.ndarray:
            mode = "RGB" if frame.ndim == 3 else "L"
            image = Image.fromarray(frame, mode=mode)
            return np.asarray(image.filter(ImageFilter.MedianFilter(size=size)), dtype=np.uint8)

        return map_frames(filt, frames)
    from cthulhu_backend.transform.parallel import map_frames

    return map_frames(
        lambda frame: median_filter(frame, size=_spatial_kernel(frame.shape, size)),
        frames,
    )


def gaussian_noise(frames: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    if sigma <= 0:
        return frames
    original_dtype = frames.dtype
    work = frames.astype(np.float32)
    if original_dtype == np.uint8:
        work /= 255.0
    noisy = np.clip(
        work + rng.standard_normal(frames.shape, dtype=np.float32) * sigma,
        0.0,
        1.0,
    )
    if original_dtype == np.uint8:
        return (noisy * 255.0).round().astype(np.uint8)
    return noisy.astype(original_dtype)


def pixel_requant(frames: np.ndarray, levels: int) -> np.ndarray:
    if levels <= 1:
        return frames
    original_dtype = frames.dtype
    work = frames.astype(np.float32)
    if original_dtype == np.uint8:
        work /= 255.0
    requantized = np.round(work * (levels - 1)) / (levels - 1)
    if original_dtype == np.uint8:
        return (requantized * 255.0).round().astype(np.uint8)
    return requantized.astype(original_dtype)


def dct_requant(frames: np.ndarray, step: float) -> np.ndarray:
    if step <= 0:
        return frames
    # 批量向量化：一次 dctn/idctn 处理所有帧（及所有通道），避免逐帧 Python 循环。
    work = np.asarray(frames, dtype=np.float32)
    if work.max() <= 1.0:
        work = work * 255.0
    h, w = work.shape[1:3]
    channels = 1 if work.ndim == 3 else 3
    pad_h, pad_w = -h % 8, -w % 8
    if pad_h or pad_w:
        work = np.pad(work, ((0, 0), (0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    ph, pw = work.shape[1:3]
    blocks = work.reshape(len(work), ph // 8, 8, pw // 8, 8, channels)
    coeffs = dctn(blocks, axes=(2, 4), norm="ortho")
    coeffs = np.round(coeffs / step) * step
    recon = idctn(coeffs, axes=(2, 4), norm="ortho").reshape(len(work), ph, pw, channels)
    result = np.clip(recon[:, :h, :w] / 255.0, 0.0, 1.0)
    if channels == 1:
        result = result[..., 0]
    if frames.dtype == np.uint8:
        return (result * 255.0).round().astype(np.uint8)
    return result.astype(frames.dtype)


def chroma_quant(frames: np.ndarray, levels: int) -> np.ndarray:
    """色度量化：把 Cb/Cr 压缩到有限级数，破坏色度域水印（灰度帧原样返回）。"""
    if levels <= 0 or frames.ndim != 4:
        return frames
    original_dtype = frames.dtype
    rgb = np.asarray(frames, dtype=np.float32)
    if original_dtype == np.uint8:
        rgb /= 255.0
    # BT.601 全范围 YCbCr 正变换与逆变换。
    y = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    cb = -0.168736 * rgb[..., 0] - 0.331264 * rgb[..., 1] + 0.5 * rgb[..., 2]
    cr = 0.5 * rgb[..., 0] - 0.418688 * rgb[..., 1] - 0.081312 * rgb[..., 2]
    cb = np.round(cb * (levels - 1)) / (levels - 1)
    cr = np.round(cr * (levels - 1)) / (levels - 1)
    out = np.empty_like(rgb)
    out[..., 0] = np.clip(y + 1.402 * cr, 0.0, 1.0)
    out[..., 1] = np.clip(y - 0.344136 * cb - 0.714136 * cr, 0.0, 1.0)
    out[..., 2] = np.clip(y + 1.772 * cb, 0.0, 1.0)
    if original_dtype == np.uint8:
        return (np.clip(out, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    return out.astype(original_dtype)


def mirror(frames: np.ndarray) -> np.ndarray:
    """水平镜像。"""
    return np.flip(frames, axis=frames.ndim - 2)


def translate_jitter(frames: np.ndarray, jitter: float, rng: np.random.Generator) -> np.ndarray:
    """逐帧随机平移抖动：随机裁剪偏移后缩回原尺寸，模拟机位晃动。"""
    if jitter <= 0:
        return frames
    from PIL import Image

    from cthulhu_backend.transform.parallel import map_frames

    is_u8 = frames.dtype == np.uint8

    def shift_one(frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        dx = int(rng.uniform(-jitter, jitter) * w)
        dy = int(rng.uniform(-jitter, jitter) * h)
        box = (
            max(0, -dx),
            max(0, -dy),
            min(w, w - dx),
            min(h, h - dy),
        )
        mode = "RGB" if frame.ndim == 3 else "L"
        if is_u8:
            image = Image.fromarray(frame, mode=mode)
        else:
            image = Image.fromarray((np.clip(frame, 0, 1) * 255).round().astype(np.uint8), mode=mode)
        shifted = image.crop(box).resize((w, h), Image.BILINEAR)
        result = np.asarray(shifted)
        if is_u8:
            return result.astype(np.uint8)
        return result.astype(np.float32) / 255.0

    return map_frames(shift_one, frames)


def drop_duplicate(frames: np.ndarray, every: int) -> np.ndarray:
    if every <= 1:
        return frames
    out = np.asarray(frames).copy()
    for index in range(every, len(out), every):
        out[index] = out[index - 1]
    return out


def apply(
    frames: np.ndarray,
    params: AssaultParams,
    rng: np.random.Generator,
) -> np.ndarray:
    """按固定顺序施加组合攻击，全部为可选开关。"""
    work = frames
    if params.mirror:
        work = mirror(work)
    if params.jitter > 0:
        work = translate_jitter(work, params.jitter, rng)
    if params.median > 0:
        work = median(work, params.median)
    if params.noise > 0:
        work = gaussian_noise(work, params.noise, rng)
    if params.requant > 0:
        work = pixel_requant(work, params.requant)
    if params.dct_step > 0:
        work = dct_requant(work, params.dct_step)
    if params.chroma_levels > 0:
        work = chroma_quant(work, params.chroma_levels)
    if params.drop_every > 0:
        work = drop_duplicate(work, params.drop_every)
    return work
