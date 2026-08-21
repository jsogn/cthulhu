"""空域攻击：滤波去噪、像素重量化、LSB 随机化。"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, median_filter
from scipy.signal import wiener


def median(frames: np.ndarray, size: int = 3) -> np.ndarray:
    return np.stack([median_filter(f, size=size) for f in frames])


def gaussian(frames: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    return np.stack([gaussian_filter(f, sigma=sigma) for f in frames])


def wiener_denoise(frames: np.ndarray, size: int = 5) -> np.ndarray:
    """逐帧 Wiener 去噪（线程并行，详见 transform/parallel）。"""
    from cthulhu_backend.transform.parallel import map_frames

    def filt(frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 3:
            return np.stack(
                [wiener(frame[..., c], (size, size)).astype(frame.dtype) for c in range(3)],
                axis=-1,
            )
        return wiener(frame, (size, size)).astype(frame.dtype)

    # scipy 的 wiener 内部升为 float64，结果转回原精度以控制分块内存。
    return map_frames(filt, frames)


def requant_pixels(frames: np.ndarray, levels: int = 32) -> np.ndarray:
    """把像素重量化到有限级数，破坏空域低位与细微扰动。"""
    return np.round(frames * (levels - 1)) / (levels - 1)


def randomize_lsb(frames: np.ndarray, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = (frames * 255).round().astype(np.uint8).copy()
    mask = rng.integers(0, 2, size=out.shape).astype(np.uint8)
    out ^= mask
    return out.astype(np.float64) / 255.0
