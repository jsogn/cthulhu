"""空域扩频水印：加性伪随机噪声图案，通过残差相关检测。"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter

from cthulhu_backend.watermark.common import with_sync


def _pattern(shape: tuple[int, int], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pattern = rng.standard_normal(shape)
    return pattern / np.sqrt(np.mean(pattern**2))


def embed(frame: np.ndarray, bits: list[int], seed: int, alpha: float = 0.03) -> np.ndarray:
    bits = with_sync(bits)
    pattern = _pattern(frame.shape, seed)
    out = frame.copy()
    n = min(len(bits), frame.shape[0])
    # 每比特占用若干行，逐行叠加 ±pattern。
    rows_per_bit = frame.shape[0] // n
    for i, bit in enumerate(bits):
        r0, r1 = i * rows_per_bit, (i + 1) * rows_per_bit
        sign = 1.0 if bit else -1.0
        out[r0:r1] += alpha * sign * pattern[r0:r1]
    return np.clip(out, 0, 1)


def extract(frame: np.ndarray, n_bits: int, seed: int) -> list[int]:
    pattern = _pattern(frame.shape, seed)
    residual = frame - median_filter(frame, size=3)
    total = len(with_sync([0] * n_bits))
    n = min(total, frame.shape[0])
    rows_per_bit = frame.shape[0] // n
    bits = []
    for i in range(n):
        r0, r1 = i * rows_per_bit, (i + 1) * rows_per_bit
        corr = np.mean(residual[r0:r1] * pattern[r0:r1])
        bits.append(1 if corr > 0 else 0)
    return bits
