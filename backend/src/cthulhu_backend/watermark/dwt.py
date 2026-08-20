"""小波域 DWT 水印：Haar 分解对角线细节子带加性嵌入。

与空域扩频互补：信号落在高频细节子带，对空域滤波更鲁棒，
可作为基准库中新方案族，检验清洗管线的通杀覆盖。
"""

from __future__ import annotations

import numpy as np

from cthulhu_backend.watermark.common import with_sync


def _decompose(frame: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    """一阶 Haar 二维分解，返回系数矩阵与裁剪后尺寸。"""
    h, w = frame.shape
    h2, w2 = h - h % 2, w - w % 2
    array = frame[:h2, :w2].astype(np.float64)
    even = array[:, 0::2]
    odd = array[:, 1::2]
    low = (even + odd) / np.sqrt(2.0)
    high = (even - odd) / np.sqrt(2.0)
    horizontal = np.concatenate([low, high], axis=1)
    even_rows = horizontal[0::2, :]
    odd_rows = horizontal[1::2, :]
    top = (even_rows + odd_rows) / np.sqrt(2.0)
    bottom = (even_rows - odd_rows) / np.sqrt(2.0)
    return np.concatenate([top, bottom], axis=0), (h2, w2)


def _reconstruct(coeffs: np.ndarray) -> np.ndarray:
    half_h = coeffs.shape[0] // 2
    top = coeffs[:half_h, :]
    bottom = coeffs[half_h:, :]
    rows = np.empty_like(coeffs)
    rows[0::2, :] = (top + bottom) / np.sqrt(2.0)
    rows[1::2, :] = (top - bottom) / np.sqrt(2.0)
    half_w = coeffs.shape[1] // 2
    low = rows[:, :half_w]
    high = rows[:, half_w:]
    out = np.empty_like(coeffs)
    out[:, 0::2] = (low + high) / np.sqrt(2.0)
    out[:, 1::2] = (low - high) / np.sqrt(2.0)
    return out


def _detail_band(coeffs: np.ndarray) -> np.ndarray:
    """对角线细节子带（右下象限）。"""
    return coeffs[coeffs.shape[0] // 2 :, coeffs.shape[1] // 2 :]


def _pattern(shape: tuple[int, int], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pattern = rng.standard_normal(shape)
    return pattern / np.sqrt(np.mean(pattern**2) + 1e-12)


def embed(frame: np.ndarray, bits: list[int], seed: int, alpha: float = 0.08) -> np.ndarray:
    bits = with_sync(bits)
    coeffs, (h2, w2) = _decompose(frame)
    band = _detail_band(coeffs)
    pattern = _pattern(band.shape, seed)
    rows_per_bit = max(1, band.shape[0] // len(bits))
    out = band.copy()
    for index, bit in enumerate(bits):
        r0, r1 = index * rows_per_bit, (index + 1) * rows_per_bit
        sign = 1.0 if bit else -1.0
        out[r0:r1] += alpha * sign * pattern[r0:r1]
    coeffs[coeffs.shape[0] // 2 :, coeffs.shape[1] // 2 :] = out
    reconstructed = _reconstruct(coeffs)
    frame_out = frame.copy()
    frame_out[:h2, :w2] = reconstructed
    return np.clip(frame_out, 0, 1)


def extract(frame: np.ndarray, n_bits: int, seed: int) -> list[int]:
    coeffs, _ = _decompose(frame)
    band = _detail_band(coeffs)
    pattern = _pattern(band.shape, seed)
    total = len(with_sync([0] * n_bits))
    rows_per_bit = max(1, band.shape[0] // total)
    bits = []
    for index in range(total):
        r0, r1 = index * rows_per_bit, (index + 1) * rows_per_bit
        correlation = float(np.mean(band[r0:r1] * pattern[r0:r1]))
        bits.append(1 if correlation > 0 else 0)
    return bits
