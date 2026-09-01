"""DFT 域扩频水印：在中频环带对复数系数做加性扩频嵌入。

与空域扩频互补：信号落在 DFT 中频，对空域滤波更鲁棒；作为基准库新方案族，
检验清洗管线对 DFT 域的覆盖（FFT 相位/幅度攻击的本地验收靶）。
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from cthulhu_backend.watermark.common import with_sync


def _band_mask(shape: tuple[int, int], inner: float = 0.10, outer: float = 0.45) -> np.ndarray:
    h, w = shape
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    radius = np.sqrt(fy**2 + fx**2)
    return (radius >= inner) & (radius <= outer)


def _pattern(shape: tuple[int, int], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pattern = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    return pattern / np.sqrt(np.mean(np.abs(pattern) ** 2) + 1e-12)


def embed(frame: np.ndarray, bits: list[int], seed: int, alpha: float = 4.0) -> np.ndarray:
    bits = with_sync(bits)
    spectrum = np.fft.rfft2(frame)
    mask = _band_mask(spectrum.shape)
    pattern = _pattern(spectrum.shape, seed)
    rows_per_bit = max(1, spectrum.shape[0] // len(bits))
    for index, bit in enumerate(bits):
        r0, r1 = index * rows_per_bit, (index + 1) * rows_per_bit
        region = spectrum[r0:r1]
        sign = 1.0 if bit else -1.0
        band = mask[r0:r1]
        if band.any():
            region[band] += alpha * sign * pattern[r0:r1][band]
    return np.clip(np.fft.irfft2(spectrum, s=frame.shape), 0, 1)


def extract(frame: np.ndarray, n_bits: int, seed: int) -> list[int]:
    spectrum = np.fft.rfft2(frame)
    residual = spectrum - gaussian_filter(spectrum, sigma=1.0, mode="reflect")
    mask = _band_mask(spectrum.shape)
    pattern = _pattern(spectrum.shape, seed)
    total = len(with_sync([0] * n_bits))
    rows_per_bit = max(1, spectrum.shape[0] // total)
    bits = []
    for index in range(total):
        r0, r1 = index * rows_per_bit, (index + 1) * rows_per_bit
        band = mask[r0:r1]
        if not band.any():
            bits.append(0)
            continue
        corr = np.real(np.mean(residual[r0:r1][band] * np.conj(pattern[r0:r1][band])))
        bits.append(1 if corr > 0 else 0)
    return bits
