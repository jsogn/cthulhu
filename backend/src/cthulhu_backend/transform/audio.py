"""音频重混：EQ 倾斜、响度归一与微变速，配合画面脱敏。"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, lfilter, resample_poly


def eq_tilt(signal: np.ndarray, sample_rate: int, rng: np.random.Generator, gain_db: float = 3.0) -> np.ndarray:
    """一阶高/低频倾斜滤波，方向随机。"""
    direction = 1.0 if rng.random() < 0.5 else -1.0
    cutoff = 1000.0
    if direction > 0:
        b, a = butter(1, cutoff / (sample_rate / 2), btype="low")
        low = lfilter(b, a, signal)
        high = signal - low
        out = low * 10 ** (gain_db / 20) + high
    else:
        b, a = butter(1, cutoff / (sample_rate / 2), btype="high")
        high = lfilter(b, a, signal)
        low = signal - high
        out = high * 10 ** (gain_db / 20) + low
    return out


def normalize_loudness(signal: np.ndarray, target_peak: float = 0.25) -> np.ndarray:
    peak = float(np.max(np.abs(signal))) or 1.0
    return signal * (target_peak / peak)


def remix(
    signal: np.ndarray,
    sample_rate: int,
    rng: np.random.Generator,
    speed_factor: float = 0.99,
) -> np.ndarray:
    """重混链：EQ 倾斜 → 微变速 → 响度归一 → 微量噪声。"""
    out = eq_tilt(signal, sample_rate, rng)
    stretched = resample_poly(out, 1000, round(1000 * speed_factor))
    n = len(signal)
    out = stretched[:n] if len(stretched) >= n else np.pad(stretched, (0, n - len(stretched)))
    out = normalize_loudness(out)
    return out + 0.002 * rng.standard_normal(n)
