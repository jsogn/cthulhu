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


def pitch_shift(signal: np.ndarray, ratio: float) -> np.ndarray:
    """变调不变速：先按 ratio 拉伸，再拉回原长。"""
    n = len(signal)
    stretched = resample_poly(signal, 1000, max(1, round(1000 * ratio)))
    return resample_poly(stretched, n, max(1, len(stretched)))


def remix_strong(
    signal: np.ndarray,
    sample_rate: int,
    rng: np.random.Generator,
    tempo: float = 1.05,
    pitch_ratio: float = 0.98,
    eq_db: float = 6.0,
    noise_floor: float = 0.004,
) -> np.ndarray:
    """强音频重混：变速 + 变调 + 强 EQ 倾斜 + 底噪，破坏 chromaprint/梅尔谱指纹。

    与 remix 的轻量频谱处理不同，本链针对音频指纹的时间-频率对齐做双重扰动。
    """
    out = eq_tilt(signal, sample_rate, rng, gain_db=eq_db)
    out = pitch_shift(out, pitch_ratio)
    n = len(signal)
    stretched = resample_poly(out, 1000, max(1, round(1000 * tempo)))
    out = stretched[:n] if len(stretched) >= n else np.pad(stretched, (0, n - len(stretched)))
    out = normalize_loudness(out)
    return out + noise_floor * rng.standard_normal(n)
