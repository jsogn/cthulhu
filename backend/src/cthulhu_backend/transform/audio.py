"""音频重混：EQ 倾斜、响度归一与可选微变速，配合画面脱敏。

生产管线只使用等长重混（EQ/变调/噪声），不改变音轨长度与内容时间轴，
保证音画同步；研究 harness 可显式传入 speed_factor/tempo 做变速实验。
"""

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
    speed_factor: float | None = None,
) -> np.ndarray:
    """等长重混链：EQ 倾斜 → 响度归一 → 微量噪声。

    speed_factor 仅在研究 harness 显式传入时做微变速；该操作会截断/补零
    改变内容时间线，可能造成音画漂移，生产管线不传。
    """
    out = eq_tilt(signal, sample_rate, rng)
    if speed_factor is not None:
        stretched = resample_poly(out, 1000, max(1, round(1000 * speed_factor)))
        n = len(signal)
        out = stretched[:n] if len(stretched) >= n else np.pad(stretched, (0, n - len(stretched)))
    out = normalize_loudness(out)
    return out + 0.002 * rng.standard_normal(len(signal))


def pitch_shift(signal: np.ndarray, ratio: float) -> np.ndarray:
    """变调不变速：先按 ratio 拉伸，再拉回原长。"""
    n = len(signal)
    stretched = resample_poly(signal, 1000, max(1, round(1000 * ratio)))
    return resample_poly(stretched, n, max(1, len(stretched)))


def remix_strong(
    signal: np.ndarray,
    sample_rate: int,
    rng: np.random.Generator,
    tempo: float | None = None,
    pitch_ratio: float = 0.98,
    eq_db: float = 6.0,
    noise_floor: float = 0.004,
) -> np.ndarray:
    """强音频重混：变调 + 强 EQ 倾斜 + 底噪，破坏 chromaprint/梅尔谱指纹。

    与 remix 的轻量频谱处理不同，本链针对音频指纹的时间-频率对齐做双重扰动。
    变调采用「先拉伸再拉回原长」的等长实现，不改变内容时间轴；
    tempo 仅在研究 harness 显式传入时做变速（会破坏音画同步，生产不传）。
    """
    out = eq_tilt(signal, sample_rate, rng, gain_db=eq_db)
    out = pitch_shift(out, pitch_ratio)
    if tempo is not None:
        stretched = resample_poly(out, 1000, max(1, round(1000 * tempo)))
        n = len(signal)
        out = stretched[:n] if len(stretched) >= n else np.pad(stretched, (0, n - len(stretched)))
    out = normalize_loudness(out)
    return out + noise_floor * rng.standard_normal(len(signal))
