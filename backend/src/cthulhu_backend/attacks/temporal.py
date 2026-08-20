"""时序攻击：抽帧替换（视频）、变速重采样（音频）。"""

from __future__ import annotations

import numpy as np
from scipy.signal import resample_poly


def drop_duplicate(frames: np.ndarray, every: int = 10) -> np.ndarray:
    """每隔 every 帧用前一帧替换，破坏帧间同步结构，保持长度不变。"""
    out = frames.copy()
    for i in range(every, len(frames), every):
        out[i] = out[i - 1]
    return out


def speed_change_audio(signal: np.ndarray, factor: float = 0.99) -> np.ndarray:
    """0.99 倍变速重采样后截回原长，破坏回声延迟与相位结构。"""
    stretched = resample_poly(signal, 1000, round(1000 * factor))
    n = len(signal)
    if len(stretched) >= n:
        return stretched[:n]
    return np.pad(stretched, (0, n - len(stretched)), mode="constant")
