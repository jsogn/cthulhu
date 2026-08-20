"""音频回声隐藏水印：不同回声延迟编码比特，倒谱域检测。"""

from __future__ import annotations

import numpy as np

from cthulhu_backend.watermark.common import with_sync


def embed(
    signal: np.ndarray,
    bits: list[int],
    sample_rate: int,
    alpha: float = 0.45,
    delay0: int = 80,
    delay1: int = 120,
    segment: float = 0.25,
) -> np.ndarray:
    bits = with_sync(bits)
    seg_len = int(sample_rate * segment)
    n = min(len(bits), len(signal) // seg_len)
    out = signal.copy()
    fade = int(sample_rate * 0.02)
    for i in range(n):
        start = i * seg_len
        delay = delay0 if bits[i] == 0 else delay1
        echo = np.zeros_like(signal)
        available = min(seg_len, len(signal) - (start + delay))
        if available <= 0:
            break
        echo[start + delay : start + delay + available] = signal[start : start + available]
        fade_len = min(fade, available)
        echo[start + delay : start + delay + fade_len] *= np.linspace(0, 1, fade_len)
        out[start : start + available] += alpha * echo[start : start + available]
    return np.clip(out, -1, 1)


def extract(
    signal: np.ndarray,
    n_bits: int,
    sample_rate: int,
    delay0: int = 80,
    delay1: int = 120,
    segment: float = 0.25,
) -> list[int]:
    seg_len = int(sample_rate * segment)
    total = len(with_sync([0] * n_bits))
    n = min(total, len(signal) // seg_len)
    bits = []
    eps = 1e-12
    for i in range(n):
        seg = signal[i * seg_len : (i + 1) * seg_len]
        spec = np.abs(np.fft.rfft(seg * np.hanning(seg_len))) ** 2
        cep = np.fft.irfft(np.log(spec + eps), n=seg_len).real
        bits.append(1 if cep[delay1] > cep[delay0] else 0)
    return bits
