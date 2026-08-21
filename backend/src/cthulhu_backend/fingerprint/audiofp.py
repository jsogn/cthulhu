"""音频指纹代理：梅尔谱哈希 + MFCC 结构哈希（Chromaprint 的近似）。

真实平台多使用鲁棒音频指纹（如 Chromaprint/AcoustID 或自研神经网络指纹），
本模块用「对数梅尔谱块均值」与「MFCC 块均值」两种可复现代理逼近其低频
鲁棒性：对音量变化不敏感，对变速/变调部分敏感。
"""

from __future__ import annotations

import numpy as np
from scipy.fft import dct, rfft

from cthulhu_backend.fingerprint.hashes import block_mean


def _frame_signal(signal: np.ndarray, sample_rate: int, frame: float = 0.025, hop: float = 0.01):
    """分帧加汉宁窗，返回 (帧数, 帧长) 的 float64 矩阵。"""
    window_len = int(sample_rate * frame)
    hop_len = int(sample_rate * hop)
    if len(signal) < window_len:
        return np.empty((0, window_len))
    count = 1 + (len(signal) - window_len) // hop_len
    indices = np.arange(window_len)[None, :] + hop_len * np.arange(count)[:, None]
    window = np.hanning(window_len)[None, :]
    return np.asarray(signal, dtype=np.float64)[indices] * window


def mel_filterbank(sample_rate: int, n_fft: int, n_mels: int = 40, fmin: float = 100.0) -> np.ndarray:
    """三角梅尔滤波器组，返回 (n_mels, n_fft//2+1)。"""
    fmax = sample_rate / 2
    mel_points = np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    bank = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(n_mels):
        left, center, right = bins[i], bins[i + 1], bins[i + 2]
        for j in range(left, center):
            bank[i, j] = (j - left) / max(center - left, 1)
        for j in range(center, right):
            bank[i, j] = (right - j) / max(right - center, 1)
    return bank


def _hz_to_mel(hz: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def _mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


def log_mel_spectrogram(signal: np.ndarray, sample_rate: int, n_mels: int = 40) -> np.ndarray:
    """对数梅尔谱：帧数 × n_mels。"""
    frames = _frame_signal(signal, sample_rate)
    if frames.shape[0] == 0:
        return np.empty((0, n_mels))
    n_fft = frames.shape[1]
    spec = np.abs(rfft(frames, axis=1))
    bank = mel_filterbank(sample_rate, n_fft, n_mels)
    mel = spec @ bank.T
    return np.log(np.maximum(mel, 1e-8))


def mel_hash(signal: np.ndarray, sample_rate: int, size: int = 32) -> np.ndarray:
    """对数梅尔谱块均值哈希：32×32 二值签名。"""
    mel = log_mel_spectrogram(signal, sample_rate)
    if mel.shape[0] < 4:
        return np.zeros(size * size, dtype=bool)
    small = block_mean(mel, size)
    return small.ravel() > np.median(small)


def mfcc_hash(signal: np.ndarray, sample_rate: int, size: int = 24, coefficients: int = 12) -> np.ndarray:
    """MFCC 结构哈希：前 12 维倒谱系数块均值二值化。"""
    mel = log_mel_spectrogram(signal, sample_rate)
    if mel.shape[0] < 4:
        return np.zeros(size * coefficients, dtype=bool)
    mfcc = dct(mel, axis=1, norm="ortho")[:, :coefficients]
    height, _ = mfcc.shape
    pad = (height // size + 1) * size
    padded = np.pad(mfcc, ((0, pad - height), (0, 0)), mode="edge")
    blocks = padded.reshape(size, pad // size, coefficients).mean(axis=1)
    return blocks.ravel() > np.median(blocks)


def hamming_bits(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(np.asarray(a) != np.asarray(b)))
