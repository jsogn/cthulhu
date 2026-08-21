"""数值归约安全工具。

本机实测（numpy 2.4.3~2.5.2，macOS arm64 / Accelerate）：对 (N, C) 且 C≥3
的 float32 数组沿 N 轴（约 2^25 个元素以上）做 mean/sum 时，numpy 走朴素
逐元素求和而非成对求和，累加器在 2^24 处饱和，返回严重错误的结果
（如均值 0.5 变成 0.25/0.125）。1D 连续数组与 float64 输入不受影响。

规避方式：逐帧/逐块小轴归约（尺寸安全），累加阶段显式用 float64。
"""

from __future__ import annotations

import numpy as np


def channel_stats(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """逐通道均值与总体标准差（float64 累加，规避大轴 float32 归约饱和）。

    frames 形状为 (F, H, W) 或 (F, H, W, C)；返回 (mean, std)，长度等于
    通道数。均值按逐帧平均再平均（数学上等价），方差用 E[x²]-E[x]² 合并。
    """
    array = np.asarray(frames)
    channels = 1 if array.ndim == 3 else array.shape[-1]
    per_frame = array.reshape(len(array), -1, channels)
    means = per_frame.mean(axis=1, dtype=np.float64)  # (F, C)，float64 累加
    squares = np.square(per_frame).mean(axis=1, dtype=np.float64)
    mean = means.mean(axis=0)
    variance = np.maximum(squares.mean(axis=0) - mean * mean, 0.0)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)
