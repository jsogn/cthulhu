"""码流特征统计：归一化熵与序列自相关峰值。"""

from __future__ import annotations

import numpy as np


def normalized_entropy(values: np.ndarray) -> float:
    """离散值的归一化信息熵（0~1），用于度量 QP 图的空间复杂度。"""
    arr = np.asarray(values).ravel()
    if arr.size < 2:
        return 0.0
    ints = arr.astype(int)
    counts = np.bincount(ints - int(ints.min()), minlength=int(ints.max() - ints.min()) + 1)
    probs = counts[counts > 0] / counts.sum()
    entropy = float(-np.sum(probs * np.log(probs)))
    return entropy / np.log(max(2, len(probs)))


def autocorr_peak(sequence: list[float] | np.ndarray, min_lag: int = 1) -> float:
    """去均值后扫描滞后自相关的最大峰值（0~1），用于发现周期性调制。"""
    x = np.asarray(sequence, dtype=np.float64)
    if len(x) < 8 or float(np.std(x)) == 0.0:
        return 0.0
    x = x - x.mean()
    best = 0.0
    for lag in range(min_lag, max(min_lag + 1, len(x) // 2)):
        left, right = x[:-lag], x[lag:]
        denom = float(np.sqrt(np.dot(left, left) * np.dot(right, right)))
        value = float(np.dot(left, right)) / denom if denom > 0 else 0.0
        best = max(best, value)
    return best
