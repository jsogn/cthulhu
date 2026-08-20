"""水印公共工具：比特序列、同步码、误码率与对齐。"""

from __future__ import annotations

import numpy as np

# 同步头：嵌入在载荷前，用于攻击后的比特流对齐。
SYNC = [1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0]


def payload_bits(seed: int, n_bits: int = 64) -> list[int]:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=n_bits).tolist()


def with_sync(bits: list[int]) -> list[int]:
    return SYNC + bits


def bit_error_rate(ref: list[int], out: list[int]) -> float:
    """按参考长度对齐后计算误码率（找不到同步头时返回 1.0）。"""
    n = len(ref)
    if n == 0 or len(out) < n:
        return 1.0
    best = min(
        np.mean(np.asarray(ref) != np.asarray(out[shift : shift + n]))
        for shift in range(len(out) - n + 1)
    )
    return float(best)
