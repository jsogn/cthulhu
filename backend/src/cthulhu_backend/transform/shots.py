"""分镜检测：基于相邻帧差分定位硬切点。"""

from __future__ import annotations

from itertools import pairwise

import numpy as np


def detect_cuts(
    frames: np.ndarray,
    threshold: float = 0.18,
    min_shot_length: int = 2,
) -> list[int]:
    """返回镜头边界（含首尾）：[0, cut1, cut2, ..., len(frames)]。"""
    arr = np.asarray(frames)
    if len(arr) < 2:
        return [0, len(arr)]
    # 逐对差分，避免为整段采样帧分配两份全尺寸临时数组。
    diffs = np.empty(len(arr) - 1, dtype=np.float32)
    for index in range(len(arr) - 1):
        diffs[index] = float(np.mean(np.abs(arr[index + 1] - arr[index])))
    cuts = [i + 1 for i, value in enumerate(diffs) if value > threshold]
    # 合并过近的切点，保证最小镜头长度。
    merged: list[int] = []
    for cut in cuts:
        if not merged or cut - merged[-1] >= min_shot_length:
            merged.append(cut)
    return [0] + merged + [len(arr)]


def split_shots(frames: np.ndarray, boundaries: list[int]) -> list[np.ndarray]:
    return [frames[a:b] for a, b in pairwise(boundaries)]
