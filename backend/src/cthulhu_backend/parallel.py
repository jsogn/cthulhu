"""帧级变换的线程并行工具。

numpy/scipy 的大块运算会释放 GIL，逐帧并发用线程池即可提速，
无需跨进程搬运整段帧数据。线程数可通过 CTHULHU_TRANSFORM_THREADS 覆盖。
"""

from __future__ import annotations

import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from itertools import pairwise

import numpy as np

_DEFAULT_WORKERS = min(8, os.cpu_count() or 1)


def default_workers() -> int:
    """默认并行线程数：环境变量优先，否则取 min(8, CPU 核数)。"""
    raw = os.environ.get("CTHULHU_TRANSFORM_THREADS")
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return _DEFAULT_WORKERS


def map_frames(
    fn: Callable[[np.ndarray], np.ndarray],
    frames: np.ndarray,
    workers: int | None = None,
) -> np.ndarray:
    """对每一帧并行应用 fn，按原顺序拼回帧数组；帧数过少时顺序执行。"""
    count = len(frames)
    workers = workers or default_workers()
    if workers <= 1 or count < 8:
        return np.stack([fn(frame) for frame in frames])
    with ThreadPoolExecutor(max_workers=min(workers, count)) as pool:
        return np.stack(list(pool.map(fn, frames)))


def map_chunks(
    frames: np.ndarray,
    fn: Callable[[np.ndarray], np.ndarray],
    workers: int | None = None,
) -> np.ndarray:
    """把帧数组按帧轴切成连续块并行处理，再按原顺序拼回。

    适用于「整批向量化但只跑单核」的阶段（重量化/色度/均值校正等）；
    每块内部仍走 numpy 向量化，块间用线程池重叠，结果与顺序执行逐位一致。
    """
    count = len(frames)
    workers = workers or default_workers()
    workers = max(1, min(workers, count))
    if workers <= 1 or count < 8:
        return fn(frames)
    bounds = np.linspace(0, count, workers + 1).astype(int)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        parts = list(pool.map(lambda pair: fn(frames[pair[0] : pair[1]]), pairwise(bounds)))
    return np.concatenate(parts, axis=0)


def map_items(
    fn: Callable[[object], object],
    items: list[object],
    workers: int | None = None,
) -> list[object]:
    """对独立项并行执行 fn，按原顺序返回结果；项数过少时顺序执行。

    适用于互不共享可变随机状态的评估组合（如方法×攻击×种子矩阵），
    结果与顺序执行逐位一致。
    """
    workers = workers or default_workers()
    if workers <= 1 or len(items) < 2:
        return [fn(item) for item in items]
    with ThreadPoolExecutor(max_workers=min(workers, len(items))) as pool:
        return list(pool.map(fn, items))
