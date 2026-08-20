"""帧级变换的线程并行工具。

numpy/scipy 的大块运算会释放 GIL，逐帧并发用线程池即可提速，
无需跨进程搬运整段帧数据。线程数可通过 CTHULHU_TRANSFORM_THREADS 覆盖。
"""

from __future__ import annotations

import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

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
