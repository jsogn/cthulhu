"""DCT 重量化的原生加速入口（可选，默认关闭）。

加载失败、未启用（未设置 CTHULHU_NATIVE_DCT）或宽高不是 8 的整数倍时
返回 None，调用方回退 numpy 实现，保证行为永远有兜底。
"""

from __future__ import annotations

import ctypes
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

_LIB: ctypes.CDLL | None | bool = False  # False=尚未尝试，None=不可用


def _load_lib() -> ctypes.CDLL | None:
    global _LIB
    if _LIB is not False:
        return _LIB
    bundled = str(Path(__file__).resolve().parent / "native" / "libdct_requant.dylib")
    candidate = os.environ.get("CTHULHU_NATIVE_DCT_LIB") or bundled
    lib = None
    try:
        lib = ctypes.CDLL(candidate)
        fn = lib.dct_requant_plane
        fn.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
        ]
        fn.restype = None
    except OSError:
        lib = None
    _LIB = lib
    return lib


def requant_plane(work: np.ndarray, step: float) -> np.ndarray | None:
    """等价于 extra_attacks.dct_requant_plane 的原生实现；不可用时返回 None。"""
    if not os.environ.get("CTHULHU_NATIVE_DCT"):
        return None
    lib = _load_lib()
    if lib is None:
        return None
    h, w = work.shape[1:3]
    if h % 8 or w % 8 or work.dtype != np.float32:
        return None
    src = np.ascontiguousarray(work, dtype=np.float32)
    out = np.empty_like(src)
    fn = lib.dct_requant_plane
    base_src, base_dst = src.ctypes.data, out.ctypes.data
    frame_bytes = h * w * 4
    workers = max(1, min(4, len(src)))

    def call(index: int) -> None:
        fn(
            ctypes.cast(base_src + index * frame_bytes, ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(base_dst + index * frame_bytes, ctypes.POINTER(ctypes.c_float)),
            h,
            w,
            float(step),
        )

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(call, range(len(src))))
    else:
        for index in range(len(src)):
            call(index)
    return out
