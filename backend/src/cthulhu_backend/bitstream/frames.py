"""逐帧码量分配：来自 ffprobe 的 packet 大小与关键帧标志。"""

from __future__ import annotations

import json
import subprocess
from itertools import pairwise

import numpy as np

from cthulhu_backend.bitstream.stats import autocorr_peak


def frame_allocation(path: str) -> tuple[list[int], list[bool]] | None:
    """返回 (帧大小列表, 关键帧标志列表)；不可用时返回 None。"""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "packet=pts_time,size,flags",
            "-of", "json", path,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    try:
        packets = json.loads(result.stdout)["packets"]
    except (ValueError, KeyError):
        return None
    sizes = [int(p["size"]) for p in packets]
    keys = [p.get("flags", "").startswith("K") for p in packets]
    return sizes, keys


def allocation_features(sizes: list[int], keys: list[bool]) -> dict:
    """码量统计：分布、I 帧周期、P 帧大小的周期性。"""
    if not sizes:
        return {"frame_packets": 0}
    arr = np.asarray(sizes, dtype=np.float64)
    p_sizes = [s for s, k in zip(sizes, keys, strict=True) if not k]
    key_index = [i for i, k in enumerate(keys) if k]
    periods = [b - a for a, b in pairwise(key_index)] if len(key_index) > 1 else []
    return {
        "frame_packets": len(sizes),
        "size_mean": float(arr.mean()),
        "size_std": float(arr.std()),
        "size_periodicity": autocorr_peak(p_sizes),
        "i_frame_period": float(np.median(periods)) if periods else 0.0,
        "i_frame_count": len(key_index),
    }
