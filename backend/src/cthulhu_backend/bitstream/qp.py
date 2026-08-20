"""从 ffmpeg -debug qp 输出解析逐宏块 QP 图。"""

from __future__ import annotations

import re
import subprocess

import numpy as np

from cthulhu_backend.bitstream.stats import autocorr_peak, normalized_entropy
from cthulhu_backend.media import ffmpeg

FRAME_RE = re.compile(r"New frame, type: ([IPB])")
ROW_RE = re.compile(r"^\s*(\d+)\s+([0-9 ]+)\s*$")


def extract_qp_maps(path: str, max_frames: int | None = None) -> list[dict] | None:
    """返回每帧 QP 图：{"type": "I/P/B", "rows": [[qp, ...], ...]}；不可用时返回 None。"""
    if not ffmpeg.has_ffmpeg():
        return None
    cmd = ["ffmpeg", "-v", "debug", "-threads", "1", "-debug", "qp", "-i", path]
    if max_frames:
        cmd += ["-frames:v", str(max_frames)]
    cmd += ["-f", "null", "-"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    frames: list[dict] = []
    current: dict | None = None
    current_rows: dict[int, list[int]] = {}

    def finalize() -> None:
        nonlocal current, current_rows
        if current is None:
            return
        # 按行号重建：ffmpeg 偶发漏输出个别宏块行，用上一行填充保证帧内行数稳定。
        if current_rows:
            keys = sorted(current_rows)
            ordered: list[list[int]] = []
            last = current_rows[keys[0]]
            for row_no in range(keys[0], keys[-1] + 1):
                row = current_rows.get(row_no, last)
                ordered.append(row)
                last = row
            current["rows"] = ordered
        current_rows = {}
        frames.append(current)

    for line in result.stderr.splitlines():
        if FRAME_RE.search(line):
            finalize()
            current = {"type": FRAME_RE.search(line).group(1), "rows": []}  # type: ignore[union-attr]
            continue
        if current is None:
            continue
        remainder = line.split("] ", 1)[-1]
        match = ROW_RE.match(remainder)
        if not match:
            continue
        digits = "".join(match.group(2).split())
        if not digits or len(digits) % 2 != 0:
            continue
        qps = [int(digits[i : i + 2]) for i in range(0, len(digits), 2)]
        if not all(0 <= q <= 51 for q in qps):
            continue
        current_rows[int(match.group(1))] = qps
    finalize()
    return frames or None


def qp_features(frames: list[dict]) -> dict:
    """QP 图统计：均值序列、空间熵、时间周期性与块级熵异常。"""
    # 真实视频的 QP 行数偶发不等长，逐行拼接避免 inhomogeneous 报错。
    maps: list[np.ndarray] = []
    for frame in frames:
        rows = frame.get("rows")
        if not rows:
            continue
        maps.append(
            np.concatenate([np.asarray(row, dtype=np.float64).ravel() for row in rows]),
        )
    if not maps:
        return {"qp_frames": 0}
    means = [float(m.mean()) for m in maps]
    entropies = [normalized_entropy(m) for m in maps]
    return {
        "qp_frames": len(maps),
        "qp_mean": round(float(np.mean(means)), 4),
        "qp_std": round(float(np.std(means)), 4),
        "qp_spatial_entropy": round(float(np.mean(entropies)), 4),
        "qp_temporal_periodicity": round(autocorr_peak(means), 4),
    }
