"""从 ffmpeg -debug qp 输出解析逐宏块 QP 图。"""

from __future__ import annotations

import re
import subprocess

import numpy as np

from cthulhu_backend.bitstream.stats import autocorr_peak, normalized_entropy
from cthulhu_backend.media import ffmpeg

FRAME_RE = re.compile(r"New frame, type: ([IPB])")
NUMBER_RE = re.compile(r"\d+")


def extract_qp_maps(path: str, max_frames: int | None = None) -> list[dict] | None:
    """返回每帧 QP 图：{"type": "I/P/B", "rows": [[qp, ...], ...]}；不可用时返回 None。"""
    if not ffmpeg.has_ffmpeg():
        return None
    cmd = [ffmpeg.FFMPEG_BIN, "-v", "debug", "-threads", "1", "-debug", "qp", "-i", path]
    if max_frames:
        cmd += ["-frames:v", str(max_frames)]
    cmd += ["-f", "null", "-"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return _parse_debug_output(result.stderr)


def _parse_debug_output(text: str) -> list[dict] | None:
    """解析 `ffmpeg -v debug -debug qp` 的 stderr 输出。"""
    frames: list[dict] = []
    current: dict | None = None
    current_lines: list[tuple[int, list[int]]] = []
    current_new_style = False

    def finalize() -> None:
        nonlocal current, current_lines, current_new_style
        if current is None:
            return
        if current_lines:
            current["rows"] = _decode_rows(current_lines, current_new_style)
        current_lines = []
        current_new_style = False
        frames.append(current)

    def parse_qp_fields(tokens: list[int]) -> list[int] | None:
        """把 QP 值还原为整数列表；格式为 %2d 紧凑输出，需按两位切分。"""
        digits = "".join(str(token) for token in tokens)
        if not digits or len(digits) % 2 != 0:
            return None
        qps = [int(digits[i : i + 2]) for i in range(0, len(digits), 2)]
        if not all(0 <= q <= 51 for q in qps):
            return None
        return qps

    for line in text.splitlines():
        if FRAME_RE.search(line):
            finalize()
            current = {"type": FRAME_RE.search(line).group(1), "rows": []}  # type: ignore[union-attr]
            continue
        if current is None:
            continue
        if "] " not in line:
            continue
        remainder = line.split("] ", 1)[-1]
        # QP 行只含数字与空白；其余调试信息（nal 头、线程等）直接忽略。
        if re.fullmatch(r"[\d\s]+", remainder) is None:
            continue
        tokens = [int(token) for token in NUMBER_RE.findall(remainder)]
        if not tokens:
            continue
        # 新格式（FFmpeg ≥7）每帧在 QP 行之前会打印 x 坐标表头：从 0 开始、
        # 均为 16 的倍数且严格递增（如 0 64 128 …）。
        if (
            tokens[0] == 0
            and all(value % 16 == 0 for value in tokens)
            and np.all(np.diff(tokens) > 0)
        ):
            current_new_style = True
            continue
        # 新格式行首是宏块行号；旧格式整行都是 QP 值。
        row_index = 0
        if current_new_style:
            if len(tokens) < 2:
                continue
            row_index, *tokens = tokens
        qps = parse_qp_fields(tokens)
        if not qps:
            continue
        current_lines.append((row_index, qps))
    finalize()
    return frames or None


def _decode_rows(lines: list[tuple[int, list[int]]], new_style: bool) -> list[list[int]]:
    """把一帧内的 QP 行还原为二维列表，兼容新旧两种 `-debug qp` 输出格式。

    旧格式（FFmpeg ≤6.1）每行只有 QP 值、没有行号；新格式（FFmpeg ≥7）每行以
    宏块行号（0/16/32/…）开头，且帧首有 x 坐标表头。由调用方依据表头判定格式。
    """
    if not new_style:
        return [list(row) for _, row in lines]
    rows_by_index: dict[int, list[int]] = {index: row for index, row in lines}
    keys = sorted(rows_by_index)
    # ffmpeg 偶发漏输出个别宏块行，用上一行填充保证帧内行数稳定。
    ordered: list[list[int]] = []
    last = rows_by_index[keys[0]]
    for row_no in range(keys[0], keys[-1] + 1):
        row = rows_by_index.get(row_no, last)
        ordered.append(row)
        last = row
    return ordered


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
