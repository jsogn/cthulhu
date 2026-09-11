"""任务进度换算：把批次内进度映射成整片百分比（纯逻辑，可独立回归）。

原先以私有名嵌在 `pipeline` 里；提为公开模块后，编码管线只负责把回调接到
控制面，进度口径本身在这里定义与测试。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# 进度条区间：分析遍 8% 起步，净化阶段占 82%，收尾各留固定余量。
PROGRESS_BASE = 8
PROGRESS_SPAN = 82
PROGRESS_MAX = 90


def purify_progress(
    base: int,
    count: int,
    fraction: float,
    total_out: int,
) -> tuple[int, str]:
    """把批次内进度换算成整段视频的全局帧数与百分比。"""
    total = max(1, int(total_out))
    bounded = max(0.0, min(1.0, float(fraction)))
    processed = round(base + bounded * max(1, count))
    processed = max(0, min(total, processed))
    percent = PROGRESS_BASE + int(processed / total * PROGRESS_SPAN)
    return min(PROGRESS_MAX, percent), f"潜空间净化 {processed}/{total}"


@dataclass
class PurifyProgress:
    """净化进度桥：批次内进度换算成整片百分比（线程本地控制面，不跨进程）。"""

    progress_cb: Any
    total_out: int
    base: int = 0
    count: int = 1

    def set_batch(self, base: int, count: int) -> None:
        self.base = base
        self.count = count

    def __call__(self, fraction: float, note: str) -> None:
        if not self.progress_cb or not self.total_out:
            return
        if note.startswith("下载"):
            percent = PROGRESS_BASE + int(self.base / self.total_out * PROGRESS_SPAN)
            self.progress_cb(min(PROGRESS_MAX, percent), note)
            return
        percent, global_note = purify_progress(
            self.base, self.count, fraction, self.total_out
        )
        self.progress_cb(percent, global_note)
