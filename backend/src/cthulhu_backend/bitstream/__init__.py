"""压缩域（码流层）检测：QP 图、码量分配、GOP 与启发式异常评分。"""

from cthulhu_backend.bitstream import analyze, frames, qp, stats

__all__ = ["analyze", "frames", "qp", "stats"]
