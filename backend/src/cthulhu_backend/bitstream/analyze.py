"""组合压缩域分析：QP + 码量 + SEI + 容器。

评分分两种模式：
- 无基准：绝对阈值启发式，周期性可能来自内容本身，仅作提示；
- 有基准（推荐）：与干净参照的差分判定，周期性超出基准的幅度才是异常信号。
"""

from __future__ import annotations

from cthulhu_backend.bitstream import frames as frames_module
from cthulhu_backend.bitstream import qp as qp_module
from cthulhu_backend.media import container, ffmpeg


def _features(
    path: str,
    max_frames: int | None = None,
    scan: dict | None = None,
    sei_count: int | None = None,
) -> tuple[dict, dict, dict, int | None]:
    qp_feat = qp_module.qp_features(qp_module.extract_qp_maps(path, max_frames) or [])
    alloc = frames_module.frame_allocation(path)
    alloc_feat = frames_module.allocation_features(*alloc) if alloc else {"frame_packets": 0}
    scan = scan if scan is not None else container.scan_mp4(path)
    sei = sei_count if sei_count is not None else container.count_sei(path)
    return qp_feat, alloc_feat, scan, sei


def _level(score: int) -> str:
    return "低" if score < 30 else "中" if score < 60 else "高"


def analyze(
    path: str,
    reference: str | None = None,
    max_frames: int | None = None,
    scan: dict | None = None,
    sei_count: int | None = None,
) -> dict:
    """码流层分析。带 reference 时输出差分评分（研究口径，不替代平台实测）。"""
    info = ffmpeg.video_info(path)
    qp_feat, alloc_feat, scan, sei = _features(
        path, max_frames, scan=scan, sei_count=sei_count
    )

    base = {
        "file": path,
        "codec": info["codec"],
        "resolution": f"{info['width']}x{info['height']}",
        "qp": qp_feat,
        "allocation": alloc_feat,
        "sei_count": sei,
        "suspicious_boxes": scan.get("suspicious", []),
        "heuristic": True,
    }

    if reference:
        ref_qp, ref_alloc, _, _ = _features(reference, max_frames)
        delta_qp = float(qp_feat.get("qp_temporal_periodicity") or 0.0) - float(
            ref_qp.get("qp_temporal_periodicity") or 0.0
        )
        delta_size = float(alloc_feat.get("size_periodicity") or 0.0) - float(
            ref_alloc.get("size_periodicity") or 0.0
        )
        score, flags = 0, []
        if delta_qp > 0.15:
            score += 40
            flags.append(f"QP 周期性高于基准 +{delta_qp:.2f}")
        if delta_size > 0.15:
            score += 40
            flags.append(f"码量周期性高于基准 +{delta_size:.2f}")
        if (sei or 0) >= 3:
            score += 10
            flags.append(f"SEI 数量偏多（{sei}）")
        if scan.get("has_xmp"):
            score += 10
            flags.append("存在 XMP 元数据")
        return {
            **base,
            "reference": reference,
            "delta_qp_periodicity": round(delta_qp, 4),
            "delta_size_periodicity": round(delta_size, 4),
            "flags": flags,
            "score": min(score, 100),
            "level": _level(score),
        }

    qp_period = float(qp_feat.get("qp_temporal_periodicity") or 0.0)
    size_period = float(alloc_feat.get("size_periodicity") or 0.0)
    score, flags = 0, []
    if qp_period > 0.45:
        score += 25
        flags.append(f"QP 序列周期性较强（{qp_period:.2f}）")
    if size_period > 0.55:
        score += 25
        flags.append(f"码量分配周期性较强（{size_period:.2f}）")
    if (sei or 0) >= 3:
        score += 25
        flags.append(f"SEI 数量偏多（{sei}）")
    if scan.get("has_xmp"):
        score += 25
        flags.append("存在 XMP 元数据")
    return {
        **base,
        "reference": None,
        "flags": flags,
        "score": min(score, 100),
        "level": _level(score),
        "note": "未提供干净基准（--reference），周期性可能来自内容本身，建议对照后再判定",
    }
