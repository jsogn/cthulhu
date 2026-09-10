"""内容脱敏评估：时序乱序度指标与评估矩阵。"""

from __future__ import annotations

import numpy as np

from cthulhu_backend import samples
from cthulhu_backend.evaluate import harness, metrics
from cthulhu_backend.transform import shots
from cthulhu_backend.transform import video as video_transform


def test_temporal_match_uses_proportional_mapping_on_static_content():
    """慢速/静态素材上纯内容 argmax 会整体塌缩到同一帧（VMAF 直接变 0）。

    这类内容相邻帧几乎一样，内容相似度无法分辨真值，必须退化为等比例映射。
    """
    base = np.linspace(0.0, 1.0, 60).reshape(60, 1, 1) * np.ones((60, 16, 16))
    processed = np.concatenate([base, base[-1:]])  # 61 帧：约 1.7% 变慢
    ref, mov, matches = metrics.temporal_match(base, processed)
    proportional = np.rint(np.linspace(0, len(ref) - 1, len(mov))).astype(int)
    assert np.array_equal(matches, proportional)


def test_temporal_match_recovers_reordered_shots():
    """段级重排仍要按内容匹配到真正的来源帧（比例映射会错位）。"""
    content = samples.make_cut_video(4, 8, 64, 48, seed=1)
    reordered = np.concatenate([content[16:], content[:16]])
    _, _, matches = metrics.temporal_match(content, reordered)
    assert np.array_equal(matches[:16], np.arange(16, 32))
    assert np.array_equal(matches[16:], np.arange(0, 16))


def test_order_disruption_detects_reordering():
    clean = samples.make_cut_video(6, 10, 320, 240, seed=0)
    assert metrics.order_disruption(clean, clean) == 0.0
    rng = np.random.default_rng(0)
    reordered = video_transform.reorder_shots(clean, shots.detect_cuts(clean), rng)
    assert metrics.order_disruption(clean, reordered) > 0.03


def test_desensitize_harness_reports_all_metrics():
    clean = samples.make_cut_video(4, 12, 320, 240, seed=1)
    report = harness.run_desensitize_harness(
        clean,
        {"平衡档": {"reorder": True, "speed": 0.955, "recrop": 0.03, "regrade": True}},
        seed=1,
    )
    entry = report["平衡档"]
    assert {
        "content_cosine",
        "motion_cosine",
        "order_disruption",
        "aligned_psnr_db",
        "psnr_db",
    } <= set(entry)
    assert 0 <= entry["order_disruption"] <= 1
