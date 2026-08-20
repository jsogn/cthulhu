"""内容脱敏评估：时序乱序度指标与评估矩阵。"""

from __future__ import annotations

import numpy as np

from cthulhu_backend import samples
from cthulhu_backend.evaluate import harness, metrics
from cthulhu_backend.transform import shots
from cthulhu_backend.transform import video as video_transform


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
