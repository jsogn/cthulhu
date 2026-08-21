"""数值归约安全工具测试：大轴 float32 场景不重蹈 numpy 饱和问题。"""

from __future__ import annotations

import numpy as np

from cthulhu_backend.numeric import channel_stats


def test_channel_stats_matches_float64_reference():
    rng = np.random.default_rng(0)
    frames = rng.random((12, 64, 48, 3), dtype=np.float32)
    mean, std = channel_stats(frames)
    reference = frames.astype(np.float64).reshape(-1, 3)
    np.testing.assert_allclose(mean, reference.mean(axis=0), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(std, reference.std(axis=0), rtol=1e-5, atol=1e-6)


def test_channel_stats_large_axis_is_accurate():
    """触发 numpy float32 大轴归约饱和的规模，安全工具仍应给出正确值。"""
    rng = np.random.default_rng(0)
    frames = rng.random((2**26, 3), dtype=np.float32)
    mean, _ = channel_stats(frames)
    assert mean.shape == (3,)
    np.testing.assert_allclose(mean, 0.5, rtol=1e-3, atol=1e-3)
