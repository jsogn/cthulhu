"""时序一致性残差估计原语的契约（不依赖任何水印模型权重）。"""

from __future__ import annotations

import numpy as np
import pytest

from cthulhu_backend.transform import temporal


def _static_band_frames(frames: int = 12, size: int = 64) -> np.ndarray:
    rng = np.random.default_rng(20260909)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    pattern = (
        0.045 * np.sin(2.0 * np.pi * xx / 28.0)
        + 0.035 * np.cos(2.0 * np.pi * yy / 19.0)
    ).astype(np.float32)
    moving = rng.normal(0.0, 0.04, (frames, size, size, 3)).astype(np.float32)
    return np.clip(moving + pattern[None, ..., None], 0.0, 1.0)


def test_estimate_recovers_cross_frame_component() -> None:
    frames = _static_band_frames()
    estimate = temporal.estimate(frames, mode="luma", max_frames=0, max_edge=0)
    assert estimate.coherence > 0.5
    assert estimate.luma is not None
    before = float(np.std(np.median(temporal._luma(frames), axis=0)))
    attacked = temporal.subtract(frames, estimate, 1.0, mode="luma")
    after = float(np.std(np.median(temporal._luma(attacked), axis=0)))
    assert after < before


def test_independent_noise_has_low_coherence() -> None:
    rng = np.random.default_rng(7)
    frames = rng.uniform(0.2, 0.8, (10, 48, 48, 3)).astype(np.float32)
    estimate = temporal.estimate(frames, mode="luma", max_frames=0, max_edge=0)
    assert estimate.coherence < 0.5


def test_dtype_roundtrip_and_chroma_mode() -> None:
    rng = np.random.default_rng(3)
    frames = rng.integers(0, 255, (8, 32, 32, 3), dtype=np.uint8)
    estimate = temporal.estimate(frames, mode="chroma", max_frames=0, max_edge=0)
    out = temporal.subtract(frames, estimate, 0.5, mode="chroma")
    assert out.dtype == np.uint8
    assert out.shape == frames.shape


def test_unknown_mode_raises() -> None:
    frames = np.zeros((4, 16, 16, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="未知时序模式"):
        temporal.estimate(frames, mode="wrong")
    estimate = temporal.estimate(frames, mode="luma", max_frames=0, max_edge=0)
    with pytest.raises(ValueError, match="未知时序模式"):
        temporal.subtract(frames, estimate, 0.5, mode="wrong")
