"""重建后处理（细节回注/后置锐化）的行为回归。

模块从 purify 提出来后独立测试：只断言可观察的画质行为，不碰净化引擎状态。
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from cthulhu_backend.transform import postprocess


def _softened_video() -> np.ndarray:
    smooth = np.full((2, 64, 64, 3), 128, dtype=np.uint8)
    smooth[:, 20:44, 20:44] = 160
    return np.stack(
        [
            np.clip(
                gaussian_filter(frame.astype(np.float32), sigma=(2.0, 2.0, 0.0)), 0, 255
            ).astype(np.uint8)
            for frame in smooth
        ]
    )


def _high_energy(values: np.ndarray) -> float:
    work = values.astype(np.float32)
    return float(np.mean(np.abs(work - gaussian_filter(work, sigma=(0.0, 1.2, 1.2, 0.0)))))


def test_unsharp_batch_restores_edges() -> None:
    """后置锐化应提升高频能量（瓶颈重建天生偏软）。"""
    softened = _softened_video()
    sharpened = postprocess.unsharp_batch(softened, 0.6, 1.2)
    assert _high_energy(sharpened) > _high_energy(softened)


def test_unsharp_batch_zero_amount_is_identity() -> None:
    softened = _softened_video()
    np.testing.assert_array_equal(
        postprocess.unsharp_batch(softened, 0.0, 1.2), softened
    )


def test_reinject_detail_restores_high_frequency() -> None:
    """回注把原帧高频还回重建结果：重建越糊，回注后高频能量越接近原帧。"""
    rng = np.random.default_rng(3)
    frames = rng.random((2, 48, 48, 3), dtype=np.float32)
    restored_frames = []
    blurred_frames = []
    for frame in frames:
        blurred = gaussian_filter(frame, sigma=(1.5, 1.5, 0.0)).astype(np.float32)
        blurred_frames.append(blurred)
        restored_frames.append(
            postprocess.reinject_detail(
                blurred.copy(),
                frame,
                strength=1.0,
                sigma=1.5,
                is_u8=False,
                gray=False,
            )
        )
    blurred = np.stack(blurred_frames)
    restored = np.stack(restored_frames)
    assert _high_energy(restored) > _high_energy(blurred)
    assert float(np.mean(np.abs(restored - frames))) < float(
        np.mean(np.abs(blurred - frames))
    )


def test_reinject_detail_zero_strength_is_passthrough() -> None:
    rng = np.random.default_rng(4)
    original = rng.random((1, 32, 32), dtype=np.float32)
    purified = np.clip(original * 0.5 + 0.2, 0.0, 1.0).astype(np.float32)
    out = postprocess.reinject_detail(
        purified, original, strength=0.0, sigma=1.2, is_u8=False, gray=True
    )
    assert out is purified


def test_detail_scale_is_bounded_and_monotonic() -> None:
    """高频能量越高回注越强，且始终夹在 [0.35, 1.0]。"""
    flat = np.zeros((16, 16), dtype=np.float32)
    textured = np.linspace(0.0, 1.0, 16 * 16, dtype=np.float32).reshape(16, 16)
    assert postprocess.detail_scale(flat) == 0.35
    assert postprocess.detail_scale(textured) > postprocess.detail_scale(flat)
    assert 0.35 <= postprocess.detail_scale(textured * 100.0) <= 1.0
