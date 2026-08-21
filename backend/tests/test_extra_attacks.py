"""底层对抗原语测试：形状保持、可测效果与确定性。"""

from __future__ import annotations

import numpy as np

from cthulhu_backend.transform import extra_attacks


def make_frames(color: bool = False) -> np.ndarray:
    rng = np.random.default_rng(0)
    shape = (6, 96, 64, 3) if color else (6, 96, 64)
    return rng.random(shape, dtype=np.float32)


def test_all_off_is_identity():
    frames = make_frames()
    out = extra_attacks.apply(frames, extra_attacks.AssaultParams(), np.random.default_rng(1))
    np.testing.assert_array_equal(out, frames)


def test_median_noise_requant_change_pixels_keep_shape():
    frames = make_frames()
    rng = np.random.default_rng(2)
    params = extra_attacks.AssaultParams(median=3, noise=0.02, requant=32)
    out = extra_attacks.apply(frames, params, rng)
    assert out.shape == frames.shape
    assert float(np.mean(np.abs(out - frames))) > 1e-3
    assert float(np.abs(out).max()) <= 1.0


def test_dct_requant_and_drop_duplicate():
    frames = make_frames()
    out = extra_attacks.apply(
        frames, extra_attacks.AssaultParams(dct_step=12.0, drop_every=3), np.random.default_rng(3)
    )
    assert out.shape == frames.shape
    np.testing.assert_array_equal(out[3], out[2])  # 抽帧复制
    assert float(np.mean(np.abs(out[0] - frames[0]))) > 0


def test_mirror_and_jitter_deterministic():
    frames = make_frames()
    params = extra_attacks.AssaultParams(mirror=True, jitter=0.01)
    first = extra_attacks.apply(frames, params, np.random.default_rng(4))
    second = extra_attacks.apply(frames, params, np.random.default_rng(4))
    np.testing.assert_array_equal(first, second)
    assert first.shape == frames.shape
    assert float(np.mean(np.abs(first - frames))) > 1e-3


def test_chroma_quant_preserves_luma_reduces_chroma():
    rng = np.random.default_rng(5)
    luma = rng.random((4, 64, 48, 1), dtype=np.float32)
    chroma = rng.normal(0, 0.2, (4, 64, 48, 3)).astype(np.float32)
    frames = np.clip(np.repeat(luma, 3, axis=-1) + chroma, 0, 1).astype(np.float32)
    out = extra_attacks.apply(
        frames, extra_attacks.AssaultParams(chroma_levels=16), np.random.default_rng(6)
    )
    assert out.shape == frames.shape

    def luma_of(array: np.ndarray) -> np.ndarray:
        return 0.299 * array[..., 0] + 0.587 * array[..., 1] + 0.114 * array[..., 2]

    cb_out = -0.168736 * out[..., 0] - 0.331264 * out[..., 1] + 0.5 * out[..., 2]
    distinct_levels = np.unique(np.round(cb_out * 15)).size
    assert distinct_levels <= 16
    assert float(np.abs(luma_of(out).mean() - luma_of(frames).mean())) < 0.05
