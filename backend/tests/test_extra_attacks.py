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


def test_requant_plane_matches_reference():
    from cthulhu_backend.attacks import dct as dct_attacks

    rng = np.random.default_rng(30)
    frames = rng.random((4, 96, 64), dtype=np.float32)
    expected = np.stack([dct_attacks.requant_dct(f, 12.0) for f in frames])
    actual = extra_attacks.dct_requant(frames, 12.0)
    np.testing.assert_allclose(actual, expected, atol=0.01)


def test_color_dct_requant_preserves_luma_and_touches_chroma():
    rng = np.random.default_rng(31)
    luma = rng.random((4, 96, 64, 1), dtype=np.float32)
    color = np.clip(np.repeat(luma, 3, axis=-1) + rng.normal(0, 0.08, (4, 96, 64, 3)), 0, 1)
    color = color.astype(np.float32)
    out = extra_attacks.dct_requant(color, 12.0)
    assert out.shape == color.shape

    def luma_of(a: np.ndarray) -> np.ndarray:
        return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]

    assert float(np.abs(luma_of(out).mean() - luma_of(color).mean())) < 0.03
    # 色度确实被改动（R-G 差分布变化）
    before = np.std(color[..., 0] - color[..., 1])
    after = np.std(out[..., 0] - out[..., 1])
    assert abs(after - before) > 1e-4


def test_mirror_and_jitter_deterministic():
    frames = make_frames()
    params = extra_attacks.AssaultParams(mirror=True, jitter=0.01)
    first = extra_attacks.apply(frames, params, np.random.default_rng(4))
    second = extra_attacks.apply(frames, params, np.random.default_rng(4))
    np.testing.assert_array_equal(first, second)
    assert first.shape == frames.shape
    assert float(np.mean(np.abs(first - frames))) > 1e-3


def test_perspective_shear_preserves_content_and_shape():
    frames = make_frames(color=True)
    params = extra_attacks.AssaultParams(perspective=0.01)
    first = extra_attacks.apply(frames, params, np.random.default_rng(7))
    second = extra_attacks.apply(frames, params, np.random.default_rng(7))
    assert first.shape == frames.shape
    np.testing.assert_array_equal(first, second)
    assert float(np.mean(np.abs(first - frames))) > 1e-4
    assert float(np.abs(first.mean() - frames.mean())) < 0.1  # 内容身份保持


def test_local_warp_deterministic_and_content_preserving():
    frames = make_frames()
    params = extra_attacks.AssaultParams(warp=0.003)
    first = extra_attacks.apply(frames, params, np.random.default_rng(8))
    second = extra_attacks.apply(frames, params, np.random.default_rng(8))
    assert first.shape == frames.shape
    np.testing.assert_array_equal(first, second)
    assert float(np.mean(np.abs(first - frames))) > 1e-4
    assert float(np.abs(first.mean() - frames.mean())) < 0.05


def test_estimate_subtract_reduces_spread_spectrum_score():
    from cthulhu_backend.watermark import common, detect, ss

    frames = make_frames()
    bits = common.payload_bits(1, 64)
    watermarked = np.stack([ss.embed(frame, bits, seed=0, alpha=0.25) for frame in frames])
    before = detect.video_scores(watermarked)["ss"]
    attacked = extra_attacks.estimate_subtract(watermarked, beta=1.2, size=3)
    after = detect.video_scores(attacked)["ss"]
    assert after < before


def test_saliency_overlay_deterministic_and_changes_content():
    frames = make_frames(color=True)
    layout = extra_attacks.saliency_layout(9, level=2)
    assert layout is not None and layout.mosaic is not None
    first = extra_attacks.salient_overlay(frames, layout)
    second = extra_attacks.salient_overlay(frames, layout)
    assert first.shape == frames.shape
    np.testing.assert_array_equal(first, second)
    assert float(np.mean(np.abs(first.astype(np.float32) - frames.astype(np.float32)))) > 0.02
    assert extra_attacks.saliency_layout(9, 0) is None


def test_saliency_higher_levels_add_strong_content():
    layout3 = extra_attacks.saliency_layout(9, level=3)
    layout4 = extra_attacks.saliency_layout(9, level=4)
    assert layout3 is not None and layout3.title_text and layout3.band_blur
    assert layout4 is not None and layout4.pip
    frames = make_frames(color=True)
    out = extra_attacks.salient_overlay(frames, layout4)
    assert out.shape == frames.shape
    assert float(np.mean(np.abs(out.astype(np.float32) - frames.astype(np.float32)))) > 0.05


def test_protect_details_restores_toward_original():
    frames = make_frames(color=True)
    attacked = np.clip(frames + 0.06, 0.0, 1.0).astype(np.float32)
    out = extra_attacks.protect_details(attacked, frames, strength=0.7)
    assert out.shape == frames.shape
    assert out.dtype == frames.dtype
    assert float(np.mean(np.abs(out - frames))) < float(np.mean(np.abs(attacked - frames)))
    assert float(np.mean(np.abs(out - attacked))) > 0


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
