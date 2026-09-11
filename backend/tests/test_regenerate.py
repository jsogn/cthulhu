"""再生族攻击测试：跨帧估计相减、小波细节带随机化与像素重写的破坏力。"""

from __future__ import annotations

import numpy as np

from cthulhu_backend import samples
from cthulhu_backend.evaluate import metrics
from cthulhu_backend.transform import regenerate
from cthulhu_backend.watermark import common, dwt, ss

BITS = common.payload_bits(1, 64)
REF = common.SYNC + BITS


def _watermark_ss(frames: np.ndarray) -> np.ndarray:
    return np.stack([ss.embed(f, BITS, seed=0, alpha=0.25) for f in frames])


def _ber_ss(frames: np.ndarray) -> float:
    return float(np.mean([metrics.ber(REF, ss.extract(f, 64, seed=0)) for f in frames]))


def test_temporal_subtract_destroys_frame_constant_ss():
    clean = samples.make_cut_video(3, 12, 320, 240, seed=2)
    watermarked = _watermark_ss(clean)
    assert _ber_ss(watermarked) < 0.05
    attacked = regenerate.temporal_subtract(watermarked, beta=1.0)
    assert _ber_ss(attacked) > 0.35


def test_dwt_detail_destroys_dwt():
    clean = samples.make_cut_video(3, 12, 320, 240, seed=2)
    watermarked = np.stack([dwt.embed(f, BITS, seed=0) for f in clean])
    before = float(np.mean([metrics.ber(REF, dwt.extract(f, 64, seed=0)) for f in watermarked]))
    assert before < 0.05
    attacked = regenerate.dwt_detail(watermarked, strength=1.0, rng=np.random.default_rng(7))
    after = float(np.mean([metrics.ber(REF, dwt.extract(f, 64, seed=0)) for f in attacked]))
    assert after > 0.35


def test_dwt_detail_handles_color_frames():
    clean = samples.make_cut_video(2, 4, 96, 64, seed=5)
    color = np.stack([np.stack([f, f, f], axis=-1) for f in clean])
    out = regenerate.dwt_detail(color, strength=0.8, rng=np.random.default_rng(7))
    assert out.shape == color.shape
    assert out.dtype == color.dtype
    assert np.isfinite(out).all()


def test_fft_phase_preserves_energy_and_layout():
    clean = samples.make_cut_video(2, 4, 96, 64, seed=4)
    out = regenerate.fft_phase(clean, strength=0.6, rng=np.random.default_rng(7))
    assert out.shape == clean.shape
    assert out.dtype == clean.dtype
    assert np.isfinite(out).all()
    drift = abs(float(np.mean(out**2)) - float(np.mean(clean**2))) / float(np.mean(clean**2))
    assert drift < 0.1


def test_hsv_jitter_color_only():
    clean = samples.make_cut_video(2, 4, 96, 64, seed=5)
    color = np.stack([np.stack([f, f, f], axis=-1) for f in clean])
    out = regenerate.hsv_jitter(color, strength=10.0, rng=np.random.default_rng(7))
    assert out.shape == color.shape
    assert out.dtype == color.dtype
    mse = float(np.mean((color - out) ** 2))
    assert 10 * np.log10(1.0 / mse) > 30
    # 灰度帧原样返回。
    gray_out = regenerate.hsv_jitter(clean, strength=10.0, rng=np.random.default_rng(7))
    np.testing.assert_array_equal(gray_out, clean)


def test_copy_attack_reduces_copy_similarity_more_than_random():
    import pytest
    from scipy.ndimage import zoom

    from cthulhu_backend.fingerprint import copy_detect

    if not copy_detect.available():
        pytest.skip("DINOv2 判重代理模型不可用")
    clean = samples.make_cut_video(2, 4, 128, 96, seed=6)
    anchor = clean[len(clean) // 2]
    base = copy_detect.embed_frame(anchor)
    out = regenerate.copy_attack(clean, strength=0.05, rng=np.random.default_rng(7))
    attacked = copy_detect.embed_frame(out[len(out) // 2])
    random_cosines = []
    for seed in range(5):
        random_theta = np.random.default_rng(seed).uniform(-0.05, 0.05, (24, 16)).astype(np.float32)
        random_field = zoom(random_theta, (96 / 24, 128 / 16), order=1)
        random_emb = copy_detect.embed_frame(np.clip(anchor + random_field, 0.0, 1.0))
        random_cosines.append(copy_detect.cosine(base, random_emb))
    assert copy_detect.cosine(base, attacked) < float(np.mean(random_cosines))
    mse = float(np.mean((clean - out) ** 2))
    assert 10 * np.log10(1.0 / mse) > 25


def test_face_perturb_is_localized_to_boxes():
    from cthulhu_backend.fingerprint import deep

    rng = np.random.default_rng(10)
    clean = rng.random((4, 96, 128), dtype=np.float32) * 0.3 + 0.3
    boxes = [(32, 24, 32, 32)]
    out = np.stack(
        [regenerate._perturb_regions(f, boxes, 0.05, np.random.default_rng(11)) for f in clean]
    )
    assert out.shape == clean.shape
    # 羽化半径外的区域保持逐位一致。
    np.testing.assert_array_equal(out[:, :10, :], clean[:, :10, :])
    np.testing.assert_array_equal(out[:, 70:, :], clean[:, 70:, :])
    np.testing.assert_array_equal(out[:, :, :18], clean[:, :, :18])
    np.testing.assert_array_equal(out[:, :, 78:], clean[:, :, 78:])
    # 框内像素被改动。
    inside_diff = float(np.mean(np.abs(out[:, 34:46, 42:54] - clean[:, 34:46, 42:54])))
    assert inside_diff > 1e-3
    mse = float(np.mean((clean - out) ** 2))
    assert 10 * np.log10(1.0 / mse) > 20
    if deep.available():
        base = deep.embed_frame(clean[1])
        attacked = deep.embed_frame(out[1])
        assert deep.cosine(base, attacked) < 0.9999


def test_face_perturb_no_face_passthrough():
    base = np.linspace(0.0, 1.0, 96, dtype=np.float32)[:, None] * np.ones((1, 128), np.float32)
    frames = np.stack([base] * 3)
    out = regenerate.face_perturb(frames, strength=0.05, rng=np.random.default_rng(1))
    np.testing.assert_array_equal(out, frames)


def test_face_perturb_refines_against_arcface(monkeypatch):
    import pytest

    from cthulhu_backend.fingerprint import face_embed

    if not face_embed.available():
        pytest.skip("ArcFace 模型不可用")
    rng = np.random.default_rng(12)
    clean = rng.random((3, 96, 128, 3), dtype=np.float32) * 0.4 + 0.3
    monkeypatch.setattr(regenerate, "_face_boxes", lambda base_u8, min_size=24: [(32, 24, 40, 40)])
    out = regenerate.face_perturb(clean, strength=0.05, rng=np.random.default_rng(13))
    assert out.shape == clean.shape
    np.testing.assert_array_equal(out[:, :8, :, :], clean[:, :8, :, :])
    np.testing.assert_array_equal(out[:, 80:, :, :], clean[:, 80:, :, :])
    inside_diff = float(np.mean(np.abs(out[:, 24:64, 32:72, :] - clean[:, 24:64, 32:72, :])))
    assert inside_diff > 1e-3
    crop_u8 = (clean[1, 24:64, 32:72, :] * 255).round().astype(np.uint8)
    out_u8 = (out[1, 24:64, 32:72, :] * 255).round().astype(np.uint8)
    base = face_embed.embed_crop(crop_u8)
    attacked = face_embed.embed_crop(out_u8)
    assert face_embed.cosine(base, attacked) < 0.9999

