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


def test_flow_disturb_changes_motion_embedding():
    from cthulhu_backend.similarity import embedding

    clean = samples.make_cut_video(2, 8, 96, 64, seed=5)
    out = regenerate.flow_disturb(clean, strength=2.0, rng=np.random.default_rng(7))
    assert out.shape == clean.shape
    assert out.dtype == clean.dtype
    before = embedding.video_motion_embedding(clean)
    after = embedding.video_motion_embedding(out)
    assert embedding.cosine(before, after) < 0.995
    mse = float(np.mean((clean - out) ** 2))
    assert 10 * np.log10(1.0 / mse) > 30


def test_flow_disturb_uint8_keeps_scale_and_dtype():
    """uint8 输入是生产快路径：输出必须保持 uint8 且首帧尺度一致。"""
    clean = samples.make_cut_video(2, 8, 96, 64, seed=5)
    frames = (clean * 255).round().astype(np.uint8)
    out = regenerate.flow_disturb(frames, strength=2.0, rng=np.random.default_rng(7))
    assert out.dtype == np.uint8
    mse = float(np.mean((frames.astype(np.float32) - out.astype(np.float32)) ** 2))
    assert 10 * np.log10(255.0**2 / mse) > 25


def test_texture_inject_changes_edge_embedding():
    from cthulhu_backend.fingerprint import hashes

    clean = samples.make_cut_video(2, 8, 96, 64, seed=5)
    out = regenerate.texture_inject(clean, strength=0.04, rng=np.random.default_rng(7))
    assert out.shape == clean.shape
    assert out.dtype == clean.dtype
    before = hashes.edge_embedding(clean[0])
    after = hashes.edge_embedding(out[0])
    assert hashes.cosine(before, after) < 0.999
    mse = float(np.mean((clean - out) ** 2))
    assert 10 * np.log10(1.0 / mse) > 30


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


def test_multiscale_perturb_shifts_hashes_across_scales():
    from cthulhu_backend.fingerprint import hashes

    clean = samples.make_cut_video(2, 8, 128, 96, seed=5)
    out = regenerate.multiscale_perturb(clean, strength=0.02, rng=np.random.default_rng(7))
    assert out.shape == clean.shape
    assert out.dtype == clean.dtype
    agreements = []
    for size in (8, 16, 32):
        for source, target in zip(clean, out):
            agreements.append(
                float(
                    np.mean(
                        hashes.phash(source, size=size, low=8)
                        == hashes.phash(target, size=size, low=8)
                    )
                )
            )
    assert float(np.mean(agreements)) < 0.99
    mse = float(np.mean((clean - out) ** 2))
    assert 10 * np.log10(1.0 / mse) > 25


def test_complexity_trap_flattens_complexity_distribution():
    from scipy.ndimage import gaussian_filter

    def complexity_std(frames: np.ndarray) -> float:
        values = []
        for frame in frames:
            band = gaussian_filter(frame, 1.0) - gaussian_filter(frame, 3.0)
            energy = gaussian_filter(np.abs(band), 3.0)
            h, w = energy.shape
            blocks = energy[: h // 8 * 8, : w // 8 * 8].reshape(h // 8, 8, w // 8, 8).mean((1, 3))
            values.append(blocks.std())
        return float(np.mean(values))

    clean = samples.make_cut_video(2, 8, 128, 96, seed=5)
    out = regenerate.complexity_trap(clean, strength=0.06, rng=np.random.default_rng(8))
    assert out.shape == clean.shape
    assert out.dtype == clean.dtype
    assert complexity_std(out) < complexity_std(clean) * 0.95
    mse = float(np.mean((clean - out) ** 2))
    assert 10 * np.log10(1.0 / mse) > 35


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


def test_temporal_blur_softens_frame_difference_and_keeps_ends():
    from cthulhu_backend.similarity import embedding

    clean = samples.make_cut_video(2, 8, 96, 64, seed=5)
    out = regenerate.temporal_blur(clean, strength=0.3)
    assert out.shape == clean.shape
    assert out.dtype == clean.dtype
    np.testing.assert_array_equal(out[0], clean[0])
    np.testing.assert_array_equal(out[-1], clean[-1])
    before = float(np.mean(np.abs(np.diff(clean, axis=0))))
    after = float(np.mean(np.abs(np.diff(out, axis=0))))
    assert after < before
    assert embedding.cosine(
        embedding.video_motion_embedding(clean), embedding.video_motion_embedding(out)
    ) < 0.999
    mse = float(np.mean((clean - out) ** 2))
    assert 10 * np.log10(1.0 / mse) > 25
