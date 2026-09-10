"""嵌入域定向攻击原语的确定性、dtype 与低频重写契约。"""

from __future__ import annotations

import numpy as np

from cthulhu_backend.transform import embedding_domain


def _band_corr_pair(reference: np.ndarray, attacked: np.ndarray) -> tuple[float, float]:
    """返回两个单通道平面的低频带(1..15)与高频带(16..63)重构相关系数。"""
    rows, cols = embedding_domain._low_band()
    matrix = embedding_domain._dct8()
    high_mask = np.ones((8, 8), dtype=bool)
    high_mask[rows, cols] = False

    def bands(plane: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h, w = plane.shape
        pad_h, pad_w = -h % 8, -w % 8
        padded = np.pad(plane, ((0, pad_h), (0, pad_w)), mode="edge")
        bh, bw = padded.shape[0] // 8, padded.shape[1] // 8
        blocks = padded.reshape(bh, 8, bw, 8).transpose(0, 2, 1, 3)
        coeffs = matrix @ blocks @ matrix.T
        return coeffs[..., rows, cols].reshape(-1), coeffs[..., high_mask].reshape(-1)

    ref_low, ref_high = bands(reference)
    atk_low, atk_high = bands(attacked)

    def corr(a: np.ndarray, b: np.ndarray) -> float:
        a = a - a.mean()
        b = b - b.mean()
        return float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))

    return corr(ref_low, atk_low), corr(ref_high, atk_high)


def _low_freq_texture(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:64, 0:64].astype(np.float32)
    plane = 0.5 + 0.2 * np.sin(2 * np.pi * xx / 32) + 0.2 * np.cos(2 * np.pi * yy / 21)
    plane += rng.normal(0, 0.02, plane.shape).astype(np.float32)
    return plane


def test_strength_zero_is_identity() -> None:
    frames = np.random.default_rng(0).uniform(0, 1, (4, 32, 32, 3)).astype(np.float32)
    out = embedding_domain.attack(frames, mode="chroma", strength=0.0)
    assert out is frames


def test_dtype_roundtrip_and_gray() -> None:
    rng = np.random.default_rng(1)
    color_u8 = rng.integers(0, 255, (3, 48, 48, 3), dtype=np.uint8)
    out = embedding_domain.attack(color_u8, mode="luma", strength=0.6, rng=rng)
    assert out.dtype == np.uint8
    assert out.shape == color_u8.shape
    gray = rng.uniform(0, 1, (3, 48, 48)).astype(np.float32)
    gray_out = embedding_domain.attack(gray, mode="both", strength=0.6, rng=rng)
    assert gray_out.dtype == np.float32
    assert gray_out.shape == gray.shape


def test_deterministic_under_same_rng() -> None:
    frames = np.random.default_rng(2).uniform(0, 1, (4, 32, 32, 3)).astype(np.float32)
    first = embedding_domain.attack(frames, mode="chroma", strength=0.8, rng=np.random.default_rng(7))
    second = embedding_domain.attack(frames, mode="chroma", strength=0.8, rng=np.random.default_rng(7))
    np.testing.assert_array_equal(first, second)


def test_low_band_rewritten_for_selected_channel() -> None:
    """chroma 模式只重写 Cb/Cr 低频：低频带去相关，高频细节与 Y 保持。"""
    rng = np.random.default_rng(3)
    texture = _low_freq_texture()
    rgb = np.repeat(texture[..., None], 3, axis=-1)[None, ...].astype(np.float32)
    rgb[..., 1] = 0.35 + 0.15 * np.sin(2 * np.pi * np.arange(64)[None, :] / 17)
    out = embedding_domain.attack(rgb, mode="chroma", strength=1.0, rng=rng)
    orig_ycbcr = embedding_domain._rgb_to_ycbcr(rgb)
    out_ycbcr = embedding_domain._rgb_to_ycbcr(out)
    low_cb, high_cb = _band_corr_pair(orig_ycbcr[0, ..., 1], out_ycbcr[0, ..., 1])
    low_y, high_y = _band_corr_pair(orig_ycbcr[0, ..., 0], out_ycbcr[0, ..., 0])
    # 低频带被重写：与原始系数不再强相关。
    assert abs(low_cb) < 0.5
    # 高频细节与 Y 平面（色度变换亮度中性）保持。
    assert high_cb > 0.95
    assert low_y > 0.95
    assert high_y > 0.95
    assert np.isfinite(out).all()


def test_unknown_mode_raises() -> None:
    frames = np.zeros((2, 16, 16, 3), dtype=np.float32)
    try:
        embedding_domain.attack(frames, mode="wrong", strength=0.5)
    except ValueError as exc:
        assert "未知嵌入域模式" in str(exc)
    else:
        raise AssertionError("未知模式应抛出 ValueError")


def test_v2_attack_bands_cover_report_top_bands_without_dc() -> None:
    rows, cols = embedding_domain._attack_bands()
    pairs = set(zip(rows, cols, strict=True))
    for index in embedding_domain._TOP_BANDS:
        assert (index // 8, index % 8) in pairs
    assert (0, 0) not in pairs


def test_v2_aggressive_is_deterministic() -> None:
    frames = np.random.default_rng(9).integers(0, 255, (3, 48, 48, 3), dtype=np.uint8)
    kwargs = {
        "mode": "both",
        "strength": 0.6,
        "variant": "v2",
        "block_jitter": True,
        "multiscale": True,
        "chroma_subsample": True,
    }
    first = embedding_domain.attack(frames, rng=np.random.default_rng(11), **kwargs)
    second = embedding_domain.attack(frames, rng=np.random.default_rng(11), **kwargs)
    np.testing.assert_array_equal(first, second)
    assert first.dtype == np.uint8
    assert first.shape == frames.shape
