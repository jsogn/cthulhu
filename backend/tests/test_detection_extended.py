"""新增盲检测族的分离度与置信度回归测试。"""

from __future__ import annotations

import numpy as np
from scipy.fftpack import dctn, idctn
from scipy.ndimage import gaussian_filter

from cthulhu_backend import samples
from cthulhu_backend.watermark import common, detect, dwt, temporal


def _video(n=64, h=64, w=64, seed=0) -> np.ndarray:
    return samples.make_video_frames(n, w, h, seed=seed)


def test_temporal_blind_separates_embedded_from_clean() -> None:
    clean = _video()
    bits = common.payload_bits(1, 64)
    marked = temporal.embed_frames(clean, bits, alpha=0.05)
    clean_score = detect.temporal_blind(clean)
    marked_score = detect.temporal_blind(marked)
    assert 0.0 <= clean_score <= 1.0
    assert marked_score > clean_score + 0.1


def test_dwt_blind_separates_embedded_from_clean() -> None:
    clean = _video()
    bits = common.payload_bits(2, 64)
    marked = np.stack([dwt.embed(frame, bits, seed=0, alpha=0.08) for frame in clean])
    clean_score = detect.dwt_blind(clean)
    marked_score = detect.dwt_blind(marked)
    assert marked_score > clean_score + 0.1


def test_chroma_blind_separates_chroma_noise_from_clean() -> None:
    rng = np.random.default_rng(0)
    # 自然化合成帧：真实灰度场景（色差恒为 0）作为干净基准。
    smooth = gaussian_filter(rng.random((12, 64, 64), dtype=np.float32), 2.5)
    luma = np.clip(0.35 + 0.25 * (smooth - smooth.mean()), 0, 1)
    base = np.repeat(luma[..., None], 3, axis=-1).astype(np.float32)
    pattern = rng.standard_normal((64, 64), dtype=np.float32)
    pattern = pattern / float(np.sqrt(np.mean(pattern**2)))
    # 经 YCbCr 往返把图案打进色差通道，再回到 RGB。
    y = 0.299 * base[..., 0] + 0.587 * base[..., 1] + 0.114 * base[..., 2]
    cb = -0.168736 * base[..., 0] - 0.331264 * base[..., 1] + 0.5 * base[..., 2]
    cr = 0.5 * base[..., 0] - 0.418688 * base[..., 1] - 0.081312 * base[..., 2]
    cb = cb + 0.04 * pattern
    cr = cr + 0.04 * pattern
    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb
    marked = np.clip(np.stack([r, g, b], axis=-1), 0, 1)
    assert detect.chroma_blind(marked) > detect.chroma_blind(base) + 0.1


def _dctmod_embed(frame: np.ndarray, bits: list[int], mod1: int = 36, mod2: int = 20) -> np.ndarray:
    """DCT 模运算参考嵌入：直流系数对齐到 mod1/mod2 格点。"""
    f8 = frame * 255.0
    h, w = f8.shape[0] - f8.shape[0] % 8, f8.shape[1] - f8.shape[1] % 8
    blocks = f8[:h, :w].reshape(h // 8, 8, w // 8, 8)
    coeffs = dctn(blocks, axes=(1, 3), norm="ortho")
    dc = coeffs[:, 0, :, 0].copy()  # 块 DC 系数位于频率索引 (0,0)
    flat = dc.ravel()
    for i in range(min(len(bits), flat.size)):
        mod = mod1 if bits[i] else mod2
        flat[i] = round(flat[i] / mod) * mod
    dc = flat.reshape(dc.shape)
    coeffs[:, 0, :, 0] = dc
    out = np.zeros_like(f8)
    spatial = idctn(
        coeffs.transpose(0, 2, 1, 3).reshape(h // 8, w // 8, 8, 8), axes=(2, 3), norm="ortho"
    )
    out[:h, :w] = spatial.transpose(0, 2, 1, 3).reshape(h, w)
    return np.clip(out / 255.0, 0, 1)


def test_dctmod_blind_separates_grid_aligned_from_clean() -> None:
    clean = _video(n=8)
    bits = common.payload_bits(3, 128)
    marked = np.stack([_dctmod_embed(frame, bits) for frame in clean])
    assert detect.dctmod_blind(marked) > detect.dctmod_blind(clean) + 0.1


def _svdmod_embed(frame: np.ndarray, bits: list[int], grid: float = 20.0) -> np.ndarray:
    """SVD 模运算参考嵌入：4×4 块前导奇异值对齐格点。"""
    f8 = frame * 255.0
    h, w = f8.shape[0] - f8.shape[0] % 4, f8.shape[1] - f8.shape[1] % 4
    blocks = f8[:h, :w].reshape(h // 4, 4, w // 4, 4).transpose(0, 2, 1, 3)
    blocks = blocks.reshape(-1, 4, 4)
    u, s, vt = np.linalg.svd(blocks)
    flat = s[:, 0].copy()
    for i in range(min(len(bits), flat.size)):
        flat[i] = round(flat[i] / grid) * grid + (grid / 2 if bits[i] else 0.0)
    s[:, 0] = flat
    out_blocks = u @ (s[..., None] * np.eye(4)[None, ...]) @ vt
    out = np.zeros_like(f8)
    out[:h, :w] = (
        out_blocks.reshape(h // 4, w // 4, 4, 4).transpose(0, 2, 1, 3).reshape(h, w)
    )
    return np.clip(out / 255.0, 0, 1)


def test_svd_blind_separates_grid_aligned_from_clean() -> None:
    clean = _video(n=8)
    bits = common.payload_bits(4, 128)
    marked = np.stack([_svdmod_embed(frame, bits) for frame in clean])
    assert detect.svd_blind(marked) > detect.svd_blind(clean) + 0.05


def test_windowed_scores_report_confidence() -> None:
    frames = _video(n=120)
    scores, confidence = detect.windowed_video_scores_with_confidence(frames, window=50)
    assert set(scores) == {"ss", "qim", "lsb", "temporal", "dwt"}
    assert set(confidence) == set(scores)
    assert all(0.0 <= confidence[key] <= 1.0 for key in confidence)


def test_video_scores_cover_new_families() -> None:
    scores = detect.video_scores(_video(n=8))
    assert set(scores) == {"ss", "qim", "lsb", "temporal", "dwt"}


def test_hits_respect_threshold_and_confidence() -> None:
    scores = {"ss": 0.9, "qim": 0.9, "temporal": 0.2, "dwt": 0.2, "chroma": 0.2}
    confidence = {"ss": 0.9, "qim": 0.4, "temporal": 0.9, "dwt": 0.9}
    assert detect.hits(scores, {}, confidence) == ["ss"]

    merged = detect.hits(
        scores, {"dctmod": 0.9, "svd": 0.1}, {"ss": 0.9, "qim": 0.9, "dctmod": 0.9, "svd": 0.9}
    )
    assert merged == ["ss", "qim", "dctmod"]
