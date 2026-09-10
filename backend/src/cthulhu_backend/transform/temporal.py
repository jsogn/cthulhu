"""时序一致性残差估计：从单个带水印视频里估计跨帧稳定的低频水印分量。

报告 §2.8 显示 VideoSeal 的水印残差帧间相关高达 0.916；这是它抗帧序/
插值的来源，也意味着“跨帧不变的低频分量”可以被单独估计并削弱。生产环境
没有干净参考帧，因此本模块只做黑盒估计：

1. 取零均值低通分量作为水印可能存在的低频带（保留空间结构、去掉整帧亮度）；
2. 用帧间运动掩掉明显运动区域，只保留低运动像素；
3. 对低运动像素做时间中值，得到跨帧稳定分量估计；
4. 用逐帧残差与估计的相关性给出 coherence，供自动画像决定是否启用减法。

这是代理估计，不是水印确认：静态画面的合法纹理也会跨帧稳定。调用方必须
把它放在“增强档/自动画像高一致性”路径，并保留画质回退。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter, zoom

from cthulhu_backend.transform import embedding_domain

_EPS = 1e-6
MODES = ("luma", "chroma", "both")


@dataclass
class TemporalEstimate:
    """跨帧稳定分量估计；数组可能位于降采样分辨率。"""

    luma: np.ndarray | None
    chroma: np.ndarray | None
    coherence: float
    motion: float
    source_shape: tuple[int, int]
    sample_shape: tuple[int, int]


def _as_float(frames: np.ndarray) -> tuple[np.ndarray, np.dtype]:
    original = frames.dtype
    work = frames.astype(np.float32)
    if original == np.uint8:
        work /= 255.0
    return work, original


def _as_original(work: np.ndarray, original: np.dtype) -> np.ndarray:
    clipped = np.clip(work, 0.0, 1.0)
    if original == np.uint8:
        return (clipped * 255.0).round().astype(np.uint8)
    return clipped.astype(original)


def _luma(work: np.ndarray) -> np.ndarray:
    if work.ndim == 4:
        return 0.299 * work[..., 0] + 0.587 * work[..., 1] + 0.114 * work[..., 2]
    return work


def _sample_frames(frames: np.ndarray, max_frames: int) -> np.ndarray:
    if max_frames <= 0 or len(frames) <= max_frames:
        return frames
    indices = np.linspace(0, len(frames) - 1, max_frames).astype(int)
    return frames[indices]


def _downscale(frames: np.ndarray, max_edge: int) -> np.ndarray:
    if max_edge <= 0:
        return frames
    height, width = frames.shape[1:3]
    longest = max(height, width)
    if longest <= max_edge:
        return frames
    scale = max_edge / longest
    target = (max(2, round(height * scale)), max(2, round(width * scale)))
    factors = (1.0, target[0] / height, target[1] / width)
    if frames.ndim == 4:
        factors += (1.0,)
    return zoom(frames, factors, order=1).astype(np.float32)


def _upscale_plane(plane: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if plane.shape[:2] == shape:
        return plane
    factors = (shape[0] / plane.shape[0], shape[1] / plane.shape[1])
    if plane.ndim == 3:
        factors += (1.0,)
    return zoom(plane, factors, order=1).astype(np.float32)


def _coherence(band: np.ndarray, estimate: np.ndarray, mask: np.ndarray) -> float:
    """逐帧低通分量与时间中值估计的归一化相关均值。"""
    if band.size == 0 or not np.any(mask):
        return 0.0
    reference = estimate[mask].reshape(-1)
    ref_norm = float(np.linalg.norm(reference))
    if ref_norm < _EPS:
        return 0.0
    scores: list[float] = []
    for frame in band:
        current = frame[mask].reshape(-1)
        denom = float(np.linalg.norm(current)) * ref_norm
        if denom < _EPS:
            continue
        scores.append(float(np.dot(current, reference) / denom))
    return float(np.mean(scores)) if scores else 0.0


def _static_weight(motion: np.ndarray) -> np.ndarray:
    """低运动像素软权重；全静态时接近 1，运动大时平滑降到接近 0。"""
    if not np.any(motion > 0):
        return np.ones_like(motion, dtype=np.float32)
    threshold = float(np.percentile(motion, 65))
    weight = np.clip(1.0 - motion / (threshold * 2.0 + _EPS), 0.0, 1.0)
    return gaussian_filter(weight, sigma=2.0).astype(np.float32)


def _lowpass(plane: np.ndarray, sigma: float) -> np.ndarray:
    """零均值低通分量：保留低频空间结构，去掉每帧整体亮度。"""
    if plane.ndim == 4:
        sigmas = (0.0, sigma, sigma, 0.0)
    else:
        sigmas = (0.0, sigma, sigma)
    low = gaussian_filter(plane, sigma=sigmas)
    return low - low.mean(axis=(1, 2), keepdims=True)


def estimate(
    frames: np.ndarray,
    *,
    mode: str = "luma",
    sigma: float = 3.0,
    max_frames: int = 24,
    max_edge: int = 0,
) -> TemporalEstimate:
    """估计跨帧稳定低频分量；返回估计、coherence 与运动强度。"""
    if mode not in MODES:
        raise ValueError(f"未知时序模式：{mode}（可选 {' / '.join(MODES)}）")
    if len(frames) == 0:
        raise ValueError("时序估计需要至少一帧")
    work, _ = _as_float(frames)
    sampled = _sample_frames(work, max_frames)
    sampled = _downscale(sampled, max_edge)
    source_shape = tuple(int(value) for value in work.shape[1:3])
    sample_shape = tuple(int(value) for value in sampled.shape[1:3])

    luma = _luma(sampled)
    if len(luma) >= 2:
        motion = np.mean(np.abs(np.diff(luma, axis=0)), axis=0)
    else:
        motion = np.zeros(sample_shape, dtype=np.float32)
    weight = _static_weight(motion)
    mask = weight > 0.35
    if not np.any(mask):
        mask = np.ones(sample_shape, dtype=bool)

    luma_estimate = None
    chroma_estimate = None
    luma_coherence = 0.0
    chroma_coherence = 0.0
    if mode in ("luma", "both"):
        band = _lowpass(luma, sigma)
        luma_estimate = np.median(band, axis=0).astype(np.float32)
        luma_estimate *= weight
        luma_coherence = _coherence(band, luma_estimate, mask)
    if mode in ("chroma", "both") and sampled.ndim == 4:
        ycbcr = embedding_domain.rgb_to_ycbcr(sampled)
        chroma = ycbcr[..., 1:3]
        chroma_band = _lowpass(chroma, sigma * 0.8)
        chroma_estimate = np.median(chroma_band, axis=0).astype(np.float32)
        chroma_estimate *= weight[..., None]
        chroma_coherence = max(
            _coherence(chroma_band[..., channel], chroma_estimate[..., channel], mask)
            for channel in range(chroma_band.shape[-1])
        )
    coherence = max(luma_coherence, chroma_coherence)
    motion_score = float(np.mean(motion)) if motion.size else 0.0
    return TemporalEstimate(
        luma=luma_estimate,
        chroma=chroma_estimate,
        coherence=round(float(coherence), 4),
        motion=round(motion_score, 6),
        source_shape=source_shape,
        sample_shape=sample_shape,
    )


def subtract(
    frames: np.ndarray,
    estimate: TemporalEstimate,
    strength: float,
    *,
    mode: str = "luma",
) -> np.ndarray:
    """按估计减去跨帧稳定分量；strength<=0 时原样返回。"""
    if strength <= 0:
        return frames
    if mode not in MODES:
        raise ValueError(f"未知时序模式：{mode}（可选 {' / '.join(MODES)}）")
    work, original = _as_float(frames)
    height, width = work.shape[1:3]
    luma_estimate = (
        _upscale_plane(estimate.luma, (height, width))
        if estimate.luma is not None
        else None
    )
    chroma_estimate = (
        _upscale_plane(estimate.chroma, (height, width))
        if estimate.chroma is not None
        else None
    )
    if work.ndim == 3:
        if luma_estimate is not None and mode in ("luma", "both"):
            work = work - strength * luma_estimate[None, ...]
        return _as_original(work, original)

    ycbcr = embedding_domain.rgb_to_ycbcr(work)
    if luma_estimate is not None and mode in ("luma", "both"):
        ycbcr[..., 0] -= strength * luma_estimate[None, ...]
    if chroma_estimate is not None and mode in ("chroma", "both"):
        ycbcr[..., 1:3] -= strength * chroma_estimate[None, ...]
    return _as_original(embedding_domain.ycbcr_to_rgb(ycbcr), original)
