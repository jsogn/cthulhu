"""重建后的画面后处理：细节回注与后置锐化（纯函数，独立于净化引擎）。

从 `purify` 提出来单独成模块：这些算法只吃帧数组、不碰模型与状态，既方便
独立回归（见 tests/test_postprocess.py），也让净化模块只留引擎编排。
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

# 后置锐化的推荐参数：瓶颈重建天生偏软，用它把边缘观感拉回来。
POST_SHARPEN_AMOUNT = 0.6
POST_SHARPEN_SIGMA = 1.2


def detail_scale(high: np.ndarray) -> float:
    """按帧内高频能量自适应细节回注强度：纹理越多回注越多，平坦区保守。"""
    energy = float(np.mean(np.abs(high)))
    return float(np.clip(0.35 + 0.65 * (energy / (energy + 0.02)), 0.35, 1.0))


def lowpass_f32(source: np.ndarray, sigma: float, gray: bool) -> np.ndarray:
    """低通（用于取高频）；cv2 可用时用它，720p 上比 scipy 快约 6 倍。"""
    if not gray:
        try:
            import cv2
        except ImportError:
            cv2 = None
        if cv2 is not None:
            return cv2.GaussianBlur(source, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return gaussian_filter(source, sigma=sigma if gray else (sigma, sigma, 0.0))


def reinject_detail(
    purified: np.ndarray,
    original: np.ndarray,
    *,
    strength: float,
    sigma: float,
    is_u8: bool,
    gray: bool,
    wide_sigma: float | None = None,
    weight: np.ndarray | None = None,
) -> np.ndarray:
    """把原帧高频细节回注到重建结果，低频水印仍由重建/重写层处理。"""
    if strength <= 0:
        return purified
    source = original.astype(np.float32, copy=True)
    if is_u8:
        source *= 1.0 / 255.0
    # purified 始终是解码后的 [0,1] float；不能再按 is_u8 除一次 255。
    # 目标数组就地运算：调用方传进来的都是当帧新算出的副本，可安全复用。
    target = (
        purified
        if purified.dtype == np.float32 and purified.flags.writeable
        else purified.astype(np.float32)
    )
    if target.size and float(target.max()) > 1.5:
        target *= 1.0 / 255.0
    # 高频 = 原帧 - 低通（同一块内存复用，省掉两次全尺寸临时数组）。
    blurred = lowpass_f32(source, sigma, gray)
    high = blurred
    np.subtract(source, blurred, out=high)
    # 字幕增强：在字幕掩膜内改用更宽的频带（把笔画从原帧捞回来），
    # 掩膜外保持安全带宽，让水印回流的面积受限于字幕区域本身。
    if wide_sigma is not None and wide_sigma > sigma and weight is not None:
        wide_blur = lowpass_f32(source, wide_sigma, gray)
        np.subtract(source, wide_blur, out=wide_blur)
        np.subtract(wide_blur, high, out=wide_blur)
        if weight.ndim == 2:
            wide_blur *= weight[..., None] if not gray else weight
        else:
            wide_blur *= weight
        np.add(high, wide_blur, out=high)
    # 能量估计按 1/8 抽样即可，避免每帧对全图求绝对值均值。
    flat = high.reshape(-1)
    scale = detail_scale(float(np.mean(np.abs(flat[::8]))))
    np.multiply(high, strength * scale, out=high)
    np.add(target, high, out=target)
    return np.clip(target, 0.0, 1.0, out=target)


def unsharp_batch(frames: np.ndarray, amount: float, sigma: float) -> np.ndarray:
    """后置锐化：瓶颈重建天生偏软，用它把边缘观感拉回来。"""
    if amount <= 0 or len(frames) == 0:
        return frames
    try:
        import cv2

        out = np.empty_like(frames)
        for index, frame in enumerate(frames):
            blurred = cv2.GaussianBlur(frame, (0, 0), sigma)
            out[index] = cv2.addWeighted(frame, 1.0 + amount, blurred, -amount, 0.0)
        return out
    except ImportError:
        from PIL import Image, ImageFilter

        return np.stack(
            [
                np.asarray(
                    Image.fromarray(frame).filter(
                        ImageFilter.UnsharpMask(
                            radius=sigma, percent=int(amount * 100), threshold=0
                        )
                    )
                )
                for frame in frames
            ]
        )
