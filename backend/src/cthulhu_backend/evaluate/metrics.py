"""BER / PSNR / SSIM 客观指标。"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, zoom

from cthulhu_backend.sample_prep.align import estimate_shift
from cthulhu_backend.watermark.common import bit_error_rate


def psnr(reference: np.ndarray, candidate: np.ndarray) -> float:
    mse = float(np.mean((np.asarray(reference) - np.asarray(candidate)) ** 2))
    if mse == 0:
        return float("inf")
    return float(-10 * np.log10(mse))


def ssim(reference: np.ndarray, candidate: np.ndarray, window: int = 11, sigma: float = 1.5) -> float:
    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    kernel = np.outer(
        np.exp(-((np.arange(window) - window // 2) ** 2) / (2 * sigma**2)),
        np.exp(-((np.arange(window) - window // 2) ** 2) / (2 * sigma**2)),
    )
    kernel /= kernel.sum()
    mu1 = gaussian_filter(ref, sigma=sigma)
    mu2 = gaussian_filter(cand, sigma=sigma)
    mu1_sq, mu2_sq, mu12 = mu1**2, mu2**2, mu1 * mu2
    sigma1_sq = gaussian_filter(ref**2, sigma=sigma) - mu1_sq
    sigma2_sq = gaussian_filter(cand**2, sigma=sigma) - mu2_sq
    sigma12 = gaussian_filter(ref * cand, sigma=sigma) - mu12
    c1, c2 = 0.01**2, 0.03**2
    num = (2 * mu12 + c1) * (2 * sigma12 + c2)
    den = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    return float(np.mean(num / den))


def ber(ref_bits: list[int], out_bits: list[int]) -> float:
    return bit_error_rate(ref_bits, out_bits)


def aligned_psnr(reference: np.ndarray, candidate: np.ndarray) -> float:
    """先做相位相关配准、裁边，再计算 PSNR（避免错位造成的虚高失真）。"""
    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    dy, dx = estimate_shift(ref, cand)
    rolled = np.roll(np.roll(cand, dy, axis=0), dx, axis=1)
    margin = max(8, abs(dy) + 1, abs(dx) + 1)
    ref_c = ref[margin:-margin, margin:-margin]
    cand_c = rolled[margin:-margin, margin:-margin]
    return psnr(ref_c, cand_c)


def aligned_ssim(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    dy, dx = estimate_shift(ref, cand)
    rolled = np.roll(np.roll(cand, dy, axis=0), dx, axis=1)
    margin = max(8, abs(dy) + 1, abs(dx) + 1)
    return ssim(ref[margin:-margin, margin:-margin], rolled[margin:-margin, margin:-margin])


def adjacent_cosine(frames: np.ndarray) -> float:
    """相邻帧平均余弦相似度：度量时序顺序一致性。

    正常视频相邻帧高度相似；乱序重组后帧边界变为硬切，该值显著下降，
    用于量化「打散时序」对内容指纹的破坏效果。
    """
    frames = np.asarray(frames, dtype=np.float64)
    if len(frames) < 2:
        return 1.0
    flat = frames.reshape(len(frames), -1)
    norms = np.linalg.norm(flat, axis=1, keepdims=True) + 1e-12
    normalized = flat / norms
    similarities = np.sum(normalized[1:] * normalized[:-1], axis=1)
    return float(np.mean(similarities))


def order_disruption(original: np.ndarray, reordered: np.ndarray) -> float:
    """时序乱序度：重排序列每帧映射回原始最近帧后，相邻映射跳变的比例。

    原顺序样本映射单调（跳变 0）；乱序重组后段边界出现大幅跳变，
    该值越高说明镜头顺序指纹被破坏得越彻底。
    """
    ref = np.asarray(original, dtype=np.float64).reshape(len(original), -1)
    mov = np.asarray(reordered, dtype=np.float64).reshape(len(reordered), -1)
    if len(ref) < 2 or len(mov) < 2:
        return 0.0
    ref = ref / (np.linalg.norm(ref, axis=1, keepdims=True) + 1e-12)
    mov = mov / (np.linalg.norm(mov, axis=1, keepdims=True) + 1e-12)
    matches = np.argmax(mov @ ref.T, axis=1).astype(np.float64)
    return float(np.mean(np.abs(np.diff(matches)) > 1.0))


def temporal_match(
    original: np.ndarray,
    processed: np.ndarray,
    cap: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按内容相似度把处理帧匹配到最近原始帧，返回 (ref, mov, matches)。

    用于消除重排/变速造成的时序错位：mov[i] 的内容最近似 ref[matches[i]]。
    """
    ref = np.asarray(original, dtype=np.float64)
    mov = np.asarray(processed, dtype=np.float64)
    if len(ref) > cap:
        indices = np.linspace(0, len(ref) - 1, cap).astype(int)
        ref = ref[indices]
    if len(mov) > cap:
        indices = np.linspace(0, len(mov) - 1, cap).astype(int)
        mov = mov[indices]
    ref_flat = ref.reshape(len(ref), -1)
    mov_flat = mov.reshape(len(mov), -1)
    ref_norm = ref_flat / (np.linalg.norm(ref_flat, axis=1, keepdims=True) + 1e-12)
    mov_norm = mov_flat / (np.linalg.norm(mov_flat, axis=1, keepdims=True) + 1e-12)
    matches = np.argmax(mov_norm @ ref_norm.T, axis=1)
    return ref, mov, matches


def temporal_aligned_psnr(original: np.ndarray, processed: np.ndarray) -> float:
    """时间对齐 PSNR：处理帧逐帧匹配原始最近帧后计算，消除重排/变速错位。"""
    ref, mov, matches = temporal_match(original, processed)
    return matched_psnr(ref, mov, matches)


def matched_psnr(ref: np.ndarray, mov: np.ndarray, matches: np.ndarray) -> float:
    """基于已匹配帧对的 PSNR（供一次性匹配后复用）。"""
    values: list[float] = []
    for frame, index in zip(mov, matches, strict=True):
        candidate = frame
        reference = ref[int(index)]
        if candidate.shape != reference.shape:
            factors = (
                reference.shape[0] / candidate.shape[0],
                reference.shape[1] / candidate.shape[1],
            )
            candidate = zoom(candidate, factors, order=1)
        values.append(psnr(reference, candidate))
    finite = [value for value in values if np.isfinite(value)]
    if not finite:
        return 60.0
    return float(min(np.mean(finite), 60.0))


def temporal_aligned_ssim(original: np.ndarray, processed: np.ndarray) -> float:
    """时间对齐 SSIM：处理帧逐帧匹配原始最近帧后计算结构相似度。"""
    ref, mov, matches = temporal_match(original, processed)
    return matched_ssim(ref, mov, matches)


def matched_ssim(ref: np.ndarray, mov: np.ndarray, matches: np.ndarray) -> float:
    """基于已匹配帧对的 SSIM（供一次性匹配后复用）。"""
    total = 0.0
    for frame, index in zip(mov, matches, strict=True):
        candidate = frame
        reference = ref[int(index)]
        if candidate.shape != reference.shape:
            factors = (
                reference.shape[0] / candidate.shape[0],
                reference.shape[1] / candidate.shape[1],
            )
            candidate = zoom(candidate, factors, order=1)
        total += ssim(reference, candidate)
    return total / max(len(mov), 1)


def temporal_stability(frames: np.ndarray) -> float:
    """画面稳定性：相邻抽样帧的平均绝对差（越小越稳，可检测闪烁/抖动）。"""
    array = np.asarray(frames, dtype=np.float32)
    if len(array) < 2:
        return 0.0
    return float(np.abs(np.diff(array, axis=0)).mean())
