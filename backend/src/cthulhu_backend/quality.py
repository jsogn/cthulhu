"""客观指标原语：PSNR / SSIM / BER。

放在中立层的原因：生产侧（`transform` 的画质门控、`similarity` 的相似度报告）
与评估侧（`evaluate`）都要用这几支纯函数。它们此前住在 `evaluate/metrics.py`，
导致 `evaluate ⇄ fingerprint ⇄ similarity` 包级互依，只能靠函数内延迟导入破环
（审计 R5）。现在两侧都只依赖本模块，依赖方向恢复单向。

`evaluate.metrics` 继续 re-export 这三个名字，历史调用方与测试不受影响；
依赖 ffmpeg/对齐的指标（VMAF、对齐 PSNR/SSIM、时序稳定性等）仍留在 evaluate。
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

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
    """比特错误率（带同步头对齐搜索，见 watermark.common.bit_error_rate）。"""
    return bit_error_rate(ref_bits, out_bits)
