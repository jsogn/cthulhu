"""DCT 域攻击：系数重量化（对 QIM 类水印的有效对抗）。"""

from __future__ import annotations

import numpy as np
from scipy.fftpack import dctn, idctn


def requant_dct(frame: np.ndarray, step: float = 12.0) -> np.ndarray:
    h, w = frame.shape
    frame8 = frame * 255.0
    padded = np.pad(frame8, ((0, -h % 8), (0, -w % 8)), mode="edge")
    blocks = padded.reshape(padded.shape[0] // 8, 8, padded.shape[1] // 8, 8)
    coeffs = dctn(blocks, axes=(1, 3), norm="ortho")
    coeffs = np.round(coeffs / step) * step
    recon = idctn(coeffs, axes=(1, 3), norm="ortho")
    recon = recon.reshape(padded.shape)
    return np.clip(recon[:h, :w] / 255.0, 0, 1)
