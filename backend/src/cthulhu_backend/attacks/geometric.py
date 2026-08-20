"""几何攻击：裁剪 + 旋转 + 缩放，破坏块对齐与同步码。"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import rotate, zoom


def crop_rotate_rescale(
    frames: np.ndarray,
    crop_frac: float = 0.05,
    angle: float = 0.5,
    rescale: float = 0.99,
) -> np.ndarray:
    out = []
    h, w = frames.shape[1:]
    cy0, cy1 = int(h * crop_frac), h - int(h * crop_frac)
    cx0, cx1 = int(w * crop_frac), w - int(w * crop_frac)
    for frame in frames:
        cropped = frame[cy0:cy1, cx0:cx1]
        rotated = rotate(cropped, angle, reshape=False, order=1, mode="nearest")
        zoomed = zoom(rotated, (h / rotated.shape[0], w / rotated.shape[1]), order=1)
        out.append(np.clip(zoomed[:h, :w], 0, 1))
    return np.asarray(out)
