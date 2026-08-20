"""协同平均攻击：同一内容的多份不同水印副本求平均。"""

from __future__ import annotations

import numpy as np


def average(copies: list[np.ndarray]) -> np.ndarray:
    return np.mean(np.asarray(copies), axis=0)
