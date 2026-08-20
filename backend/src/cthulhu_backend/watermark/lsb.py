"""空域 LSB 水印：最低位平面嵌入，每比特多像素冗余。"""

from __future__ import annotations

import numpy as np

from cthulhu_backend.watermark.common import with_sync

REDUNDANCY = 8


def _pixels(h: int, w: int, n_bits: int, seed: int) -> np.ndarray:
    """为 n_bits 个比特生成冗余像素坐标，形状 (n_bits, REDUNDANCY, 2)。"""
    rng = np.random.default_rng(seed)
    return rng.integers(0, [h, w], size=(n_bits, REDUNDANCY, 2))


def embed(frame: np.ndarray, bits: list[int], seed: int) -> np.ndarray:
    bits = with_sync(bits)
    h, w = frame.shape
    pixels = _pixels(h, w, len(bits), seed)
    out = (frame * 255).round().astype(np.uint8).copy()
    for pos, bit in zip(pixels, bits, strict=True):
        for y, x in pos:
            out[y, x] = (int(out[y, x]) & 0xFE) | bit
    return out.astype(np.float64) / 255.0


def extract(frame: np.ndarray, n_bits: int, seed: int) -> list[int]:
    h, w = frame.shape
    pixels = _pixels(h, w, n_bits + 12, seed)  # 含同步头
    img = (frame * 255).round().astype(np.uint8)
    bits = []
    for pos in pixels:
        vote = sum(int(img[y, x]) & 1 for y, x in pos)
        bits.append(1 if vote > REDUNDANCY / 2 else 0)
    return bits
