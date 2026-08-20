"""DCT 域 QIM 水印：8×8 分块、中频系数奇偶量化索引调制。"""

from __future__ import annotations

import numpy as np
from scipy.fftpack import dctn, idctn

from cthulhu_backend.watermark.common import with_sync

# 8×8 Zigzag 顺序（行优先索引表）。
ZIGZAG = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5,
    12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
]
MID_BAND = ZIGZAG[12:36]  # Zigzag 12~35 位中频系数


def _block_grid(frame: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    h, w = frame.shape
    pad_h, pad_w = -h % 8, -w % 8
    padded = np.pad(frame, ((0, pad_h), (0, pad_w)), mode="edge")
    ph, pw = padded.shape
    return padded.reshape(ph // 8, 8, pw // 8, 8), (h, w)


def embed_synced(frame: np.ndarray, synced_bits: list[int], delta: float = 6.0) -> np.ndarray:
    """把已含同步头的比特序列嵌入单帧（供带纠错冗余的方案复用）。"""
    bits = synced_bits
    frame8 = frame * 255.0
    blocks, (h, w) = _block_grid(frame8)
    coeffs = dctn(blocks, axes=(1, 3), norm="ortho")  # (bh, 8, bw, 8)
    bh, bw = coeffs.shape[0], coeffs.shape[2]

    # 每个块按索引取模承载一个比特，块内 24 个中频系数冗余调制同一比特。
    bits_arr = np.asarray(bits)
    bits_grid = bits_arr[np.arange(bh * bw) % len(bits)].reshape(bh, 1, bw, 1)
    flat = coeffs.transpose(0, 2, 1, 3).reshape(bh, bw, 64)  # 行优先 8×8 展平
    mid = flat[:, :, MID_BAND]  # (bh, bw, 24)
    q = np.round(mid / delta).astype(int)
    q = q - (q % 2) + bits_grid.reshape(bh, bw, 1)
    flat[:, :, MID_BAND] = q * delta
    coeffs = flat.reshape(bh, bw, 8, 8).transpose(0, 2, 1, 3)

    recon = idctn(coeffs, axes=(1, 3), norm="ortho")
    recon = recon.reshape(coeffs.shape[0] * 8, coeffs.shape[2] * 8)
    return np.clip(recon[:h, :w] / 255.0, 0, 1)


def embed(frame: np.ndarray, bits: list[int], delta: float = 6.0) -> np.ndarray:
    return embed_synced(frame, with_sync(bits), delta)


def extract_total(frame: np.ndarray, total: int, delta: float = 6.0) -> list[int]:
    """提取 total 位（含同步头）的比特序列。"""
    blocks, _ = _block_grid(frame * 255.0)
    coeffs = dctn(blocks, axes=(1, 3), norm="ortho")
    bh, bw = coeffs.shape[0], coeffs.shape[2]
    flat = coeffs.transpose(0, 2, 1, 3).reshape(bh, bw, 64)
    mid = flat[:, :, MID_BAND]
    q = np.round(mid / delta).astype(int)
    # 块内 24 个系数多数表决 → 块比特，再按索引取模聚合为 total 位。
    block_bits = (np.mean(q % 2, axis=2) >= 0.5).astype(int).reshape(-1)
    out: list[int] = []
    for k in range(total):
        votes = block_bits[k::total]
        out.append(1 if len(votes) and np.mean(votes) >= 0.5 else 0)
    return out


def extract(frame: np.ndarray, n_bits: int, delta: float = 6.0) -> list[int]:
    return extract_total(frame, len(with_sync([0] * n_bits)), delta)
