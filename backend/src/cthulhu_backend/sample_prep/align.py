"""FFT 相位相关配准：估计平移、对齐并裁出公共区域。"""

from __future__ import annotations

import numpy as np


def estimate_shift(reference: np.ndarray, moving: np.ndarray) -> tuple[int, int]:
    """相位相关估计对齐校正量 (dy, dx)：np.roll(moving, (dy, dx)) 后与 reference 重合。"""
    ref = np.asarray(reference, dtype=np.float64)
    mov = np.asarray(moving, dtype=np.float64)
    fft_ref = np.fft.fft2(ref)
    fft_mov = np.fft.fft2(mov)
    cross = fft_ref * fft_mov.conj()
    cross /= np.maximum(np.abs(cross), 1e-12)
    corr = np.fft.ifft2(cross).real
    dy, dx = np.unravel_index(int(np.argmax(corr)), corr.shape)
    if dy > ref.shape[0] // 2:
        dy -= ref.shape[0]
    if dx > ref.shape[1] // 2:
        dx -= ref.shape[1]
    return dy, dx


def align_pair(
    reference: np.ndarray,
    moving: np.ndarray,
    shift: tuple[int, int] | None = None,
    border: int = 8,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """按平移对齐 moving 并裁出共同区域，返回 (ref_crop, moving_crop, shift)。"""
    ref = np.asarray(reference, dtype=np.float64)
    mov = np.asarray(moving, dtype=np.float64)
    dy, dx = shift if shift is not None else estimate_shift(ref, mov)
    rolled = np.roll(np.roll(mov, dy, axis=0), dx, axis=1)
    margin = max(border, abs(dy) + 1, abs(dx) + 1)
    end_y = ref.shape[0] - margin if ref.shape[0] > 2 * margin else None
    end_x = ref.shape[1] - margin if ref.shape[1] > 2 * margin else None
    return ref[margin:end_y, margin:end_x], rolled[margin:end_y, margin:end_x], (dy, dx)


def align_videos(
    reference_frames: np.ndarray,
    moving_frames: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """用中位帧估计平移，逐帧对齐并裁公共区域。"""
    ref = np.asarray(reference_frames, dtype=np.float64)
    mov = np.asarray(moving_frames, dtype=np.float64)
    shift = estimate_shift(np.median(ref, axis=0), np.median(mov, axis=0))
    aligned_ref = np.stack([align_pair(ref[i], mov[i], shift)[0] for i in range(min(len(ref), len(mov)))])
    aligned_mov = np.stack([align_pair(ref[i], mov[i], shift)[1] for i in range(min(len(ref), len(mov)))])
    return aligned_ref, aligned_mov, shift
