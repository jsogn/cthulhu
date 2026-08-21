"""签名域对抗扰动：对 pHash 低频系数做有预算的边际翻转。

思路：pHash 最终坍缩为「8×8 低频 DCT 系数相对均值的正负号」。与其在像素域
盲目加噪，不如把 resize→DCT→阈值这条链建成可微函数，直接求最小视觉扰动，
把 64 个符号位推向反方向。扰动是低频内容，天然能扛住 32×32 下采样与重编码。
"""

from __future__ import annotations

import numpy as np

from cthulhu_backend.fingerprint import hashes
from cthulhu_backend.similarity import embedding

_DCT_CACHE: dict[int, np.ndarray] = {}
_UP_CACHE: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]] = {}


def _axis_weights(n_in: int, n_out: int) -> np.ndarray:
    """一维双线性插值权重矩阵 (n_out, n_in)。"""
    rows = np.arange(n_out)
    coord = (rows + 0.5) * n_in / n_out - 0.5
    lo = np.clip(np.floor(coord).astype(int), 0, n_in - 1)
    hi = np.clip(lo + 1, 0, n_in - 1)
    frac = coord - np.floor(coord)
    weights = np.zeros((n_out, n_in))
    weights[rows, lo] += 1.0 - frac
    weights[rows, hi] += frac
    return weights


def dct_matrix(n: int) -> np.ndarray:
    """正交 DCT-II 变换矩阵，等价于 scipy.fft.dctn(norm="ortho")。"""
    cached = _DCT_CACHE.get(n)
    if cached is not None:
        return cached
    frequency = np.arange(n)[:, None]
    sample = np.arange(n)[None, :]
    basis = np.cos(np.pi / n * (sample + 0.5) * frequency)
    basis *= np.sqrt(2.0 / n)
    basis[0, :] /= np.sqrt(2.0)
    _DCT_CACHE[n] = basis
    return basis


def _upscale_matrices(size: int, h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    key = (size, h, w)
    cached = _UP_CACHE.get(key)
    if cached is not None:
        return cached
    pair = (_axis_weights(size, h), _axis_weights(size, w))
    _UP_CACHE[key] = pair
    return pair


def attack_phash(
    frame: np.ndarray,
    epsilon: float | None = 0.08,
    flip_fraction: float = 1.0,
    margin: float = 0.15,
    iterations: int = 120,
    outer: int = 4,
    size: int = 32,
    low: int = 8,
) -> tuple[np.ndarray, int, int]:
    """对单帧做 pHash 对抗扰动。

    外层循环每轮用 PIL 精确下采样取得当前 32×32，在其系数域按「最易翻转
    优先」求解符号翻转扰动，双线性上采样后叠加，再进入下一轮修正插值残差。
    扰动是平滑低频内容，可扛下采样与重编码。epsilon 为总扰动幅度的上界
    （None 表示不限制）。返回 (扰动后帧, 0, 翻转位数)。
    """
    x0 = np.clip(np.asarray(frame, dtype=np.float64), 0.0, 1.0)
    original = hashes.phash(x0)
    h, w = x0.shape
    up_y, up_x = _upscale_matrices(size, h, w)
    x = x0.copy()
    for _ in range(outer):
        current = hashes.resize_gray(x, size)
        flipped_small = _flip_coeffs(current, flip_fraction, margin, iterations, low, size)
        full = up_y @ (flipped_small - current) @ up_x.T
        x = np.clip(x + full, 0.0, 1.0)
        if epsilon is not None:
            x = np.clip(x, x0 - epsilon, x0 + epsilon)
    flipped = int(hashes.hamming_bits(original, hashes.phash(x)))
    return x.astype(np.float32), 0, flipped


def _flip_coeffs(
    current: np.ndarray,
    flip_fraction: float,
    margin: float,
    iterations: int,
    low: int,
    size: int,
) -> np.ndarray:
    """在 32×32 系数域求解符号翻转，返回翻转后的空域 32×32。"""
    dct = dct_matrix(size)
    coeffs = dct @ current @ dct.T
    flat = coeffs[:low, :low].ravel()
    margins = flat - flat[1:].mean()
    target_count = max(1, round(low * low * flip_fraction))
    # 跳过 DC（索引 0）：翻转 DC 需要大幅整体调亮/调暗，视觉代价最高且
    # 平台哈希对亮度平移同样不敏感，属于无效开销。
    targets = np.argsort(np.abs(margins[1:]))[: max(target_count - 1, 1)] + 1
    target_mask = np.zeros(low * low, dtype=bool)
    target_mask[targets] = True
    work = coeffs.copy()
    lr = 0.12
    for _ in range(iterations):
        flat = work[:low, :low].ravel()
        threshold = flat[1:].mean()
        margins = flat - threshold
        side = np.where(margins > 0, -1.0, 1.0)
        active = target_mask & (side * margins <= margin)
        if not active.any():
            break
        update = np.zeros(low * low)
        update[active] = side[active] * lr
        work[:low, :low] += update.reshape(low, low)
    return dct.T @ work @ dct.T


def attack_frames(
    frames: np.ndarray,
    epsilon: float = 0.08,
    iterations: int = 120,
) -> np.ndarray:
    """对帧数组逐帧并行施加 pHash 对抗扰动，保持原 dtype。"""
    from cthulhu_backend.transform.parallel import map_frames

    # 彩色帧：只攻击亮度分量，再按比例回写 RGB，保持色相不被破坏。
    if frames.ndim == 4:
        original_dtype = frames.dtype
        work = frames.astype(np.float32) / 255.0 if original_dtype == np.uint8 else frames
        luma = 0.299 * work[..., 0] + 0.587 * work[..., 1] + 0.114 * work[..., 2]
        attacked = map_frames(
            lambda frame: attack_phash(frame, epsilon=epsilon, iterations=iterations)[0],
            luma.astype(np.float32),
        )
        # 亮度均值回填：扰动整体亮度，只保留结构扰动，避免画面变暗。
        attacked = attacked - (
            attacked.mean(axis=(1, 2), keepdims=True) - luma.mean(axis=(1, 2), keepdims=True)
        )
        ratio = np.divide(
            attacked, luma, out=np.ones_like(luma, dtype=np.float32), where=luma > 1e-6
        )
        result = np.clip(work * ratio[..., None], 0.0, 1.0)
        if original_dtype == np.uint8:
            return (result * 255.0).round().astype(np.uint8)
        return result.astype(np.float32)

    if frames.dtype == np.uint8:
        work = frames.astype(np.float32) / 255.0
        attacked = map_frames(
            lambda frame: attack_phash(frame, epsilon=epsilon, iterations=iterations)[0],
            work,
        )
        return (np.clip(attacked, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    return map_frames(
        lambda frame: attack_phash(frame, epsilon=epsilon, iterations=iterations)[0],
        np.asarray(frames, dtype=np.float32),
    )


def _hash_dict(frame: np.ndarray) -> dict[str, np.ndarray]:
    """帧的多哈希签名（与代理评估器同一口径）。"""
    return {
        "phash": hashes.phash(frame),
        "ahash": hashes.ahash(frame),
        "dhash": embedding.dhash(frame).ravel(),
        "dct_sign": hashes.dct_sign(frame),
    }


def _joint_gradient(small: np.ndarray, k: int = 10) -> np.ndarray:
    """多哈希联合梯度：把 pHash/aHash/dHash/DCT 符号的边际损失合到同一目标。"""
    n = small.shape[0]
    dct = dct_matrix(n)
    coeffs = dct @ small @ dct.T
    flat = small.ravel()
    grad = np.zeros_like(small)

    # pHash：低频系数越过「其余系数均值」阈值。
    low = coeffs[:8, :8].ravel()
    phash_margins = low - low[1:].mean()
    for index in np.argsort(np.abs(phash_margins))[:k]:
        row, col = divmod(int(index), 8)
        grad -= np.sign(phash_margins[index]) * np.outer(dct[:, row], dct[:, col])

    # aHash：像素越过全局均值。
    ahash_margins = flat - flat.mean()
    selected = np.argsort(np.abs(ahash_margins))[:k]
    update = np.zeros(flat.size)
    update[selected] = -np.sign(ahash_margins[selected])
    update += update.sum() / flat.size
    grad += update.reshape(small.shape)

    # dHash：水平相邻像素差分越过 0。
    dhash_margins = small[:, 1:] - small[:, :-1]
    for position in np.argsort(np.abs(dhash_margins), axis=None)[:k]:
        row, col = np.unravel_index(int(position), dhash_margins.shape)
        sign = np.sign(dhash_margins[row, col])
        grad[row, col + 1] -= sign
        grad[row, col] += sign

    # DCT 符号：低频系数符号越过 0。
    sign_margins = coeffs[:24, :24].ravel()
    for index in np.argsort(np.abs(sign_margins))[:k]:
        row, col = divmod(int(index), 24)
        grad -= np.sign(sign_margins[index]) * np.outer(dct[:, row], dct[:, col])

    peak = float(np.abs(grad).max()) or 1.0
    return grad / peak


def multi_hash_attack(
    frame: np.ndarray,
    epsilon: float = 0.05,
    inner: int = 100,
    outer: int = 3,
    k: int = 10,
    lr: float = 0.4,
    size: int = 32,
) -> tuple[np.ndarray, int, dict[str, int]]:
    """对单帧做多哈希联合对抗扰动，返回 (帧, 0, 各哈希翻转位数)。"""
    x0 = np.clip(np.asarray(frame, dtype=np.float64), 0.0, 1.0)
    original = _hash_dict(x0)
    h, w = x0.shape
    up_y, up_x = _upscale_matrices(size, h, w)
    x = x0.copy()
    for _ in range(outer):
        base = hashes.resize_gray(x, size)
        small = base.copy()
        for _ in range(inner):
            small = small - lr * _joint_gradient(small, k=k)
            small = np.clip(small, base - epsilon, base + epsilon)
            small = np.clip(small, 0.0, 1.0)
        full = up_y @ (small - base) @ up_x.T
        x = np.clip(x + full, x0 - epsilon, x0 + epsilon)
        x = np.clip(x, 0.0, 1.0)
    flips = {
        key: int(hashes.hamming_bits(original[key], value))
        for key, value in _hash_dict(x).items()
    }
    return x.astype(np.float32), 0, flips


def attack_frames_joint(
    frames: np.ndarray,
    epsilon: float = 0.05,
    iterations: int = 100,
) -> np.ndarray:
    """对帧数组并行施加多哈希联合扰动，保持原 dtype；彩色帧只攻击亮度。"""
    from cthulhu_backend.transform.parallel import map_frames

    if frames.ndim == 4:
        original_dtype = frames.dtype
        work = frames.astype(np.float32) / 255.0 if original_dtype == np.uint8 else frames
        luma = 0.299 * work[..., 0] + 0.587 * work[..., 1] + 0.114 * work[..., 2]
        attacked = map_frames(
            lambda frame: multi_hash_attack(
                frame, epsilon=epsilon, inner=iterations
            )[0],
            luma.astype(np.float32),
        )
        attacked = attacked - (
            attacked.mean(axis=(1, 2), keepdims=True) - luma.mean(axis=(1, 2), keepdims=True)
        )
        ratio = np.divide(
            attacked, luma, out=np.ones_like(luma, dtype=np.float32), where=luma > 1e-6
        )
        result = np.clip(work * ratio[..., None], 0.0, 1.0)
        if original_dtype == np.uint8:
            return (result * 255.0).round().astype(np.uint8)
        return result.astype(np.float32)

    if frames.dtype == np.uint8:
        work = frames.astype(np.float32) / 255.0
        attacked = map_frames(
            lambda frame: multi_hash_attack(
                frame, epsilon=epsilon, inner=iterations
            )[0],
            work,
        )
        return (np.clip(attacked, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    return map_frames(
        lambda frame: multi_hash_attack(frame, epsilon=epsilon, inner=iterations)[0],
        np.asarray(frames, dtype=np.float32),
    )
