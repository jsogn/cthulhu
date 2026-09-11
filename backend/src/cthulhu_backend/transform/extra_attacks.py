"""底层对抗原语：把研究 harness 里已有的攻击能力接入清洗管线。

所有原语均为可选、可组合、确定性（同一 seed 下结果一致），灰度/彩色
帧通用。攻击顺序：几何（镜像/平移抖动）→ 空域（中值/噪声/像素重量化）
→ 频域（DCT 重量化）→ 色度量化 → 时域（抽帧复制）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import median_filter

from cthulhu_backend.quality import ssim as quality_ssim

_DCT8_CACHE: np.ndarray | None = None


def _dct8() -> np.ndarray:
    """8×8 正交 DCT-II 矩阵（与 scipy norm="ortho" 一致），模块级缓存。"""
    global _DCT8_CACHE
    if _DCT8_CACHE is None:
        frequency = np.arange(8)[:, None]
        sample = np.arange(8)[None, :]
        basis = np.cos(np.pi / 8 * (sample + 0.5) * frequency)
        basis *= np.sqrt(2.0 / 8)
        basis[0, :] /= np.sqrt(2.0)
        _DCT8_CACHE = basis.astype(np.float32)
    return _DCT8_CACHE


@dataclass
class AssaultParams:
    """底层攻击组合参数；零值表示关闭对应原语。"""

    jitter: float = 0.0
    perspective: float = 0.0
    warp: float = 0.0
    median: int = 0
    noise: float = 0.0
    requant: int = 0
    dct_step: float = 0.0
    temporal_sub: float = 0.0
    fft_phase: float = 0.0
    dwt_detail: float = 0.0
    copy_field: np.ndarray | None = None
    face_field: np.ndarray | None = None
    face_perturb: float = 0.0
    copy_attack: float = 0.0

    @property
    def enabled(self) -> bool:
        return any(
            (
                self.jitter > 0,
                self.perspective > 0,
                self.warp > 0,
                self.median > 0,
                self.noise > 0,
                self.requant > 0,
                self.dct_step > 0,
                self.temporal_sub > 0,
                self.fft_phase > 0,
                self.dwt_detail > 0,
                self.face_perturb > 0,
                self.copy_attack > 0,
            )
        )


def quality_gate(
    original: np.ndarray,
    attacked: np.ndarray,
    psnr_target: float,
    ssim_target: float,
) -> np.ndarray:
    """画质门控：把攻击结果按全局系数回退到满足 PSNR/SSIM 目标的最强程度。

    攻击 delta 方向不变，仅整体缩放：PSNR 对缩放系数单调，可闭式求解；
    若 SSIM 仍不达标，再在 [0,k] 上二分回退（系数越小越接近原帧，SSIM
    单调趋近 1）。SSIM 在「帧抽样 + 空间降采样」的代理上评估，避免整块
    高斯滤波的内存与耗时；攻击本身已足够温和时（达标）原样返回，不做放大。
    """
    original_dtype = original.dtype
    ref = original.astype(np.float32)
    cand = attacked.astype(np.float32)
    if original_dtype == np.uint8:
        ref /= 255.0
        cand /= 255.0
    delta = cand - ref
    mse_delta = float(np.mean(delta**2))
    if mse_delta <= 1e-12:
        return attacked
    target_mse = 10.0 ** (-psnr_target / 10.0)
    if mse_delta <= target_mse and _ssim_proxy(ref, cand) >= ssim_target:
        return attacked
    # 0.995 余量吸收 float32 舍入，保证门控后 PSNR 稳定不低于目标。
    k = min(1.0, float(np.sqrt(target_mse / mse_delta)) * 0.995)
    out = ref + k * delta
    if _ssim_proxy(ref, out) < ssim_target:
        lo, hi = 0.0, k
        for _ in range(6):
            mid = (lo + hi) / 2.0
            if _ssim_proxy(ref, ref + mid * delta) >= ssim_target:
                lo = mid
            else:
                hi = mid
        out = ref + lo * delta
    out = np.clip(out, 0.0, 1.0)
    if original_dtype == np.uint8:
        return (out * 255.0).round().astype(np.uint8)
    return out.astype(original_dtype)


def _ssim_proxy(
    reference: np.ndarray,
    candidate: np.ndarray,
    max_frames: int = 8,
    max_edge: int = 256,
) -> float:
    """SSIM 代理：帧抽样 + 空间降采样，控制块级评估的内存与耗时。"""
    from scipy.ndimage import zoom

    ref = np.asarray(reference, dtype=np.float32)
    cand = np.asarray(candidate, dtype=np.float32)
    if ref.ndim >= 3 and len(ref) > max_frames:
        indices = np.linspace(0, len(ref) - 1, max_frames).astype(int)
        ref = ref[indices]
        cand = cand[indices]
    if ref.ndim == 4:
        h, w = ref.shape[1], ref.shape[2]
        scale = min(1.0, max_edge / max(h, w))
        if scale < 1.0:
            ref = zoom(ref, (1.0, scale, scale, 1.0), order=1)
            cand = zoom(cand, (1.0, scale, scale, 1.0), order=1)
    elif ref.ndim == 3:
        h, w = ref.shape[1], ref.shape[2]
        scale = min(1.0, max_edge / max(h, w))
        if scale < 1.0:
            ref = zoom(ref, (1.0, scale, scale), order=1)
            cand = zoom(cand, (1.0, scale, scale), order=1)
    return quality_ssim(ref, cand)


def _spatial_kernel(shape: tuple[int, ...], size: int) -> tuple[int, ...]:
    """把灰度 (H,W) 的滤波尺寸推广到彩色 (H,W,3)（色通道不做滤波）。"""
    if len(shape) == 2:
        return (size, size)
    return (size, size, 1)


def median(frames: np.ndarray, size: int) -> np.ndarray:
    if size <= 0:
        return frames
    # PIL 的 MedianFilter 是 C 实现，比 scipy 快一个数量级；浮点帧先转 uint8。
    if frames.dtype == np.uint8:
        from PIL import Image, ImageFilter

        from cthulhu_backend.parallel import map_frames

        def filt(frame: np.ndarray) -> np.ndarray:
            mode = "RGB" if frame.ndim == 3 else "L"
            image = Image.fromarray(frame, mode=mode)
            return np.asarray(image.filter(ImageFilter.MedianFilter(size=size)), dtype=np.uint8)

        return map_frames(filt, frames)
    from cthulhu_backend.parallel import map_frames

    return map_frames(
        lambda frame: median_filter(frame, size=_spatial_kernel(frame.shape, size)),
        frames,
    )


def gaussian_noise(frames: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    if sigma <= 0:
        return frames
    original_dtype = frames.dtype
    # uint8 域直接加减少量整数扰动，内存带宽降到 float32 的四分之一。
    if original_dtype == np.uint8:
        delta = rng.integers(-1, 2, frames.shape, dtype=np.int8).astype(np.int16)
        work = frames.astype(np.int16) + delta * max(1, round(sigma * 255))
        return np.clip(work, 0, 255).astype(np.uint8)
    work = frames.astype(np.float32)
    delta = rng.integers(-1, 2, frames.shape, dtype=np.int8).astype(np.float32) * sigma
    return np.clip(work + delta, 0.0, 1.0).astype(original_dtype)


def pixel_requant(frames: np.ndarray, levels: int) -> np.ndarray:
    if levels <= 1:
        return frames
    original_dtype = frames.dtype
    if original_dtype == np.uint8:
        # 查表实现：256 项一次算好，逐像素查表比 float 往返快且零精度损失。
        scale = (levels - 1) / 255.0
        values = np.rint(np.arange(256, dtype=np.float32) * scale) / scale
        table = np.rint(values).clip(0, 255).astype(np.uint8)
        return table[frames]
    work = frames.astype(np.float32)
    requantized = np.round(work * (levels - 1)) / (levels - 1)
    return requantized.astype(original_dtype)


def dct_requant_plane(work: np.ndarray, step: float, sub_batch: int = 32) -> np.ndarray:
    """对 (F,H,W) float32 平面做 8×8 DCT 重量化（子批 matmul，内存受控）。"""
    h, w = work.shape[1:3]
    # 可选原生加速：未启用/库缺失/尺寸不满足时返回 None，走 numpy 兜底。
    from cthulhu_backend import native_dct

    fast = native_dct.requant_plane(work, step)
    if fast is not None:
        return fast
    pad_h, pad_w = -h % 8, -w % 8
    if pad_h or pad_w:
        work = np.pad(work, ((0, 0), (0, pad_h), (0, pad_w)), mode="edge")
    ph, pw = work.shape[1:3]
    matrix = _dct8()
    out = np.empty_like(work)
    jobs = [
        (start, min(start + sub_batch, len(work)))
        for start in range(0, len(work), sub_batch)
    ]

    def run(job: tuple[int, int]) -> tuple[int, int, np.ndarray]:
        start, end = job
        sub = work[start:end]
        blocks = sub.reshape(len(sub), ph // 8, 8, pw // 8, 8)
        stacked = blocks.transpose(0, 1, 3, 2, 4).reshape(-1, 8, 8)
        tmp = np.matmul(matrix, stacked)
        coeffs = np.matmul(tmp, matrix.T)
        coeffs = np.round(coeffs / step) * step
        np.matmul(matrix.T, coeffs, out=tmp)
        recon = np.matmul(tmp, matrix)
        recon_frame = recon.reshape(len(sub), ph // 8, pw // 8, 8, 8).transpose(
            0, 1, 3, 2, 4
        ).reshape(len(sub), ph, pw)
        return start, end, recon_frame

    from concurrent.futures import ThreadPoolExecutor

    from cthulhu_backend import parallel

    # 与逐帧武器共用同一个线程旋钮（默认 4：子批 matmul 再多收益有限）。
    with ThreadPoolExecutor(max_workers=parallel.configured_workers(4)) as pool:
        for start, end, recon_frame in pool.map(run, jobs):
            out[start:end] = recon_frame
    return out[:, :h, :w]


def dct_requant(frames: np.ndarray, step: float) -> np.ndarray:
    """DCT 重量化：彩色帧只对亮度做重量化、差值叠加回三通道。

    免去逐通道三份 DCT 与 YCbCr 往返，仅为亮度加权的两次全图加减；
    灰度帧保持原有单平面路径。"""
    if step <= 0:
        return frames
    original_dtype = frames.dtype
    work = np.asarray(frames, dtype=np.float32)
    if original_dtype != np.uint8:
        # float [0,1] 域：DCT 线性，量化步长同步除以 255，免去全图 ×255 往返。
        effective_step = step / 255.0
        if work.ndim == 3:
            return np.clip(dct_requant_plane(work, effective_step), 0.0, 1.0).astype(
                original_dtype
            )
        luma = 0.299 * work[..., 0] + 0.587 * work[..., 1] + 0.114 * work[..., 2]
        new_luma = dct_requant_plane(luma, effective_step)
        return np.clip(work + (new_luma - luma)[..., None], 0.0, 1.0).astype(original_dtype)

    if work.ndim == 3:
        result = np.clip(dct_requant_plane(work, step) / 255.0, 0.0, 1.0)
        return (result * 255.0).round().astype(np.uint8)
    luma = 0.299 * work[..., 0] + 0.587 * work[..., 1] + 0.114 * work[..., 2]
    new_luma = dct_requant_plane(luma, step)
    result = np.clip(work + (new_luma - luma)[..., None], 0.0, 255.0)
    return result.round().astype(np.uint8)


def perspective_shear(
    frames: np.ndarray,
    strength: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """逐帧透视剪切：每行/列随位置线性平移，模拟轻微梯形畸变。

    比全量单应变换便宜（两次双线性采样），足以破坏块对齐与几何同步，
    又不会像大角度旋转那样引人注意。
    """
    if strength <= 0:
        return frames
    from cthulhu_backend.parallel import map_frames

    is_u8 = frames.dtype == np.uint8
    work = frames.astype(np.float32)
    if is_u8:
        work /= 255.0
    kx = rng.uniform(-strength, strength, len(work))
    ky = rng.uniform(-strength, strength, len(work))

    def shear_one(pair: tuple[np.ndarray, float, float]) -> np.ndarray:
        frame, shear_x, shear_y = pair
        h, w = frame.shape[:2]
        rows = np.arange(h, dtype=np.float32) / h - 0.5
        cols = np.arange(w, dtype=np.float32) / w - 0.5
        x = np.arange(w, dtype=np.float32)[None, :] - (shear_x * rows * w)[:, None]
        y = np.arange(h, dtype=np.float32)[:, None] - (shear_y * cols * h)[None, :]
        x0 = np.clip(np.floor(x).astype(int), 0, w - 1)
        x1 = np.clip(x0 + 1, 0, w - 1)
        y0 = np.clip(np.floor(y).astype(int), 0, h - 1)
        y1 = np.clip(y0 + 1, 0, h - 1)
        fx = (x - x0).astype(np.float32)
        fy = (y - y0).astype(np.float32)
        if frame.ndim == 3:
            fx = fx[..., None]
            fy = fy[..., None]
        top = frame[y0, x0] * (1 - fx) + frame[y0, x1] * fx
        bottom = frame[y1, x0] * (1 - fx) + frame[y1, x1] * fx
        return np.clip(top * (1 - fy) + bottom * fy, 0.0, 1.0)

    result = map_frames(shear_one, list(zip(work, kx, ky)))
    if is_u8:
        return (result * 255.0).round().astype(np.uint8)
    return result.astype(frames.dtype)


def local_warp(
    frames: np.ndarray,
    strength: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """局部平滑扭曲：粗网格随机位移上采样为平滑光流，逐帧重采样。

    模拟镜头微畸变/水波效果，破坏任何依赖精确空间对齐的检测；
    位移场平滑连续，视觉上接近不可见。
    """
    if strength <= 0:
        return frames
    from scipy.ndimage import map_coordinates, zoom

    from cthulhu_backend.parallel import map_frames

    is_u8 = frames.dtype == np.uint8
    work = frames.astype(np.float32)
    if is_u8:
        work /= 255.0
    h, w = work.shape[1:3]
    span = strength * min(h, w)
    grid = (5, 8)
    fields = rng.uniform(-1.0, 1.0, (len(work), 2, grid[0], grid[1])).astype(np.float32) * span

    def warp_one(pair: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        frame, field = pair
        dy = zoom(field[0], (h / grid[0], w / grid[1]), order=1)
        dx = zoom(field[1], (h / grid[0], w / grid[1]), order=1)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        coords = np.stack([yy + dy, xx + dx])
        if frame.ndim == 3:
            return np.stack(
                [map_coordinates(frame[..., c], coords, order=1, mode="nearest") for c in range(3)],
                axis=-1,
            )
        return map_coordinates(frame, coords, order=1, mode="nearest")

    result = map_frames(warp_one, list(zip(work, fields)))
    if is_u8:
        return (np.clip(result, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    return np.clip(result, 0.0, 1.0).astype(frames.dtype)


def jitter_offsets(
    count: int,
    width: int,
    height: int,
    jitter: float,
    rng: np.random.Generator,
    trajectory: str = "noise",
) -> tuple[np.ndarray, np.ndarray]:
    """生成逐帧平移偏移序列（x/y，单位像素，整数）。

    ``noise`` 为历史白噪声轨迹：相邻帧位移完全独立，高频晃动观感明显、
    易诱发晕眩。``sine`` 是约 8 秒周期的低频正弦漂移：相邻帧位移差远低于
    人眼可感知的帧间运动阈值，
    观感接近缓慢机位漂移，但每帧仍相对源片整体错位，保留空间对齐破坏力。
    垂直幅度保持与水平一致：SS/DFT 按行提取，纵向错位不可压缩。
    """
    if jitter <= 0 or count <= 0:
        return np.zeros(max(count, 0), dtype=int), np.zeros(max(count, 0), dtype=int)
    ax = jitter * width
    ay = jitter * height
    if trajectory == "sine":
        # 周期与 rotate_de_sync 的低频轨迹一致：约 8 秒 @25fps。
        period = 200
        phase_x = float(rng.uniform(0, 2 * np.pi))
        phase_y = float(rng.uniform(0, 2 * np.pi))
        t = np.arange(count)
        dx = np.round(ax * np.sin(2 * np.pi * t / period + phase_x)).astype(int)
        dy = np.round(ay * np.sin(2 * np.pi * t / period + phase_y)).astype(int)
        return dx, dy
    if trajectory != "noise":
        raise ValueError(f"未知抖动轨迹：{trajectory}")
    draws = rng.uniform(-jitter, jitter, count * 2)
    dx = np.round(draws[0::2] * width).astype(int)
    dy = np.round(draws[1::2] * height).astype(int)
    return dx, dy


def translate_jitter(
    frames: np.ndarray,
    jitter: float,
    rng: np.random.Generator,
    trajectory: str = "noise",
) -> np.ndarray:
    """逐帧平移抖动：随机裁剪偏移后缩回原尺寸。

    默认 ``noise`` 保留历史白噪声行为；``sine`` 为低频平滑轨迹，
    观感轻微但仍逐帧破坏与源片的空间对齐。偏移序列一次性生成，避免并行
    帧处理时共享 RNG 的取数顺序不确定。
    """
    if jitter <= 0:
        return frames
    from PIL import Image

    from cthulhu_backend.parallel import map_frames

    is_u8 = frames.dtype == np.uint8
    height, width = frames.shape[1:3]
    offsets = jitter_offsets(
        len(frames), width, height, jitter, rng, trajectory=trajectory,
    )

    def shift_one(pair: tuple[int, np.ndarray]) -> np.ndarray:
        index, frame = pair
        dx = int(offsets[0][index])
        dy = int(offsets[1][index])
        h, w = frame.shape[:2]
        box = (
            max(0, -dx),
            max(0, -dy),
            min(w, w - dx),
            min(h, h - dy),
        )
        mode = "RGB" if frame.ndim == 3 else "L"
        if is_u8:
            image = Image.fromarray(frame, mode=mode)
        else:
            image = Image.fromarray((np.clip(frame, 0, 1) * 255).round().astype(np.uint8), mode=mode)
        shifted = image.crop(box).resize((w, h), Image.BILINEAR)
        result = np.asarray(shifted)
        if is_u8:
            return result.astype(np.uint8)
        return result.astype(np.float32) / 255.0

    return map_frames(shift_one, list(enumerate(frames)))


def apply(
    frames: np.ndarray,
    params: AssaultParams,
    rng: np.random.Generator,
) -> np.ndarray:
    """按固定顺序施加组合攻击，全部为可选开关。"""
    from cthulhu_backend.transform import regenerate

    work = frames
    if params.jitter > 0:
        work = translate_jitter(work, params.jitter, rng, trajectory="sine")
    if params.perspective > 0:
        work = perspective_shear(work, params.perspective, rng)
    if params.warp > 0:
        work = local_warp(work, params.warp, rng)
    if params.median > 0:
        work = median(work, params.median)
    if params.noise > 0:
        work = gaussian_noise(work, params.noise, rng)
    if params.requant > 0:
        work = pixel_requant(work, params.requant)
    if params.dct_step > 0:
        work = dct_requant(work, params.dct_step)
    if params.temporal_sub > 0:
        work = regenerate.temporal_subtract(work, params.temporal_sub)
    if params.fft_phase > 0:
        work = regenerate.fft_phase(work, params.fft_phase, rng)
    if params.dwt_detail > 0:
        work = regenerate.dwt_detail(work, params.dwt_detail, rng)
    if params.copy_attack > 0:
        work = regenerate.copy_attack(work, params.copy_attack, rng, field=params.copy_field)
    if params.face_perturb > 0:
        work = regenerate.face_perturb(work, params.face_perturb, rng, field=params.face_field)
    return work
