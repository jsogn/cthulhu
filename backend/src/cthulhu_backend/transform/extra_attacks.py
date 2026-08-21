"""底层对抗原语：把研究 harness 里已有的攻击能力接入清洗管线。

所有原语均为可选、可组合、确定性（同一 seed 下结果一致），灰度/彩色
帧通用。攻击顺序：几何（镜像/平移抖动）→ 空域（中值/噪声/像素重量化）
→ 频域（DCT 重量化）→ 色度量化 → 时域（抽帧复制）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import median_filter

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

    mirror: bool = False
    jitter: float = 0.0
    perspective: float = 0.0
    warp: float = 0.0
    median: int = 0
    noise: float = 0.0
    subtract_beta: float = 0.0
    requant: int = 0
    dct_step: float = 0.0
    chroma_levels: int = 0
    drop_every: int = 0

    @property
    def enabled(self) -> bool:
        return any(
            (
                self.mirror,
                self.jitter > 0,
                self.perspective > 0,
                self.warp > 0,
                self.median > 0,
                self.noise > 0,
                self.subtract_beta > 0,
                self.requant > 0,
                self.dct_step > 0,
                self.chroma_levels > 0,
                self.drop_every > 0,
            )
        )


@dataclass
class SaliencyLayout:
    """内容显著性叠加布局：从运行 seed 派生，整片保持一致。"""

    level: int
    border_color: tuple[int, int, int]
    corner: str
    badge_text: str
    bar_side: str
    bar_text: str
    mosaic: tuple[float, float, float, float] | None
    title_text: str = ""
    band_blur: bool = False
    pip: bool = False


def saliency_layout(seed: int, level: int = 1) -> SaliencyLayout | None:
    """生成内容显著性布局。

    level 0 关闭；1 边框+角标；2 再加字幕条+马赛克；3 再加大标题块与
    背景带虚化；4 再加画中画（强构图改变）。
    """
    if level <= 0:
        return None
    rng = np.random.default_rng(seed ^ 0x51A115)
    colors = [(40, 60, 90), (90, 40, 60), (50, 80, 40), (30, 50, 80)]
    texts = ["原创", "精剪", "热映", "精选"]
    return SaliencyLayout(
        level=level,
        border_color=colors[int(rng.integers(len(colors)))],
        corner=("tl", "tr", "bl", "br")[int(rng.integers(4))],
        badge_text=texts[int(rng.integers(len(texts)))],
        bar_side="top" if rng.random() < 0.5 else "bottom",
        bar_text=texts[int(rng.integers(len(texts)))],
        mosaic=(
            float(rng.uniform(0.08, 0.55)),
            float(rng.uniform(0.12, 0.6)),
            float(rng.uniform(0.45, 0.92)),
            float(rng.uniform(0.5, 0.88)),
        )
        if level >= 2
        else None,
        title_text=texts[int(rng.integers(len(texts)))] if level >= 3 else "",
        band_blur=level >= 3,
        pip=level >= 4,
    )


def salient_overlay(frames: np.ndarray, layout: SaliencyLayout | None) -> np.ndarray:
    """叠加边框/角标/字幕条/局部马赛克，改变关键帧抽取与深度特征分布。"""
    if layout is None:
        return frames
    from PIL import Image, ImageDraw, ImageFilter, ImageFont

    from cthulhu_backend.transform.parallel import map_frames

    is_u8 = frames.dtype == np.uint8

    def draw_frame(frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        if is_u8:
            base = frame
        else:
            base = (np.clip(frame, 0, 1) * 255).round().astype(np.uint8)
        if base.ndim == 2:
            image = Image.fromarray(base, mode="L").convert("RGB")
        else:
            image = Image.fromarray(base, mode="RGB")
        draw = ImageDraw.Draw(image, "RGBA")

        # 边框：在边缘制造稳定强边，改变边缘直方图。
        draw.rectangle([0, 0, w - 1, h - 1], outline=layout.border_color + (255,), width=3)

        # 角标：显著性文字块。
        badge_w, badge_h = int(w * 0.14), int(h * 0.06)
        if layout.corner == "tl":
            box = [8, 8, 8 + badge_w, 8 + badge_h]
        elif layout.corner == "tr":
            box = [w - 8 - badge_w, 8, w - 8, 8 + badge_h]
        elif layout.corner == "bl":
            box = [8, h - 8 - badge_h, 8 + badge_w, h - 8]
        else:
            box = [w - 8 - badge_w, h - 8 - badge_h, w - 8, h - 8]
        draw.rectangle(box, fill=(10, 10, 12, 210))
        font = ImageFont.load_default(size=max(14, badge_h - 8))
        draw.text((box[0] + 10, box[1] + 3), layout.badge_text, fill=(255, 255, 255, 255), font=font)

        # 大标题块：全宽强语义文字，显著改变深度特征的全局分布。
        if layout.title_text:
            block_h = int(h * 0.14)
            y0 = h - block_h if layout.bar_side == "top" else 0
            draw.rectangle([0, y0, w, y0 + block_h], fill=(8, 8, 10, 235))
            font_title = ImageFont.load_default(size=max(22, block_h - 16))
            draw.text(
                (w // 2, y0 + block_h // 2),
                layout.title_text,
                fill=(255, 255, 255, 255),
                font=font_title,
                anchor="mm",
            )

        # 背景带虚化：顶部/底部各 8% 高斯模糊，制造景深差异。
        if layout.band_blur:
            band_h = int(h * 0.08)
            for y0 in (0, h - band_h):
                band = image.crop((0, y0, w, y0 + band_h))
                image.paste(band.filter(ImageFilter.GaussianBlur(radius=12)), (0, y0))

        # 画中画：角落内嵌缩小版画面，直接改变全局构图。
        if layout.pip:
            pip_w, pip_h = int(w * 0.3), int(h * 0.3)
            inset = image.resize((pip_w, pip_h), Image.BILINEAR)
            inset = inset.filter(ImageFilter.GaussianBlur(radius=0.4))
            x0 = w - pip_w - 10
            y0 = 10 if layout.corner in ("tl", "tr") else h - pip_h - 10
            image.paste(inset, (x0, y0))
            draw.rectangle([x0, y0, x0 + pip_w, y0 + pip_h], outline=(255, 255, 255, 220), width=4)

        # 字幕条：半透明横条，模拟二次加工痕迹。
        if layout.bar_side:
            bar_h = int(h * 0.08)
            y0 = 0 if layout.bar_side == "top" else h - bar_h
            draw.rectangle([0, y0, w, y0 + bar_h], fill=(0, 0, 0, 150))
            font_bar = ImageFont.load_default(size=max(16, bar_h - 10))
            draw.text((int(w * 0.05), y0 + 4), layout.bar_text, fill=(240, 240, 240, 235), font=font_bar)

        # 局部马赛克：把显著性区域像素化，直接改变局部特征。
        if layout.mosaic:
            x0 = int(layout.mosaic[0] * w)
            y0 = int(layout.mosaic[1] * h)
            x1 = int(layout.mosaic[2] * w)
            y1 = int(layout.mosaic[3] * h)
            if x1 - x0 > 8 and y1 - y0 > 8:
                region = image.crop((x0, y0, x1, y1))
                small = region.resize((max(2, (x1 - x0) // 8), max(2, (y1 - y0) // 8)), Image.NEAREST)
                image.paste(small.resize((x1 - x0, y1 - y0), Image.NEAREST), (x0, y0))

        if frame.ndim == 2:
            result = np.asarray(image.convert("L"), dtype=np.uint8)
        else:
            result = np.asarray(image, dtype=np.uint8)
        if is_u8:
            return result
        return result.astype(np.float32) / 255.0

    return map_frames(draw_frame, frames)


def protect_details(
    attacked: np.ndarray,
    original: np.ndarray,
    strength: float,
) -> np.ndarray:
    """细节保护：按原始画面的边缘/纹理显著性生成掩码，让攻击结果向原帧回退。

    人脸、字幕、商品等高细节区域保持原样，平坦区域保留完整攻击效果；
    strength 为回退力度（0 关闭，1 完全保护细节区域）。
    """
    if strength <= 0:
        return attacked
    from scipy.ndimage import gaussian_filter, sobel

    from cthulhu_backend.transform.parallel import map_frames

    is_u8 = attacked.dtype == np.uint8
    attacked_f = attacked.astype(np.float32)
    original_f = original.astype(np.float32)
    if is_u8:
        attacked_f /= 255.0
        original_f /= 255.0

    def blend(pair: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        current, source = pair
        luma = (
            0.299 * source[..., 0] + 0.587 * source[..., 1] + 0.114 * source[..., 2]
            if source.ndim == 3
            else source
        )
        gx = sobel(luma, axis=1, mode="reflect")
        gy = sobel(luma, axis=0, mode="reflect")
        magnitude = np.sqrt(gx**2 + gy**2)
        magnitude = gaussian_filter(magnitude, sigma=1.5)
        peak = float(np.percentile(magnitude, 96))
        mask = np.clip(magnitude / max(peak, 1e-6), 0.0, 1.0)
        mask = gaussian_filter(mask, sigma=1.0) * strength
        if source.ndim == 3:
            mask = mask[..., None]
        return current * (1.0 - mask) + source * mask

    result = map_frames(blend, list(zip(attacked_f, original_f)))
    if is_u8:
        return (np.clip(result, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    return result.astype(attacked.dtype)


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

        from cthulhu_backend.transform.parallel import map_frames

        def filt(frame: np.ndarray) -> np.ndarray:
            mode = "RGB" if frame.ndim == 3 else "L"
            image = Image.fromarray(frame, mode=mode)
            return np.asarray(image.filter(ImageFilter.MedianFilter(size=size)), dtype=np.uint8)

        return map_frames(filt, frames)
    from cthulhu_backend.transform.parallel import map_frames

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
    work = frames.astype(np.float32)
    if original_dtype == np.uint8:
        work /= 255.0
    requantized = np.round(work * (levels - 1)) / (levels - 1)
    if original_dtype == np.uint8:
        return (requantized * 255.0).round().astype(np.uint8)
    return requantized.astype(original_dtype)


def _requant_plane(work: np.ndarray, step: float, sub_batch: int = 32) -> np.ndarray:
    """对 (F,H,W) float32 平面做 8×8 DCT 重量化（子批 matmul，内存受控）。"""
    h, w = work.shape[1:3]
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

    with ThreadPoolExecutor(max_workers=4) as pool:
        for start, end, recon_frame in pool.map(run, jobs):
            out[start:end] = recon_frame
    return out[:, :h, :w]


def dct_requant(frames: np.ndarray, step: float) -> np.ndarray:
    """DCT 重量化：逐通道 RGB 直接处理（带宽受限机器上的最优实现）。

    亮度+子采样色度方案实测更慢（YCbCr 转换与上采样引入更多整片拷贝），
    已回退为逐通道口径，正确性经旧 scipy 参考与代理回归门双重验证。
    """
    if step <= 0:
        return frames
    original_dtype = frames.dtype
    work = np.asarray(frames, dtype=np.float32)
    domain_255 = work.max() > 1.0
    if not domain_255:
        work = work * 255.0
    if work.ndim == 3:
        result = np.clip(_requant_plane(work, step) / 255.0, 0.0, 1.0)
        if original_dtype == np.uint8:
            return (result * 255.0).round().astype(np.uint8)
        return result.astype(original_dtype)

    planes = [_requant_plane(work[..., c], step) for c in range(3)]
    result = np.clip(np.stack(planes, axis=-1) / 255.0, 0.0, 1.0)
    if frames.dtype == np.uint8:
        return (result * 255.0).round().astype(np.uint8)
    return result.astype(frames.dtype)


def chroma_quant(frames: np.ndarray, levels: int) -> np.ndarray:
    """色度量化：把 Cb/Cr 压缩到有限级数，破坏色度域水印（灰度帧原样返回）。"""
    if levels <= 0 or frames.ndim != 4:
        return frames
    original_dtype = frames.dtype
    rgb = frames.astype(np.uint8) if original_dtype == np.uint8 else (
        (np.clip(frames, 0, 1) * 255).round().astype(np.uint8)
    )

    def quantize(sub: np.ndarray) -> np.ndarray:
        # BT.601 定点整数正变换（Cb/Cr 中心化到 ±128）。
        r = sub[..., 0].astype(np.int32)
        g = sub[..., 1].astype(np.int32)
        b = sub[..., 2].astype(np.int32)
        y = (77 * r + 150 * g + 29 * b + 128) >> 8
        cb = ((-43 * r - 85 * g + 128 * b + 128) >> 8).astype(np.int16) - 128
        cr = ((128 * r - 107 * g - 21 * b + 128) >> 8).astype(np.int16) - 128
        scale = (levels - 1) / 255.0
        cb = (np.round(cb * scale) / scale).astype(np.int16)
        cr = (np.round(cr * scale) / scale).astype(np.int16)
        cb32 = cb.astype(np.int32)
        cr32 = cr.astype(np.int32)
        out = np.empty_like(sub)
        # Cb/Cr 已在 ±128 中心化，逆变换需把 0.5 偏移折算回整数常量。
        out[..., 0] = np.clip((y + ((359 * cr32) >> 8) + 179), 0, 255).astype(np.uint8)
        out[..., 1] = np.clip(
            (y - ((88 * cb32) >> 8) - ((183 * cr32) >> 8) - 135), 0, 255
        ).astype(np.uint8)
        out[..., 2] = np.clip((y + ((454 * cb32) >> 8) + 227), 0, 255).astype(np.uint8)
        return out

    out = quantize(rgb)
    if original_dtype == np.uint8:
        return out
    return out.astype(np.float32) / 255.0


def mirror(frames: np.ndarray) -> np.ndarray:
    """水平镜像。"""
    return np.flip(frames, axis=frames.ndim - 2)


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
    from cthulhu_backend.transform.parallel import map_frames

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

    from cthulhu_backend.transform.parallel import map_frames

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


def estimate_subtract(
    frames: np.ndarray,
    beta: float,
    size: int = 3,
) -> np.ndarray:
    """盲估计相减：把「原帧与去噪帧之差」当作水印估计，按 beta 倍过减。

    beta=1 等价于去噪；beta>1 对加性扩频水印形成过减攻击（经典 Stirmark
    类手段），在不去噪整片的前提下压低载荷相关。
    """
    if beta <= 0:
        return frames
    from cthulhu_backend.attacks.spatial import wiener_denoise

    original_dtype = frames.dtype
    work = frames.astype(np.float32)
    if original_dtype == np.uint8:
        work /= 255.0
    denoised = wiener_denoise(work, size=size)
    attacked = np.clip(work - beta * (work - denoised), 0.0, 1.0)
    if original_dtype == np.uint8:
        return (attacked * 255.0).round().astype(np.uint8)
    return attacked.astype(original_dtype)


def translate_jitter(frames: np.ndarray, jitter: float, rng: np.random.Generator) -> np.ndarray:
    """逐帧随机平移抖动：随机裁剪偏移后缩回原尺寸，模拟机位晃动。"""
    if jitter <= 0:
        return frames
    from PIL import Image

    from cthulhu_backend.transform.parallel import map_frames

    is_u8 = frames.dtype == np.uint8

    def shift_one(frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        dx = int(rng.uniform(-jitter, jitter) * w)
        dy = int(rng.uniform(-jitter, jitter) * h)
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

    return map_frames(shift_one, frames)


def drop_duplicate(frames: np.ndarray, every: int) -> np.ndarray:
    if every <= 1:
        return frames
    out = np.asarray(frames).copy()
    for index in range(every, len(out), every):
        out[index] = out[index - 1]
    return out


def apply(
    frames: np.ndarray,
    params: AssaultParams,
    rng: np.random.Generator,
) -> np.ndarray:
    """按固定顺序施加组合攻击，全部为可选开关。"""
    work = frames
    if params.mirror:
        work = mirror(work)
    if params.jitter > 0:
        work = translate_jitter(work, params.jitter, rng)
    if params.perspective > 0:
        work = perspective_shear(work, params.perspective, rng)
    if params.warp > 0:
        work = local_warp(work, params.warp, rng)
    if params.median > 0:
        work = median(work, params.median)
    if params.noise > 0:
        work = gaussian_noise(work, params.noise, rng)
    if params.subtract_beta > 0:
        work = estimate_subtract(work, params.subtract_beta)
    if params.requant > 0:
        work = pixel_requant(work, params.requant)
    if params.dct_step > 0:
        work = dct_requant(work, params.dct_step)
    if params.chroma_levels > 0:
        work = chroma_quant(work, params.chroma_levels)
    if params.drop_every > 0:
        work = drop_duplicate(work, params.drop_every)
    return work
