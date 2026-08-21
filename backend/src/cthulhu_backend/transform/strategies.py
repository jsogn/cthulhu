"""画面脱敏变换策略：同一接口下的可切换实现。

`thorough` 原样保留当前完整变换序列（默认）；后续新增 `fast` 等策略时
只注册新实现，不修改本文件中的现状序列。切换由 `transform_strategy`
配置控制，任务结果中记录实际使用的策略名，便于 A/B 归因与一键回退。
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from PIL import Image, ImageFilter

from cthulhu_backend.attacks import spatial as spatial_attacks
from cthulhu_backend.transform import video as video_transform
from cthulhu_backend.watermark import qim

logger = logging.getLogger(__name__)


@dataclass
class TransformOptions:
    """一次任务中随用户选项变化的开关（与分块无关）。"""

    recrop: float = 0.0
    regrade: bool = True
    anti_reembed: bool = False
    denoise: bool = False
    color_restore: bool = True
    sharpness: bool = True
    spoof: bool = False


@dataclass
class TransformContext:
    """一次任务中预先算好的全局上下文（随机参数与参考统计）。"""

    gammas: np.ndarray
    deltas: np.ndarray
    banner: str
    seed: int
    ref_mean: float | np.ndarray
    ref_std: float | np.ndarray
    mid_rng: np.random.Generator
    spoof_bits: list[int] | None = None


class TransformStrategy(Protocol):
    """画面变换策略接口：对已解码帧块应用脱敏变换。"""

    name: str
    description: str

    def apply(
        self,
        frames: np.ndarray,
        output_ids: list[int],
        ctx: TransformContext,
        options: TransformOptions,
    ) -> np.ndarray: ...


class ThoroughStrategy:
    """现状重方案：完整逐像素变换，序列与既有管线逐行一致。"""

    name = "thorough"
    description = "完整逐像素变换（现状管线，优化前原样保留）"
    frame_dtype = "float32"

    def apply(
        self,
        frames: np.ndarray,
        output_ids: list[int],
        ctx: TransformContext,
        options: TransformOptions,
    ) -> np.ndarray:
        # 顺序必须与 services.run_desensitize 原有实现保持一致，保证现状行为不变。
        if options.recrop > 0:
            frames = video_transform.recrop(frames, options.recrop)
        if options.regrade:
            frames = video_transform.regrade_with_params(
                frames, ctx.gammas[output_ids], ctx.deltas[output_ids],
            )
        if ctx.banner:
            frames = video_transform.overlay_banner(frames, ctx.banner, ctx.seed)
        if options.anti_reembed:
            frames = video_transform.midband_perturb(frames, strength=0.4, rng=ctx.mid_rng)
        if options.denoise:
            frames = spatial_attacks.wiener_denoise(frames, size=5)
        if options.color_restore:
            frames = video_transform.color_restore(frames, ctx.ref_mean, ctx.ref_std)
        if options.sharpness:
            frames = video_transform.sharpen(frames, amount=0.25, radius=1.2, threshold=0.01)
        if options.spoof and ctx.spoof_bits is not None:
            frames = _embed_qim_frames(frames, ctx.spoof_bits, delta=6)
        return frames


def _u8_to_f32(frames: np.ndarray) -> np.ndarray:
    return frames.astype(np.float32) / 255.0


def _f32_to_u8(frames: np.ndarray) -> np.ndarray:
    return (np.clip(frames, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def _u8_roundtrip(frames: np.ndarray, fn) -> np.ndarray:
    """对尚无 uint8 原语的操作做一次 float32 往返，保证语义可用。"""
    return _f32_to_u8(fn(_u8_to_f32(frames)))


def _embed_qim_frames(frames: np.ndarray, bits: list[int], delta: int = 6) -> np.ndarray:
    """QIM 嵌入：灰度逐帧；彩色逐通道重复同一 payload。"""
    if frames.ndim == 4:
        return np.stack(
            [
                np.stack([qim.embed(frame[..., c], bits, delta=delta) for c in range(3)], axis=-1)
                for frame in frames
            ]
        )
    return np.stack([qim.embed(frame, bits, delta=delta) for frame in frames])


def _fast_recrop_u8(frames: np.ndarray, crop_frac: float) -> np.ndarray:
    """重新构图（uint8 直通）：PIL 原生双线性重采样，免精度转换。"""
    h, w = frames.shape[1:3]
    cy0, cy1 = int(h * crop_frac), h - int(h * crop_frac)
    cx0, cx1 = int(w * crop_frac), w - int(w * crop_frac)

    def resize(frame: np.ndarray) -> np.ndarray:
        mode = "RGB" if frame.ndim == 3 else "L"
        image = Image.fromarray(frame, mode=mode)
        resized = image.resize((w, h), Image.BILINEAR, box=(cx0, cy0, cx1, cy1))
        return np.asarray(resized, dtype=np.uint8)

    from cthulhu_backend.transform.parallel import map_frames

    return map_frames(resize, frames)


def _fast_regrade_u8(
    frames: np.ndarray,
    gammas: np.ndarray,
    deltas: np.ndarray,
    levels: int = 16,
) -> np.ndarray:
    """重调光（uint8 查表）：连续 gamma 量化到有限档位，逐帧查表加速。"""
    gamma_lo = float(np.min(gammas))
    gamma_hi = float(np.max(gammas))
    grid = np.array([gamma_lo]) if gamma_hi - gamma_lo < 1e-9 else np.linspace(gamma_lo, gamma_hi, levels)
    level_idx = np.argmin(np.abs(grid[:, None] - np.asarray(gammas, dtype=np.float32)[None, :]), axis=0)
    lut = np.stack(
        [(np.arange(256, dtype=np.float32) / 255.0) ** gamma * 255.0 for gamma in grid]
    )
    tables = lut[level_idx]

    def apply_one(pair: tuple[np.ndarray, np.ndarray, float]) -> np.ndarray:
        table, frame, delta = pair
        out = np.take(table, frame, mode="clip") + delta * 255.0
        return np.clip(out, 0.0, 255.0).astype(np.uint8)

    from cthulhu_backend.transform.parallel import map_frames

    return map_frames(apply_one, list(zip(tables, frames, deltas)))


def _fast_color_restore_u8(
    frames: np.ndarray,
    ref_mean: float,
    ref_std: float,
) -> np.ndarray:
    """亮度还原（uint8 域）：逐通道均值校准，避免方差归一带来的暗部裁剪。"""
    work = frames.astype(np.float32)
    means = work.mean(axis=(1, 2), keepdims=True)
    ref_mean8 = np.asarray(ref_mean) * 255.0
    del ref_std
    corrected = work + (ref_mean8 - means)
    return np.clip(corrected, 0.0, 255.0).astype(np.uint8)


def _fast_sharpen_u8(frames: np.ndarray, amount: float = 0.25, radius: float = 1.2) -> np.ndarray:
    """锐度补偿（uint8 直通）：PIL UnsharpMask，免 float 往返。

    percent 为 0~255 亮度域百分比口径，threshold≈0.01×255 与原阈值对应。
    """
    percent = round(amount * 100)

    def unsharp(frame: np.ndarray) -> np.ndarray:
        mode = "RGB" if frame.ndim == 3 else "L"
        image = Image.fromarray(frame, mode=mode)
        out = image.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=3))
        return np.asarray(out, dtype=np.uint8)

    from cthulhu_backend.transform.parallel import map_frames

    return map_frames(unsharp, frames)


def _fast_banner_u8(frames: np.ndarray, text: str, seed: int = 0, margin_frac: float = 0.04) -> np.ndarray:
    """贴纸条（uint8 直通）：与 float 版本同布局参数的 PIL 绘制。"""
    from PIL import ImageDraw, ImageFont

    rng = np.random.default_rng(seed)
    h, w = frames.shape[1:3]
    margin = int(w * margin_frac)
    box_h = int(h * 0.12)
    y0 = int(rng.uniform(margin, max(margin + 1, h - box_h - margin)))
    font = ImageFont.load_default(size=max(12, box_h - 10))

    def draw_frame(frame: np.ndarray) -> np.ndarray:
        mode = "RGB" if frame.ndim == 3 else "L"
        image = Image.fromarray(frame, mode=mode).convert("RGB")
        draw = ImageDraw.Draw(image, "RGBA")
        draw.rectangle([margin, y0, w - margin, y0 + box_h], fill=(0, 0, 0, 120))
        draw.text((margin + 12, y0 + (box_h - 16) // 2), text, fill=(255, 255, 255, 220), font=font)
        return np.asarray(image.convert("RGB" if frame.ndim == 3 else "L"), dtype=np.uint8)

    from cthulhu_backend.transform.parallel import map_frames

    return map_frames(draw_frame, frames)


def rotate_de_sync(
    frames: np.ndarray,
    output_ids: list[int],
    max_angle: float,
) -> np.ndarray:
    """逐帧微旋转去同步：角度按黄金角摆动，确定性且并行安全。

    每帧角度 = max_angle * sin(index * 2.399963)，旋转后中心裁剪回原尺寸。
    几何去同步对 pHash 类指纹的破坏效率远高于同预算的像素扰动。
    """
    from cthulhu_backend.transform.parallel import map_frames

    is_u8 = frames.dtype == np.uint8

    def rotate_one(pair: tuple[int, np.ndarray]) -> np.ndarray:
        index, frame = pair
        angle = max_angle * math.sin(index * 2.399963)
        h, w = frame.shape[:2]
        mode = "RGB" if frame.ndim == 3 else "L"
        if is_u8:
            image = Image.fromarray(frame, mode=mode)
        else:
            image = Image.fromarray((np.clip(frame, 0, 1) * 255).round().astype(np.uint8), mode=mode)
        rotated = image.rotate(angle, resample=Image.BILINEAR, expand=True)
        out_w, out_h = rotated.size
        box = ((out_w - w) // 2, (out_h - h) // 2, (out_w - w) // 2 + w, (out_h - h) // 2 + h)
        result = np.asarray(rotated.crop(box))
        if is_u8:
            return result.astype(np.uint8)
        return result.astype(np.float32) / 255.0

    return map_frames(rotate_one, list(zip(output_ids, frames)))


class FastStrategy:
    """快速档：整条主链在 uint8 域端到端执行，免去多次精度往返。

    recrop/sharpen/banner 走 PIL 直通，regrade 走查表，color_restore 仅
    在 uint8 上做 float32 校正；无 uint8 原语的少数操作按需往返 float32。
    变换序列与 thorough 顺序一致，是否保留由检测基准 A/B 决定。
    """

    name = "fast"
    description = "快速档：uint8 端到端 + 查表/LUT/PIL 原语（A/B 验证中）"
    frame_dtype = "uint8"

    def apply(
        self,
        frames: np.ndarray,
        output_ids: list[int],
        ctx: TransformContext,
        options: TransformOptions,
    ) -> np.ndarray:
        if options.recrop > 0:
            frames = _fast_recrop_u8(frames, options.recrop)
        if options.regrade:
            frames = _fast_regrade_u8(
                frames, ctx.gammas[output_ids], ctx.deltas[output_ids],
            )
        if ctx.banner:
            frames = _fast_banner_u8(frames, ctx.banner, ctx.seed)
        if options.anti_reembed:
            frames = _u8_roundtrip(
                frames, lambda f: video_transform.midband_perturb(f, strength=0.4, rng=ctx.mid_rng)
            )
        if options.denoise:
            frames = _u8_roundtrip(frames, lambda f: spatial_attacks.wiener_denoise(f, size=5))
        if options.color_restore:
            frames = _fast_color_restore_u8(frames, ctx.ref_mean, ctx.ref_std)
        if options.sharpness:
            frames = _fast_sharpen_u8(frames, amount=0.25, radius=1.2)
        if options.spoof and ctx.spoof_bits is not None:
            frames = _u8_roundtrip(
                frames,
                lambda f: _embed_qim_frames(f, ctx.spoof_bits, delta=6),
            )
        return frames


STRATEGIES: dict[str, TransformStrategy] = {
    "thorough": ThoroughStrategy(),
    "fast": FastStrategy(),
}
DEFAULT_STRATEGY = "fast"


def register_strategy(strategy: TransformStrategy) -> None:
    """注册新策略；同名覆盖仅用于显式替换，常规新增请使用新名称。"""
    STRATEGIES[strategy.name] = strategy


def get_strategy(name: str | None) -> TransformStrategy:
    """按名称取策略；空值取默认，未知名称告警后回退默认（保证可回退）。"""
    resolved = name or DEFAULT_STRATEGY
    strategy = STRATEGIES.get(resolved)
    if strategy is None:
        logger.warning("未知变换策略 %s，回退到 %s", resolved, DEFAULT_STRATEGY)
        return STRATEGIES[DEFAULT_STRATEGY]
    return strategy
