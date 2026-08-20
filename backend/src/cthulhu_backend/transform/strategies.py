"""画面脱敏变换策略：同一接口下的可切换实现。

`thorough` 原样保留当前完整变换序列（默认）；后续新增 `fast` 等策略时
只注册新实现，不修改本文件中的现状序列。切换由 `transform_strategy`
配置控制，任务结果中记录实际使用的策略名，便于 A/B 归因与一键回退。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

import numpy as np

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
    ref_mean: float
    ref_std: float
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
            frames = np.stack([qim.embed(frame, ctx.spoof_bits, delta=6) for frame in frames])
        return frames


STRATEGIES: dict[str, TransformStrategy] = {"thorough": ThoroughStrategy()}
DEFAULT_STRATEGY = "thorough"


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
