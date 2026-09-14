"""分析遍的纯规划计算：内存预算、镜头边界、分段与调光曲线。

这些函数原先以私有名散在 `pipeline` 里，但它们的契约（确定性、边界夹取、
累计取整）本身值得单独回归，遂提升为公开模块；`pipeline` 只负责解码/编码编排。
"""

from __future__ import annotations

import os
from itertools import pairwise

import numpy as np

from cthulhu_backend.schemas import DesensitizeOptions
from cthulhu_backend.transform import shots

# 低内存机器兜底：256MiB。
MAX_WORKING_BYTES = 256 * 1024**2

# 工作预算占物理内存的比例，按机器档位分：预算给太满会把机器推进 swap，
# 反而拖出「事件循环几十秒答不上探针」的长卡顿（2026-09-14 现场反馈）。
_BUDGET_FRACTION_HIGH_RAM = 0.55  # ≥24GiB：内存宽裕，允许更长的分块
_BUDGET_FRACTION_LOW_RAM = 0.45  # 4~24GiB：系统、界面与其他应用也要吃内存
_BUDGET_FRACTION_TINY_RAM = 0.4  # <4GiB：连系统本身都紧张


def memory_budget_bytes() -> int:
    """按物理内存自适应工作预算（256MB~32GB）。

    大内存按 55% 取用；16GB 这一档只给 45%——用户机器上还有系统、浏览器和
    界面的常驻占用，预算给满就会进 swap；<4GB 的小机器按 40% 取用。
    """
    try:
        total = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, ValueError, OSError):
        total = 0
    if total <= 0:
        return MAX_WORKING_BYTES
    if total >= 24 * 1024**3:
        fraction = _BUDGET_FRACTION_HIGH_RAM
    elif total >= 4 * 1024**3:
        fraction = _BUDGET_FRACTION_LOW_RAM
    else:
        fraction = _BUDGET_FRACTION_TINY_RAM
    return max(MAX_WORKING_BYTES, min(32 * 1024**3, int(total * fraction)))


def plan_shots(
    sampled: np.ndarray,
    sampled_starts: list[int],
    total_in: int,
    need_shots: bool,
) -> list[tuple[int, int]]:
    """抽样窗口内检测切点并映射回全局帧号，返回相邻切点区间。"""
    if need_shots and len(sampled) > 2 and sampled_starts:
        # 抽样按窗口返回；逐窗口检测切点再映射回全局帧号，避免窗口拼接处的假切点。
        per_window = max(1, len(sampled) // len(sampled_starts))
        boundaries = []
        for window, start in enumerate(sampled_starts):
            segment = sampled[window * per_window : (window + 1) * per_window]
            if len(segment) < 2:
                continue
            for cut in shots.detect_cuts(segment):
                if 0 < cut < len(segment):
                    boundaries.append(start + cut)
        boundaries = sorted({boundary for boundary in boundaries if 0 < boundary < total_in})
        boundaries = [0] + boundaries + [total_in]
    else:
        boundaries = [0, total_in]
    return list(pairwise(boundaries))


def build_segments(
    shot_ranges: list[tuple[int, int]],
    opts: DesensitizeOptions,
    speed: float,
    rng: np.random.Generator,
) -> tuple[list[tuple[int, int, int]], list[float], list[int], list[int], int]:
    """逐镜头切点漂移/变速与重排，输出编码区间（消耗 rng 流，顺序敏感）。"""
    shot_offsets: list[tuple[int, int]] = []
    shot_factors: list[float] = []
    for start, end in shot_ranges:
        length = end - start
        drop_start = int(rng.integers(0, opts.cut_jitter + 1)) if opts.cut_jitter > 0 else 0
        drop_end = int(rng.integers(0, opts.cut_jitter + 1)) if opts.cut_jitter > 0 else 0
        if drop_start + drop_end >= length:
            drop_start = min(drop_start, max(0, length - 1))
            drop_end = 0
        shot_offsets.append((drop_start, drop_end))
        shot_factors.append(
            float(rng.uniform(opts.shot_retime_min, opts.shot_retime_max))
            if opts.shot_retime
            else 1.0
        )
    order = rng.permutation(len(shot_ranges)) if opts.reorder else np.arange(len(shot_ranges))
    segments: list[tuple[int, int, int]] = []
    seg_factors: list[float] = []
    seg_out_lens: list[int] = []
    seg_shot_indices: list[int] = []
    cursor = 0
    # 累计取整（Bresenham）：逐镜头独立 round 会把每段的舍入误差留成常驻偏差，
    # 等长镜头下最多每段 ±0.5 帧，50 段就能累积成半秒音画不同步。改为「先累计
    # 精确输出长度、再取整到帧」，任意切点处的偏差都被夹在一帧以内。
    exact_out = 0.0
    for shot_index in order:
        orig_start, orig_end = shot_ranges[int(shot_index)]
        drop_start, drop_end = shot_offsets[int(shot_index)]
        eff_start = orig_start + drop_start
        eff_len = (orig_end - orig_start) - drop_start - drop_end
        factor = speed * shot_factors[int(shot_index)]
        retimed = speed != 1.0 or opts.shot_retime
        if retimed:
            exact_out += eff_len / factor
            out_len = max(1, round(exact_out) - cursor)
        else:
            out_len = eff_len
            exact_out = cursor + eff_len
        segments.append((cursor, eff_start, eff_len))
        seg_factors.append(factor)
        seg_out_lens.append(out_len)
        seg_shot_indices.append(int(shot_index))
        cursor += out_len
    return segments, seg_factors, seg_out_lens, seg_shot_indices, cursor


def regrade_curves(
    rng: np.random.Generator,
    total_out: int,
    perturb: float,
    regrade: bool,
) -> tuple[
    np.ndarray,
    np.ndarray,
    float,
    float,
    tuple[float, float, float, float],
    tuple[float, float, float, float],
]:
    """调光曲线：低频平滑的 gamma/亮度轨迹（确定性，消耗 rng 流）。"""
    gammas = np.ones(total_out, dtype=np.float32)
    deltas = np.zeros(total_out, dtype=np.float32)
    gamma_strength = 0.03 + 0.2 * perturb
    brightness = 0.02 + 0.06 * perturb
    periods = (0.0, 0.0, 0.0, 0.0)
    phases = (0.0, 0.0, 0.0, 0.0)
    if regrade:
        # 时间平滑：调光参数沿低频轨迹变化，避免逐帧独立随机造成的暗部闪烁。
        # 幅度与旧实现一致（gamma ±gamma_strength、亮度 ±brightness），对抗
        # 语义不变，只是相邻帧连续过渡。
        t = np.arange(total_out, dtype=np.float32)
        periods = tuple(float(rng.uniform(80.0, 180.0)) for _ in range(4))
        phases = tuple(float(rng.uniform(0.0, 2.0 * np.pi)) for _ in range(4))
        period_a, period_b, period_c, period_d = periods
        phase_a, phase_b, phase_c, phase_d = phases
        gammas = 1.0 + gamma_strength * (
            0.6 * np.sin(2.0 * np.pi * t / period_a + phase_a)
            + 0.4 * np.sin(2.0 * np.pi * t / period_b + phase_b)
        ).astype(np.float32)
        deltas = brightness * (
            0.6 * np.sin(2.0 * np.pi * t / period_c + phase_c)
            + 0.4 * np.sin(2.0 * np.pi * t / period_d + phase_d)
        ).astype(np.float32)
    return gammas, deltas, gamma_strength, brightness, periods, phases
