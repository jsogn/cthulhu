"""合成样本生成：研究闭环的干净基准媒体（视频帧 / 音频）。"""

from __future__ import annotations

import numpy as np


def make_video_frames(
    n_frames: int = 16,
    width: int = 320,
    height: int = 240,
    seed: int = 0,
    motion_speed: float = 1.0,
) -> np.ndarray:
    """生成带运动结构的干净灰度视频帧，形状 (F, H, W)，取值 [0, 1]。"""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width]
    frames = []
    for i in range(n_frames):
        # 随时间移动的渐变背景 + 圆与条纹，保证块内存在纹理（DCT 中频能量）。
        t = i / max(1, n_frames - 1)
        bg = 0.25 + 0.5 * (yy / height)
        cx = width * (0.2 + 0.6 * t)
        cy = height * 0.45
        circle = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 28**2)))
        stripe = 0.08 * np.sin(2 * np.pi * (xx + 3 * motion_speed * i) / 32)
        frame = np.clip(bg + 0.35 * circle + stripe + 0.02 * rng.standard_normal((height, width)), 0, 1)
        frames.append(frame)
    return np.asarray(frames, dtype=np.float64)


def make_audio(
    seconds: float = 2.0,
    sample_rate: int = 16000,
    seed: int = 0,
) -> np.ndarray:
    """生成和弦叠加的干净音频，取值 [-1, 1]。"""
    rng = np.random.default_rng(seed)
    n = int(seconds * sample_rate)
    t = np.arange(n) / sample_rate
    # 和声 + 包络 + 轻微噪声，覆盖多个频段（回声检测需要宽带信号）。
    sig = 0.3 * np.sin(2 * np.pi * 220 * t) + 0.25 * np.sin(2 * np.pi * 330 * t) + 0.15 * np.sin(2 * np.pi * 523 * t)
    env = np.minimum(t / 0.02, 1.0) * np.minimum((seconds - t) / 0.02, 1.0)
    sig = sig * env + 0.01 * rng.standard_normal(n)
    return np.clip(sig, -1, 1)


def make_cut_video(
    segments: int = 4,
    frames_per_segment: int = 8,
    width: int = 320,
    height: int = 240,
    seed: int = 0,
) -> np.ndarray:
    """生成带硬切镜头的合成视频：段间亮度与运动结构明显不同，便于分镜检测测试。"""
    parts = []
    for i in range(segments):
        segment = make_video_frames(frames_per_segment, width, height, seed + i, motion_speed=0.5 + i)
        segment = np.clip(segment + 0.35 * (i % 3) - 0.35, 0, 1)
        parts.append(segment)
    return np.concatenate(parts, axis=0)
