"""内容复杂度画像与嵌入域自动选型。

**边界说明**：报告 §2.8 的嵌入域画像使用「干净 vs 带水印」白盒差分；
单个带水印视频里静态水印与静态画面内容不可分离（median−mean 会抵消
静态分量），因此纯黑盒无法可靠识别 Y/Cb/Cr 嵌入域。

本模块做两件可行的事：
1. 黑盒估计内容复杂度（纹理/运动），在净化甜点区内建议强度与步数；
2. 嵌入域按已知方案证据映射（VideoSeal→luma、WAM→chroma），未知方案
   保守回退 both，避免错误定向。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.ndimage import gaussian_filter

from cthulhu_backend.transform import temporal

_EPS = 1e-6

# 已知公开方案的嵌入域（报告 §2.8 画像结论）。
_SCHEME_ATTACK = {
    "luma": "luma",
    "chroma": "chroma",
    "videoseal": "luma",
    "pixelseal": "luma",
    "trustmark": "luma",
    "mbrs": "luma",
    "wam": "chroma",
}

# 已知方案的清晰度档位上限（research §19.7，1080×1920 实测 BA@h264）：
#   亮度/低频类 512→0.87、256→0.56、192→0.49，必须压到 192；
#   色度类（WAM）512→0.55、256→0.47，256 就够，没必要牺牲画质。
# 0 表示"没有方案信息、不限制"，此时完全按用户选的档位执行。
_SCHEME_MAX_EDGE = {
    "luma": 192,
    "chroma": 256,
    "videoseal": 192,
    "pixelseal": 192,
    "trustmark": 192,
    "mbrs": 192,
    "wam": 256,
}


@dataclass
class Profile:
    """内容复杂度与选型建议。"""

    complexity: float
    motion: float
    edge_density: float
    temporal_coherence: float = 0.0
    suggested_attack: str = "both"
    suggested_purify_strength: float = 0.15
    suggested_purify_temporal: float = 0.0
    suggested_purify_max_edge: int = 0
    note: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def _as_float(frames: np.ndarray) -> np.ndarray:
    work = frames.astype(np.float32)
    if frames.dtype == np.uint8:
        work /= 255.0
    return work


def _luma(frames: np.ndarray) -> np.ndarray:
    if frames.ndim == 4:
        return 0.299 * frames[..., 0] + 0.587 * frames[..., 1] + 0.114 * frames[..., 2]
    return frames


def profile_frames(frames: np.ndarray, scheme: str = "") -> Profile:
    """估计内容复杂度并给出净化/嵌入域建议；scheme 为空时嵌入域用 both。"""
    if len(frames) == 0:
        raise ValueError("画像需要至少一帧")
    work = _as_float(frames)
    luma = _luma(work)
    low = gaussian_filter(luma, sigma=(0.0, 2.0, 2.0))
    high = luma - low
    low_energy = float(np.mean(low**2))
    high_energy = float(np.mean(high**2))
    edge_density = float(np.mean(np.abs(high)))
    if len(luma) >= 2:
        motion = float(np.mean(np.abs(np.diff(luma, axis=0))))
    else:
        motion = 0.0

    # 跨帧一致性代理：只用于判断是否值得尝试时序减法，不把它当成水印确认。
    # 降到 384 长边/16 帧以内，避免自动画像在 1080p 长片上额外吃内存。
    coherence = 0.0
    try:
        estimate = temporal.estimate(
            work,
            mode="luma",
            max_frames=16,
            max_edge=384,
        )
        coherence = estimate.coherence
    except Exception:  # noqa: BLE001 - 画像失败不影响主流程
        coherence = 0.0

    # 高频/低频能量比 + 帧间运动：值越大越复杂。
    texture_ratio = high_energy / (low_energy + _EPS)
    complexity = float(
        np.clip(0.7 * np.sqrt(texture_ratio) + 0.3 * (motion / 0.05), 0.0, 1.0)
    )
    # 保守甜点区：低复杂内容保守取 0.15，高复杂内容多给一点开关强度。
    strength = round(0.15 + 0.10 * complexity, 3)
    # 时序减法只在几乎静态、跨帧高度一致时给建议；默认档不轻易开启，避免鬼影。
    temporal_strength = 0.0
    if coherence >= 0.75 and motion <= 0.01:
        temporal_strength = round(min(0.25, (coherence - 0.75) / 0.25 * 0.25), 3)
    attack = _SCHEME_ATTACK.get(scheme, "both")
    scheme_edge = _SCHEME_MAX_EDGE.get(scheme, 0)
    if scheme in _SCHEME_ATTACK:
        note = f"已知方案 {scheme}，按报告画像使用 {attack}"
        if scheme_edge:
            note += f"，清晰度档位限制在长边 {scheme_edge}"
    else:
        note = "未知方案：黑盒无法可靠识别嵌入域，保守使用 both"
    return Profile(
        complexity=round(complexity, 4),
        motion=round(motion, 6),
        edge_density=round(edge_density, 6),
        temporal_coherence=round(float(coherence), 4),
        suggested_attack=attack,
        suggested_purify_strength=strength,
        suggested_purify_temporal=temporal_strength,
        suggested_purify_max_edge=scheme_edge,
        note=note,
    )


def _sampled_global_ids(sampled: np.ndarray, sampled_starts: list[int]) -> list[int]:
    """把 decode_sampled 的窗口拼接帧映射回全局帧号。"""
    if len(sampled) == 0 or not sampled_starts:
        return []
    per_window = max(1, len(sampled) // len(sampled_starts))
    ids: list[int] = []
    cursor = 0
    for index, start in enumerate(sampled_starts):
        remaining = len(sampled) - cursor
        if remaining <= 0:
            break
        length = per_window if index < len(sampled_starts) - 1 else remaining
        length = min(length, remaining)
        ids.extend(range(start, start + length))
        cursor += length
    return ids


def profile_shots(
    frames: np.ndarray,
    sampled_starts: list[int],
    shot_ranges: list[tuple[int, int]],
    *,
    scheme: str = "",
    fallback: Profile | None = None,
) -> list[Profile]:
    """按镜头切分抽样帧并逐镜头画像；短镜头/无样本镜头回退到全片画像。"""
    if len(frames) == 0 or not shot_ranges:
        return []
    global_ids = _sampled_global_ids(frames, sampled_starts)
    if not global_ids:
        return []
    global_ids = global_ids[: len(frames)]
    fallback_profile = fallback or profile_frames(frames, scheme)
    profiles: list[Profile] = []
    for start, end in shot_ranges:
        selected = [
            frame
            for frame, global_id in zip(frames, global_ids, strict=False)
            if start <= global_id < end
        ]
        if not selected:
            profiles.append(fallback_profile)
            continue
        profiles.append(profile_frames(np.stack(selected), scheme))
    return profiles
