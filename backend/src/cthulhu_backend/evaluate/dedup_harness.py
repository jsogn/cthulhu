"""本地判重代理基准。

用一个可复现的「代理判重栈」代替真实平台，量化两个视频之间的重复度：

- 图像感知哈希族：pHash / dHash / wavelet / block-mean（逐帧聚合汉明距离）；
- 音频：对数梅尔谱指纹（60 秒采样）；
- 结构：采样窗口内的镜头切点向量；
- 综合：加权距离与重复风险分（研究口径，非平台实测）。

用途：清洗前后对比「去重距离」、评估每个对抗原语对判重栈的破坏力，
以及后续多版本候选排序。所有分数只表示相对差异，阈值需平台实测标定。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.fft import dctn, rfft

from cthulhu_backend.media import ffmpeg

HASH_BITS = 64
_WINDOWS = [(8.0, 14), (0.25, 14), (0.5, 14), (0.75, 14), (None, 14)]  # 五个采样窗口

# 无关联内容的经验距离锚点（来自不同素材对实测），用于把绝对距离归一到 0~1。
# 这是第一版标定，后续用平台真值样本修正。
_CHANCE_DIST = {
    "phash": 0.50,
    "dhash": 0.35,
    "whash": 0.38,
    "blockhash": 0.34,
    "audio": 0.54,
    "structure": 0.09,
    "deep": 0.6,
}
_WEIGHTS_DEEP = {"image": 0.35, "temporal": 0.15, "deep": 0.25, "audio": 0.15, "structure": 0.1}
_WEIGHTS_FALLBACK = {"image": 0.5, "temporal": 0.25, "audio": 0.15, "structure": 0.1}


def _zoom(frame: np.ndarray, size: int) -> np.ndarray:
    """忽略宽高比缩放到 size×size（感知哈希标准做法，位长恒定）。"""
    factor_y = size / frame.shape[0]
    factor_x = size / frame.shape[1]
    g = ndimage.zoom(frame, (factor_y, factor_x), order=1)
    out = np.zeros((size, size), dtype=g.dtype)
    h = min(size, g.shape[0])
    w = min(size, g.shape[1])
    out[:h, :w] = g[:h, :w]
    return out


def _to_bits(image: np.ndarray, size: int, threshold: str) -> np.ndarray:
    """缩放并按全局均值/中位数二值化，返回 01 位数组。"""
    g = _zoom(image, size)
    value = float(np.mean(g)) if threshold == "mean" else float(np.median(g))
    return (g > value).reshape(-1)


def phash(frame: np.ndarray) -> np.ndarray:
    g = _zoom(frame, 32)
    c = dctn(g, norm="ortho")[:8, :8]
    c[0, 0] = 0.0
    return (c > float(np.mean(c))).reshape(-1)


def dhash(frame: np.ndarray) -> np.ndarray:
    # 8×9 网格比较相邻列 → 8×8=64 位。
    g = _zoom(frame, 8)
    g = np.concatenate([g, g[:, :1]], axis=1)  # 补一列得到 9 列
    return (g[:, :-1] > g[:, 1:]).reshape(-1)


def whash(frame: np.ndarray) -> np.ndarray:
    g = _zoom(frame, 32)
    for _ in range(2):  # 两级 Haar 平均 → 8×8
        g = (g[0::2, 0::2] + g[1::2, 0::2] + g[0::2, 1::2] + g[1::2, 1::2]) / 4.0
    return (g > float(np.median(g))).reshape(-1)


def blockhash(frame: np.ndarray) -> np.ndarray:
    return _to_bits(frame, 8, "mean")


_FAMILIES = {"phash": phash, "dhash": dhash, "whash": whash, "blockhash": blockhash}


def _hamming(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(a != b))


def _sample_frames(path: str) -> tuple[np.ndarray, dict]:
    """采样五个窗口的灰度帧（共 70 帧），返回 (frames, info)。"""
    info = ffmpeg.video_info(path)
    parts = []
    for start_sec, count in _WINDOWS:
        if start_sec is None:
            start = max(0.0, info["duration"] - count / info["fps"] - 2.0)
        else:
            start = min(start_sec if start_sec < 1 else info["duration"] * start_sec, info["duration"] - count / info["fps"])
        frames, _ = ffmpeg.decode_video_range(path, int(start * info["fps"]), count)
        parts.append(frames)
    return np.concatenate(parts, axis=0), info


def _color_frames(path: str, count: int = 16) -> np.ndarray:
    """深度嵌入用的彩色抽样帧（四个位置 × count/4 帧）。"""
    info = ffmpeg.video_info(path)
    parts = []
    per = max(1, count // 4)
    for frac in (0.08, 0.35, 0.6, 0.85):
        start = max(0, int(info["duration"] * frac * info["fps"]))
        frames, _ = ffmpeg.decode_video_range(path, start, per, grayscale=False)
        parts.append(frames)
    return np.concatenate(parts, axis=0)


def deep_distance(original: str, candidate: str) -> float | None:
    """CLIP ViT-B/32 视觉嵌入的余弦距离（1 - 余弦相似度）；模型缺失时返回 None。"""
    try:
        from cthulhu_backend.fingerprint import deep

        if not deep.available():
            return None
        frames_a = _color_frames(original)
        frames_b = _color_frames(candidate)
        return 1.0 - deep.cosine(
            deep.video_embedding(frames_a, max_frames=16),
            deep.video_embedding(frames_b, max_frames=16),
        )
    except Exception:  # noqa: BLE001 - 嵌入维度失败时自动降级
        return None


def frame_hashes(path: str) -> dict:
    frames, _ = _sample_frames(path)
    out: dict = {}
    for name, fn in _FAMILIES.items():
        bits = np.stack([fn(f) for f in frames])
        out[name] = np.packbits(bits, axis=1).tolist()
    out["frames"] = len(frames)
    return out


def _frame_dists(frames_a: np.ndarray, frames_b: np.ndarray, search: int = 30) -> dict:
    """逐帧哈希距离；candidate 允许 ±search 帧错位（抵抗变速造成的时移）。"""
    n = min(len(frames_a), len(frames_b))
    result: dict = {}
    for name, fn in _FAMILIES.items():
        bits_a = np.stack([fn(f) for f in frames_a[:n]])
        bits_b = np.stack([fn(f) for f in frames_b])
        distances = np.empty(n)
        for i in range(n):
            lo = max(0, i - search)
            hi = min(len(bits_b), i + search + 1)
            distances[i] = min(_hamming(bits_a[i], bits_b[j]) for j in range(lo, hi))
        result[name] = float(np.mean(distances))
    return result


def _shot_vector(frames: np.ndarray) -> np.ndarray:
    diff = np.mean(np.abs(np.diff(frames, axis=0)), axis=(1, 2))
    if len(diff) == 0 or float(np.median(diff)) == 0:
        return np.zeros(len(frames), dtype=bool)
    return diff > 3.0 * float(np.median(diff))


def _structure_distance(frames_a: np.ndarray, frames_b: np.ndarray) -> float:
    n = min(len(frames_a), len(frames_b))
    va = _shot_vector(frames_a[:n])
    vb = _shot_vector(frames_b[:n])
    if not np.any(va) and not np.any(vb):
        return 0.0
    return float(np.mean(va != vb))


def _temporal_sequence_distance(frames_a: np.ndarray, frames_b: np.ndarray) -> float:
    """时序指纹：逐帧哈希序列按索引直接比较（不做 ±search 对齐）。

    变速/逐镜头变速会平移帧序列，即使单帧内容几乎一致，序列指纹也会产生
    距离。这是内容对齐距离无法捕捉、平台视频级指纹却高度敏感的一维。
    """
    n = min(len(frames_a), len(frames_b))
    distances = []
    for fn in _FAMILIES.values():
        bits_a = np.stack([fn(f) for f in frames_a[:n]])
        bits_b = np.stack([fn(f) for f in frames_b[:n]])
        distances.append(float(np.mean([_hamming(bits_a[i], bits_b[i]) for i in range(n)])))
    return float(np.mean(distances))


def _mel_filterbank(n_mels: int, n_fft: int, rate: int) -> np.ndarray:
    lo, hi = 300.0, min(8000.0, rate / 2)
    mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)  # noqa: E731
    inverse = lambda m: 700.0 * (10.0 ** (m / 2595.0) - 1.0)  # noqa: E731
    points = np.linspace(mel(lo), mel(hi), n_mels + 2)
    hz = inverse(points)
    bins = np.floor((n_fft + 1) * hz / rate).astype(int)
    filters = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(n_mels):
        for f in range(bins[i], bins[i + 1]):
            filters[i, f] = (f - bins[i]) / max(bins[i + 1] - bins[i], 1)
        for f in range(bins[i + 1], min(bins[i + 2] + 1, n_fft // 2 + 1)):
            filters[i, f] = (bins[i + 2] - f) / max(bins[i + 2] - bins[i + 1], 1)
    return filters


def _audio_mel_bits(path: str) -> np.ndarray | None:
    decoded = ffmpeg.decode_audio(path, max_seconds=60)
    if decoded is None:
        return None
    signal, rate = decoded
    n_fft = 4096
    hop = n_fft // 2
    frames_count = (len(signal) - n_fft) // hop
    if frames_count < 2:
        return None
    windows = np.lib.stride_tricks.sliding_window_view(signal[: frames_count * hop + n_fft], n_fft)[::hop]
    spectrum = np.abs(rfft(windows, n=n_fft, axis=1)) ** 2
    mel = spectrum @ _mel_filterbank(40, n_fft, rate).T
    logmel = np.log10(np.maximum(mel, 1e-10))
    # 32 个时间片 × 40 个梅尔带，按带内中位数二值化。
    idx = np.linspace(0, logmel.shape[0] - 1, 32).astype(int)
    grid = logmel[idx]
    bits = grid > np.median(grid, axis=0, keepdims=True)
    return bits.reshape(-1)


def audio_distance(a: str, b: str) -> float | None:
    bits_a = _audio_mel_bits(a)
    bits_b = _audio_mel_bits(b)
    if bits_a is None or bits_b is None or len(bits_a) != len(bits_b):
        return None
    return _hamming(bits_a, bits_b)


def compare(original: str, candidate: str) -> dict:
    """计算两个视频的代理判重距离与风险分。"""
    frames_a, info_a = _sample_frames(original)
    frames_b, _ = _sample_frames(candidate)
    image = _frame_dists(frames_a, frames_b)
    temporal = _temporal_sequence_distance(frames_a, frames_b)
    structure = _structure_distance(frames_a, frames_b)
    audio = audio_distance(original, candidate)
    deep_dist = deep_distance(original, candidate)
    weights = dict(_WEIGHTS_DEEP if deep_dist is not None else _WEIGHTS_FALLBACK)

    def norm(value: float, chance: float) -> float:
        return max(0.0, min(1.0, (chance - value) / chance))

    image_norm = float(
        np.mean([norm(image[k], _CHANCE_DIST[k]) for k in ("phash", "dhash", "whash", "blockhash")])
    )
    temporal_norm = norm(temporal, 0.36)  # 与图像均值同一量级锚点
    audio_norm = norm(audio, _CHANCE_DIST["audio"]) if audio is not None else image_norm
    structure_norm = norm(structure, _CHANCE_DIST["structure"])
    deep_norm = norm(deep_dist, _CHANCE_DIST["deep"]) if deep_dist is not None else 0.0
    risk = (
        weights["image"] * image_norm
        + weights["temporal"] * temporal_norm
        + weights.get("deep", 0.0) * deep_norm
        + weights["audio"] * audio_norm
        + weights["structure"] * structure_norm
    )
    level = "高" if risk >= 0.75 else "中" if risk >= 0.45 else "低"
    return {
        "original": original,
        "candidate": candidate,
        "original_info": {"duration": info_a["duration"], "fps": info_a["fps"]},
        "distances": {
            **{f"image_{k}": round(v, 4) for k, v in image.items()},
            "structure": round(structure, 4),
            "audio": None if audio is None else round(audio, 4),
            "temporal": round(temporal, 4),
            "deep": None if deep_dist is None else round(deep_dist, 4),
        },
        "duplicate_risk": round(risk, 4),
        "risk_level": level,
        "normalized": {
            "image": round(image_norm, 4),
            "temporal": round(temporal_norm, 4),
            "audio": round(audio_norm, 4),
            "structure": round(structure_norm, 4),
            "deep": None if deep_dist is None else round(deep_norm, 4),
        },
        "weights": weights,
        "note": "距离按无关联内容锚点归一；含 CLIP 深度嵌入维度，研究口径阈值需平台实测标定。",
    }


def save_report(report: dict, out_dir: str, tag: str) -> str:
    out = Path(out_dir) / f"dedup-{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(out)
