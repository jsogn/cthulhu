"""自建感知相似度量化：内容 embedding（顺序无关）、运动 embedding（顺序敏感）、dHash、SSIM。"""

from __future__ import annotations

import numpy as np


def resize_blockmean(arr: np.ndarray, size: int) -> np.ndarray:
    """块均值降采样到 size×size，等价于低通缩略图。"""
    h, w = arr.shape
    bh, bw = h // size, w // size
    cropped = arr[: bh * size, : bw * size]
    return cropped.reshape(size, bh, size, bw).mean(axis=(1, 3))


def frame_embedding(frame: np.ndarray, size: int = 16) -> np.ndarray:
    small = resize_blockmean(np.asarray(frame, dtype=np.float64), size)
    vector = small.ravel().astype(np.float64)
    vector = vector - vector.mean()
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else vector


def video_content_embedding(frames: np.ndarray, size: int = 16) -> np.ndarray:
    pooled = np.mean([frame_embedding(f, size) for f in frames], axis=0)
    norm = float(np.linalg.norm(pooled))
    return pooled / norm if norm > 0 else pooled


def video_motion_embedding(frames: np.ndarray, size: int = 16, bins: int = 8) -> np.ndarray:
    """相邻帧差分按时间分箱拼接：对镜头顺序与节奏敏感（顺序感知）。"""
    if len(frames) < 2:
        return np.zeros(size * size * bins)
    diffs = np.abs(np.asarray(frames)[1:] - np.asarray(frames)[:-1])
    pooled = []
    for chunk in np.array_split(diffs, bins):
        if len(chunk):
            vector = np.mean([frame_embedding(f, size) for f in chunk], axis=0)
        else:
            vector = np.zeros(size * size)
        pooled.append(vector)
    vector = np.concatenate(pooled)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else vector


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b)) / denom if denom > 0 else 0.0


def dhash(frame: np.ndarray, size: int = 8) -> np.ndarray:
    small = resize_blockmean(np.asarray(frame, dtype=np.float64), size + 1)
    return small[:, 1:] > small[:, :-1]


def dhash_agreement(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(a == b))


def similarity_report(
    reference_frames: np.ndarray,
    candidate_frames: np.ndarray,
    max_frames: int = 60,
) -> dict:
    """输出内容/运动/哈希/画质四维相似度与下降量。"""
    # 函数级延迟导入：破 evaluate ⇄ similarity ⇄ fingerprint 的静态循环依赖，
    # 与 transform/extra_attacks.py 的既有破环方式一致。
    from cthulhu_backend.evaluate import metrics

    ref = np.asarray(reference_frames)
    cand = np.asarray(candidate_frames)
    # 内容 embedding 是帧均值池化，抽样不影响口径；运动 embedding 抽样后仍保留节奏信息。
    if len(ref) > max_frames:
        indices = np.linspace(0, len(ref) - 1, max_frames).astype(int)
        ref = ref[indices]
    if len(cand) > max_frames:
        indices = np.linspace(0, len(cand) - 1, max_frames).astype(int)
        cand = cand[indices]
    content_cos = cosine(video_content_embedding(ref), video_content_embedding(cand))
    motion_cos = cosine(video_motion_embedding(ref), video_motion_embedding(cand))

    n = min(len(ref), len(cand), 8)
    sample_idx = np.linspace(0, min(len(ref), len(cand)) - 1, max(n, 1)).astype(int)
    dhash_agree = float(np.mean([dhash_agreement(dhash(ref[i]), dhash(cand[i])) for i in sample_idx]))
    ssim_mean = float(np.mean([metrics.ssim(ref[i], cand[i]) for i in sample_idx]))

    return {
        "content_cosine": round(content_cos, 4),
        "motion_cosine": round(motion_cos, 4),
        "dhash_agreement": round(dhash_agree, 4),
        "ssim_mean": round(ssim_mean, 4),
        "reduction": {
            "content": round(1.0 - content_cos, 4),
            "motion": round(1.0 - motion_cos, 4),
            "dhash": round(1.0 - dhash_agree, 4),
        },
    }
