"""盲检测置信度：无先验 payload 的启发式检测器。

每个检测器输出 0~1 的置信度分数，分数越高越可疑。这些统计量是研究口径的
启发式结果，不替代平台实测；最终判定应结合差分基准与多维度证据。
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.fftpack import dctn
from scipy.ndimage import gaussian_filter, median_filter

from cthulhu_backend.watermark import dwt as dwt_module
from cthulhu_backend.watermark.qim import MID_BAND

# 合成样本标定阈值（docs/baselines/detect-calibration.json，2026-08-27）。
# 仅作「疑似命中」提示：真实判定仍需干净同源差分与平台实测。
CALIBRATED_THRESHOLDS = {
    "ss": 0.58,
    "qim": 0.60,
    "temporal": 0.53,
    "dwt": 0.67,
    "chroma": 0.58,
    "dctmod": 0.44,
    "svd": 0.24,
    "echo": 0.60,
}

# 置信度下限：窗口间一致性不足时，即使分数越过阈值也不算命中。
MIN_HIT_CONFIDENCE = 0.75

# 统计类检测器的计算域：长边超过该值先降采样，统计量对分辨率不敏感；
# 结构类检测器（qim/dctmod/svd）必须保持原始分辨率，不经过此降采样。
STATS_LONG_EDGE = 640

# 结构类检测器的帧数上限：块结构对帧数不敏感，抽样足够稳定。
STRUCT_FRAMES = 90
STRUCT_WINDOW_FRAMES = 15


def _sigmoid(value: float, center: float, scale: float) -> float:
    return float(1.0 / (1.0 + np.exp(-(value - center) / max(scale, 1e-9))))


def _stats_frames(frames: np.ndarray) -> np.ndarray:
    """统计检测器计算域：float32 + 长边 ≤ STATS_LONG_EDGE 的等比例降采样。"""
    frames = np.asarray(frames, dtype=np.float32)
    if frames.ndim == 2:
        frames = frames[None, ...]
    long_edge = max(frames.shape[-2], frames.shape[-1])
    scale = STATS_LONG_EDGE / long_edge
    if scale >= 1.0:
        return frames
    from scipy.ndimage import zoom

    if frames.ndim == 3:
        return zoom(frames, (1.0, scale, scale), order=1).astype(np.float32)
    return zoom(frames, (1.0, scale, scale, 1.0), order=1).astype(np.float32)


def ss_blind(frames: np.ndarray) -> float:
    """空域扩频协同检测：同图案水印跨帧叠加，多帧残差均值能量异常。

    单帧输入退化为中性分数，建议至少 4 帧以获得稳定统计。
    """
    frames = _stats_frames(frames)
    if len(frames) < 2:
        return 0.5
    # 两遍增量：不再为整段堆残差栈，内存与帧数无关。
    # mean_res = E[frame - median(frame)]；content_energy = E[(残差 - mean_res)^2]。
    sum_res = np.zeros(frames.shape[1:], dtype=np.float64)
    for frame in frames:
        sum_res += frame - median_filter(frame, size=3)
    mean_res = sum_res / len(frames)
    mean_energy = float(np.mean(mean_res**2))
    sum_sq = 0.0
    for frame in frames:
        residual = frame - median_filter(frame, size=3)
        sum_sq += float(np.mean((residual - mean_res) ** 2))
    content_energy = sum_sq / len(frames)
    ratio = mean_energy / max(content_energy, 1e-12)
    # 校准依据：合成样本干净基线 ratio≈0.07，水印 ratio≈2.0；
    # H.264 编码后干净≈0.09、水印≈1.4，仍保持一个数量级以上的差距。
    return _sigmoid(np.log10(1.0 + ratio), 0.15, 0.12)


def qim_blind(frames: np.ndarray) -> float:
    """DCT-QIM 量化格检测：中频系数到最近格点的距离异常集中。"""
    frames = np.asarray(frames, dtype=np.float64)
    if frames.ndim == 2:
        frames = frames[None, ...]
    indices = np.linspace(0, len(frames) - 1, min(STRUCT_FRAMES, len(frames))).astype(int)
    frames = frames[indices]
    best = 0.0
    # 每帧只做一次 8×8 DCT，三个 delta 共用同一份中频系数（约 3 倍提速）。
    mid_coeffs: list[np.ndarray] = []
    for frame in frames:
        frame8 = frame * 255.0
        pad_h, pad_w = -frame8.shape[0] % 8, -frame8.shape[1] % 8
        padded = np.pad(frame8, ((0, pad_h), (0, pad_w)), mode="edge")
        blocks = padded.reshape(padded.shape[0] // 8, 8, padded.shape[1] // 8, 8)
        coeffs = dctn(blocks, axes=(1, 3), norm="ortho")
        bh, bw = coeffs.shape[0], coeffs.shape[2]
        mid_coeffs.append(coeffs.transpose(0, 2, 1, 3).reshape(bh, bw, 64)[:, :, MID_BAND])
    for delta in (4.0, 6.0, 8.0):
        total = 0
        near = 0
        for mid in mid_coeffs:
            quantized = np.round(mid / delta)
            dist = np.abs(mid - quantized * delta)
            total += dist.size
            near += int(np.count_nonzero(dist < delta * 0.08))
        if total == 0:
            continue
        near_lattice = near / total
        best = max(best, _sigmoid(near_lattice, 0.35, 0.15))
    return best


def temporal_blind(frames: np.ndarray) -> float:
    """时域盲检测：帧差能量图的噪声化程度。

    平滑内容的帧差能量集中在运动区域，活动图空间上连续平滑；
    时域加性图案使活动图变为空间噪声（χ² 分布，方差≈2×均值²）。
    启发式取「活动图空间粗糙度」=std/mean；真实颗粒噪点同样会抬高分数，
    需与 ss_blind 与置信度联合判读，不能单独下结论。
    """
    frames = _stats_frames(frames)
    if len(frames) < 3:
        return 0.5
    # 帧差能量图：自然内容的能量集中在运动边缘（空间方差大），
    # 差异域加性图案抬高全画面均匀底噪（均值升、方差几乎不变）。
    activity = np.mean(np.diff(frames, axis=0) ** 2, axis=0)
    roughness = float(activity.std() / (activity.mean() + 1e-9))
    return _sigmoid(roughness, 0.5, 0.4)


def dwt_blind(frames: np.ndarray) -> float:
    """小波域盲检测：对角线细节子带（HH）能量相对水平/垂直子带异常。

    DWT 加性嵌入（本仓库 dwt 方案即嵌入 HH）会抬高 HH 能量；
    自然图像的能量跨子带平滑衰减，比值落在稳定区间。
    """
    frames = _stats_frames(frames)
    ratios = []
    for frame in frames:
        coeffs, _ = dwt_module._decompose(frame)
        half_h, half_w = coeffs.shape[0] // 2, coeffs.shape[1] // 2
        hh = coeffs[half_h:, half_w:]
        hl = coeffs[:half_h, half_w:]
        lh = coeffs[half_h:, :half_w]
        hh_energy = float(np.mean(hh**2))
        cross_energy = float(np.mean(hl**2) + np.mean(lh**2))
        ratios.append(hh_energy / max(cross_energy, 1e-12))
    ratio = float(np.mean(ratios))
    return _sigmoid(np.log10(1.0 + ratio), 0.2, 0.35)


def dctmod_blind(frames: np.ndarray, max_frames: int = 8) -> float:
    """DCT 模运算水印盲检测：低/中频系数对齐到非零模数格点（FireKeeper 族）。

    近零系数天然落在任何格点上（0 是任意模数的倍数），必须排除，
    否则平滑内容会整体误报。
    """
    frames = np.asarray(frames, dtype=np.float64)
    if frames.ndim == 2:
        frames = frames[None, ...]
    indices = np.linspace(0, len(frames) - 1, min(max_frames, len(frames))).astype(int)
    frames = frames[indices]
    best = 0.0
    for mod in (12.0, 20.0, 36.0):
        # 逐系数位置独立统计：载体可能落在 DC 或任意一个低/中频系数上，
        # 混池会被边缘块的大量未嵌入 AC 系数稀释，逐位取最大值更稳。
        totals = np.zeros(22, dtype=np.int64)
        nears = np.zeros(22, dtype=np.int64)
        for frame in frames:
            frame8 = frame * 255.0
            pad_h, pad_w = -frame8.shape[0] % 8, -frame8.shape[1] % 8
            padded = np.pad(frame8, ((0, pad_h), (0, pad_w)), mode="edge")
            blocks = padded.reshape(padded.shape[0] // 8, 8, padded.shape[1] // 8, 8)
            coeffs = dctn(blocks, axes=(1, 3), norm="ortho")
            bh, bw = coeffs.shape[0], coeffs.shape[2]
            flat = coeffs.transpose(0, 2, 1, 3).reshape(bh, bw, 64)
            low_mid = flat[:, :, :22]  # 直流 + 低/中频
            dist = np.abs(low_mid - np.round(low_mid / mod) * mod)
            valid = np.abs(low_mid) > mod * 0.25
            totals += np.count_nonzero(valid, axis=(0, 1))
            nears += np.count_nonzero(valid & (dist < mod * 0.04), axis=(0, 1))
        for total, near in zip(totals, nears, strict=True):
            if total > 0:
                best = max(best, _sigmoid(near / total, 0.45, 0.2))
    return best


def svd_blind(frames: np.ndarray, max_frames: int = 8) -> float:
    """SVD 模运算水印盲检测：4×4 块前导奇异值的格点对齐统计。

    DWT-DCT-SVD 族（guofei blind_watermark）在块奇异值上做量化嵌入，
    使前导奇异值向格点聚集；自然图像奇异值分布连续平滑。
    """
    frames = np.asarray(frames, dtype=np.float64)
    if frames.ndim == 2:
        frames = frames[None, ...]
    indices = np.linspace(0, len(frames) - 1, min(max_frames, len(frames))).astype(int)
    values = []
    for i in indices:
        frame = frames[i] * 255.0
        h, w = frame.shape[0] - frame.shape[0] % 4, frame.shape[1] - frame.shape[1] % 4
        blocks = frame[:h, :w].reshape(h // 4, 4, w // 4, 4).transpose(0, 2, 1, 3)
        blocks = blocks.reshape(-1, 4, 4)
        sv = np.linalg.svd(blocks, compute_uv=False)
        values.append(sv[:, 0])
    leading = np.concatenate(values)
    best = 0.0
    # 半格偏移是常见变体（bit=1 打在 grid/2），故额外探测 10 与 18。
    for mod in (8.0, 10.0, 18.0, 20.0, 36.0):
        dist = np.abs(leading - np.round(leading / mod) * mod)
        near = float(np.mean(dist < mod * 0.04))
        best = max(best, _sigmoid(near, 0.5, 0.25))
    return best


def chroma_blind(rgb_frames: np.ndarray) -> float:
    """色度盲检测：色差通道高频残差能量相对亮度的异常抬升。

    自然压缩视频的色度经 4:2:0 抽样后残差远低于亮度；
    色度域水印（如 Cb/Cr 加性图案）会显著抬高该比值。
    """
    rgb = _stats_frames(rgb_frames)
    if rgb.ndim == 3:
        rgb = rgb[None, ...]
    luma = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    cb = -0.168736 * rgb[..., 0] - 0.331264 * rgb[..., 1] + 0.5 * rgb[..., 2]
    cr = 0.5 * rgb[..., 0] - 0.418688 * rgb[..., 1] - 0.081312 * rgb[..., 2]

    def residual_energy(plane: np.ndarray) -> float:
        return float(np.mean([np.std(frame - gaussian_filter(frame, 1.5)) for frame in plane]))

    luma_n = residual_energy(luma)
    chroma_n = residual_energy(cb) + residual_energy(cr)
    return _sigmoid(chroma_n / max(luma_n, 1e-9), 0.25, 0.15)


def lsb_blind(frames: np.ndarray) -> float:
    """LSB 检测（χ² PoV）：同高七位的像素对中，LSB 分布偏离随机。

    局限（研究结论，需如实引用）：
    - 低负载「直接写位」的 LSB 在数学上不可盲检，分数保持中性；
    - 有损压缩视频的量化会破坏 LSB 随机性，产生强烈误报，
      本检测器仅适用于未压缩像素域样本。
    """
    frames = np.asarray(frames, dtype=np.float64)
    if frames.ndim == 2:
        frames = frames[None, ...]
    indices = np.linspace(0, len(frames) - 1, min(STRUCT_FRAMES, len(frames))).astype(int)
    frames = frames[indices]
    chi_values: list[float] = []
    for frame in frames:
        image = (frame * 255.0).round().astype(np.uint8)
        pairs = (image // 2).ravel()
        total = np.bincount(pairs, minlength=128).astype(np.float64)
        zeros = np.bincount(
            pairs,
            weights=((image & 1) == 0).ravel(),
            minlength=128,
        ).astype(np.float64)
        expected = total / 2.0
        active = expected > 4.0
        if np.count_nonzero(active) < 8:
            chi_values.append(1.0)
            continue
        deviation = (zeros[active] - expected[active]) ** 2 / expected[active]
        chi_values.append(float(np.mean(deviation)))
    return _sigmoid(float(np.mean(chi_values)), 1.6, 1.1)


def echo_blind(signal: np.ndarray, sample_rate: int) -> float:
    """音频回声隐藏检测：典型回声延迟窗口的倒谱峰值异常。"""
    signal = np.asarray(signal, dtype=np.float64)
    segment = signal[: int(sample_rate * 0.5)]
    if len(segment) < 512:
        return 0.5
    spec = np.abs(np.fft.rfft(segment * np.hanning(len(segment)))) ** 2
    cepstrum = np.fft.irfft(np.log(spec + 1e-12), n=len(segment)).real
    window = np.abs(cepstrum[40:200])
    peak = float(np.max(window))
    baseline = float(np.median(window))
    return _sigmoid(np.log10(peak / max(baseline, 1e-9)), 0.6, 0.35)


def video_scores(frames: np.ndarray) -> dict[str, float]:
    """视频各方案盲检测置信度。"""
    stats = _stats_frames(frames)
    return {
        "ss": round(ss_blind(stats), 4),
        "qim": round(qim_blind(frames), 4),
        "lsb": round(lsb_blind(frames), 4),
        "temporal": round(temporal_blind(stats), 4),
        "dwt": round(dwt_blind(stats), 4),
    }


def structural_scores(frames: np.ndarray) -> dict[str, float]:
    """结构性盲检测（SVD / DCT 模运算）：内部降帧抽样，控制耗时。"""
    return {
        "dctmod": round(dctmod_blind(frames, max_frames=8), 4),
        "svd": round(svd_blind(frames), 4),
    }


def windowed_video_scores(frames: np.ndarray, window: int = 50) -> dict[str, float]:
    """跨全片抽样的逐窗口盲检测，取窗口分数均值，避免跨场景统计混叠。"""
    scores, _ = windowed_video_scores_with_confidence(frames, window)
    return scores


def windowed_video_scores_with_confidence(
    frames: np.ndarray, window: int = 50
) -> tuple[dict[str, float], dict[str, float]]:
    """逐窗口盲检测 + 窗口间一致性置信度（越高越像真实水印而非内容纹理）。"""
    frames = np.asarray(frames)
    if frames.ndim == 2:
        frames = frames[None, ...]
    if len(frames) <= window:
        scores = video_scores(frames)
        return scores, {key: 1.0 for key in scores}

    def window_scores(part: np.ndarray) -> dict[str, float]:
        stats = _stats_frames(part)
        struct_idx = np.linspace(0, len(part) - 1, min(STRUCT_WINDOW_FRAMES, len(part))).astype(int)
        struct = np.asarray(part, dtype=np.float64)[struct_idx]
        return {
            "ss": round(ss_blind(stats), 4),
            "qim": round(qim_blind(struct), 4),
            "lsb": round(lsb_blind(struct), 4),
            "temporal": round(temporal_blind(stats), 4),
            "dwt": round(dwt_blind(stats), 4),
        }

    chunks = []
    for start in range(0, len(frames), window):
        part = frames[start : start + window]
        if len(part) >= 2:
            chunks.append(part)
    # 窗口相互独立：scipy 滤波释放 GIL，多线程近线性提速（内存有界）。
    workers = min(len(chunks), os.cpu_count() or 2)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        parts = list(pool.map(window_scores, chunks))
    if not parts:
        return video_scores(frames), {key: 1.0 for key in video_scores(frames)}
    scores = {
        key: round(float(np.mean([part[key] for part in parts])), 4) for key in parts[0]
    }
    confidence = {
        key: round(
            float(max(0.0, 1.0 - np.std([part[key] for part in parts]) / (scores[key] + 1e-9))),
            4,
        )
        for key in parts[0]
    }
    return scores, confidence


def audio_scores(signal: np.ndarray, sample_rate: int) -> dict[str, float]:
    """音频各方案盲检测置信度。"""
    return {"echo": round(echo_blind(signal, sample_rate), 4)}


def hits(
    scores: dict[str, float],
    structural: dict[str, float] | None = None,
    confidence: dict[str, float] | None = None,
) -> list[str]:
    """按标定阈值 + 窗口置信度汇总疑似命中项。"""
    merged = {**scores, **(structural or {})}
    confidence = confidence or {}
    found = []
    for key, threshold in CALIBRATED_THRESHOLDS.items():
        value = merged.get(key)
        if value is None:
            continue
        conf = confidence.get(key, 1.0)
        if value >= threshold and conf >= MIN_HIT_CONFIDENCE:
            found.append(key)
    return found
