"""盲检测置信度：无先验 payload 的启发式检测器。

每个检测器输出 0~1 的置信度分数，分数越高越可疑。这些统计量是研究口径的
启发式结果，不替代平台实测；最终判定应结合差分基准与多维度证据。
"""

from __future__ import annotations

import numpy as np
from scipy.fftpack import dctn
from scipy.ndimage import median_filter

from cthulhu_backend.watermark.qim import MID_BAND


def _sigmoid(value: float, center: float, scale: float) -> float:
    return float(1.0 / (1.0 + np.exp(-(value - center) / max(scale, 1e-9))))


def ss_blind(frames: np.ndarray) -> float:
    """空域扩频协同检测：同图案水印跨帧叠加，多帧残差均值能量异常。

    单帧输入退化为中性分数，建议至少 4 帧以获得稳定统计。
    """
    frames = np.asarray(frames, dtype=np.float64)
    if frames.ndim == 2:
        frames = frames[None, ...]
    if len(frames) < 2:
        return 0.5
    residuals = np.stack([frame - median_filter(frame, size=3) for frame in frames])
    mean_res = residuals.mean(axis=0)
    mean_energy = float(np.mean(mean_res**2))
    content_energy = float(np.mean((residuals - mean_res) ** 2))
    ratio = mean_energy / max(content_energy, 1e-12)
    # 校准依据：合成样本干净基线 ratio≈0.07，水印 ratio≈2.0；
    # H.264 编码后干净≈0.09、水印≈1.4，仍保持一个数量级以上的差距。
    return _sigmoid(np.log10(1.0 + ratio), 0.15, 0.12)


def qim_blind(frames: np.ndarray) -> float:
    """DCT-QIM 量化格检测：中频系数到最近格点的距离异常集中。"""
    frames = np.asarray(frames, dtype=np.float64)
    if frames.ndim == 2:
        frames = frames[None, ...]
    best = 0.0
    for delta in (4.0, 6.0, 8.0):
        distances: list[np.ndarray] = []
        for frame in frames:
            frame8 = frame * 255.0
            pad_h, pad_w = -frame8.shape[0] % 8, -frame8.shape[1] % 8
            padded = np.pad(frame8, ((0, pad_h), (0, pad_w)), mode="edge")
            ph, pw = padded.shape
            blocks = padded.reshape(ph // 8, 8, pw // 8, 8)
            coeffs = dctn(blocks, axes=(1, 3), norm="ortho")
            bh, bw = coeffs.shape[0], coeffs.shape[2]
            flat = coeffs.transpose(0, 2, 1, 3).reshape(bh, bw, 64)
            mid = flat[:, :, MID_BAND]
            quantized = np.round(mid / delta)
            distances.append(np.abs(mid - quantized * delta).ravel())
        pooled = np.concatenate(distances)
        near_lattice = float(np.mean(pooled < delta * 0.08))
        best = max(best, _sigmoid(near_lattice, 0.35, 0.15))
    return best


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
    return {
        "ss": round(ss_blind(frames), 4),
        "qim": round(qim_blind(frames), 4),
        "lsb": round(lsb_blind(frames), 4),
    }


def audio_scores(signal: np.ndarray, sample_rate: int) -> dict[str, float]:
    """音频各方案盲检测置信度。"""
    return {"echo": round(echo_blind(signal, sample_rate), 4)}
