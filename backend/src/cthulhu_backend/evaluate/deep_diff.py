"""同源视频细粒度差分：定位超出转码噪声的可观测嵌入痕迹。

对「原版 vs 平台下载版」做帧对齐后的多维度对比：
1. 分块 DCT 中频能量差与逐系数残差结构；
2. 逐帧亮度均值的时间频谱（周期性亮度扰动）；
3. 色度通道差分；
4. 音频功率谱差分（窄带峰值）。

所有输出均为研究口径观察值，不直接下“有水印/无水印”结论。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.fft import dctn, rfft
from scipy.signal import welch

from cthulhu_backend.media import ffmpeg


def _phase_shift(a: np.ndarray, b: np.ndarray) -> tuple[int, int, float]:
    """相位相关估计 b 相对 a 的整数平移 (dy, dx)，返回 (dy, dx, 峰值)。"""
    fa = np.fft.fft2(a)
    fb = np.fft.fft2(b)
    cross = np.fft.ifft2(fa * np.conj(fb)).real
    peak = np.unravel_index(np.argmax(cross), cross.shape)
    dy = peak[0] if peak[0] <= a.shape[0] // 2 else peak[0] - a.shape[0]
    dx = peak[1] if peak[1] <= a.shape[1] // 2 else peak[1] - a.shape[1]
    return int(dy), int(dx), float(np.max(cross))


def _roll(frames: np.ndarray, dy: int, dx: int) -> np.ndarray:
    return np.roll(np.roll(frames, dy, axis=1), dx, axis=2)


def _to_gray(rgb: np.ndarray) -> np.ndarray:
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def _dct_blocks(frames: np.ndarray) -> np.ndarray:
    """(F,H,W) -> (F,H//8,W//8,8,8) 的 8×8 正交 DCT 系数。"""
    f, h, w = frames.shape
    blocks = frames.reshape(f, h // 8, 8, w // 8, 8).transpose(0, 1, 3, 2, 4)
    return dctn(blocks, axes=(-2, -1), norm="ortho")


def _mid_energy(coeffs: np.ndarray) -> np.ndarray:
    """中频带能量 (F,H//8,W//8)。"""
    u = np.arange(8)
    v = np.arange(8)
    band = ((u[:, None] + v[None, :]) >= 4) & ((u[:, None] + v[None, :]) <= 11)
    band[0, 0] = False
    return np.sum(coeffs[..., band] ** 2, axis=-1)


def _top_peaks(signal: np.ndarray, k: int = 5, skip_low: int = 2) -> list[dict]:
    """一维频谱中除最低频外的前 k 个局部峰值（幅值、归一化频率、周期）。"""
    mag = np.abs(signal)
    order = np.argsort(mag)[::-1]
    picked: list[dict] = []
    for index in order:
        if index <= skip_low:
            continue
        if any(abs(index - item["index"]) <= 1 for item in picked):
            continue
        picked.append(
            {
                "index": int(index),
                "freq": float(index / len(signal)),
                "period_frames": float(len(signal) / max(index, 1)),
                "magnitude": float(mag[index]),
            }
        )
        if len(picked) >= k:
            break
    return picked


def _audio_delta(original: str, download: str) -> dict:
    a = ffmpeg.decode_audio(original, max_seconds=60)
    b = ffmpeg.decode_audio(download, max_seconds=60)
    if a is None or b is None:
        return {"error": "音轨缺失，无法差分"}
    sa, ra = a
    sb, rb = b
    length = min(len(sa), len(sb))
    rate = min(ra, rb)
    fa, pxx_a = welch(sa[:length], rate, nperseg=4096)
    fb, pxx_b = welch(sb[:length], rate, nperseg=4096)
    if not np.allclose(fa, fb):
        pxx_b = np.interp(fa, fb, pxx_b)
    log_delta = 10 * np.log10(np.maximum(pxx_b, 1e-12)) - 10 * np.log10(np.maximum(pxx_a, 1e-12))
    noise = np.std(log_delta)
    peaks = []
    for i in range(2, len(fa) - 2):
        window = log_delta[i - 1 : i + 2]
        if window[1] == max(window) and window[1] > 2.5 * noise:
            peaks.append(
                {
                    "freq_hz": round(float(fa[i]), 1),
                    "delta_db": round(float(log_delta[i]), 2),
                    "prominence_db": round(float(window[1] - np.median(window)), 2),
                }
            )
    peaks.sort(key=lambda item: item["delta_db"], reverse=True)
    return {
        "log_psd_delta_std_db": round(float(noise), 3),
        "narrowband_peaks": peaks[:8],
        "mean_abs_delta_db": round(float(np.mean(np.abs(log_delta))), 3),
    }


def run_deep_diff(original: str, download: str, tag: str, out_dir: str) -> dict:
    """执行一对视频的细粒度差分并落盘 JSON。"""
    info_o = ffmpeg.video_info(original)
    info_d = ffmpeg.video_info(download)
    fps = min(info_o["fps"], info_d["fps"])
    scale = None
    if (info_d["height"], info_d["width"]) != (info_o["height"], info_o["width"]):
        scale = float(info_o["height"]) / info_d["height"]

    observations: dict = {"tag": tag, "original": original, "download": download}

    # 两个 120 帧段落：开头与中段，保证可复现。
    segments = [
        (int(fps * 15), 120, "15s"),
        (int(fps * info_d["duration"] * 0.55), 120, "55%"),
    ]
    dct_deltas: list[dict] = []
    temporal: list[dict] = []
    for start, count, label in segments:
        frames_o, _ = ffmpeg.decode_video_range(original, start, count)
        frames_d, _ = ffmpeg.decode_video_range(download, start, count)
        if scale is not None:
            zoom = (1.0, scale, scale)
            frames_d = ndimage.zoom(frames_d, zoom, order=1)
        if frames_d.shape != frames_o.shape:
            frames_d = frames_d[: frames_o.shape[0]]
            frames_d = frames_d[:, : frames_o.shape[1], : frames_o.shape[2]]
        small_o = ndimage.zoom(frames_o[0], 0.125, order=1)
        small_d = ndimage.zoom(frames_d[0], 0.125, order=1)
        dy, dx, peak = _phase_shift(small_o, small_d)
        frames_d = _roll(frames_d, dy * 8, dx * 8)

        coeff_o = _dct_blocks(frames_o)
        coeff_d = _dct_blocks(frames_d)
        energy_o = _mid_energy(coeff_o)
        energy_d = _mid_energy(coeff_d)
        delta = energy_o - energy_d
        rel = delta / (np.mean(energy_o) + 1e-9)
        residual = coeff_d - coeff_o
        # 逐 DCT 系数的跨帧均值残差（有固定嵌入则呈结构化图案）。
        coeff_pattern = np.mean(residual, axis=(0, 1, 2))
        coeff_pattern = (coeff_pattern / (np.std(residual, axis=(0, 1, 2)) + 1e-9)).round(3)
        dct_deltas.append(
            {
                "segment": label,
                "shift": [dy * 8, dx * 8],
                "phase_peak": round(peak, 3),
                "mid_energy_mean_delta": round(float(np.mean(delta)), 6),
                "mid_energy_rel_mean": round(float(np.mean(rel)), 4),
                "mid_energy_rel_std": round(float(np.std(rel)), 4),
                "mid_energy_corr": round(float(np.corrcoef(energy_o.ravel(), energy_d.ravel())[0, 1]), 4),
                "coeff_pattern_z": [[round(float(v), 2) for v in row] for row in coeff_pattern],
                "coeff_pattern_abs_max": float(np.max(np.abs(coeff_pattern))),
                "coeff_pattern_abs_argmax": [
                    int(np.unravel_index(np.argmax(np.abs(coeff_pattern)), coeff_pattern.shape)[0]),
                    int(np.unravel_index(np.argmax(np.abs(coeff_pattern)), coeff_pattern.shape)[1]),
                ],
            }
        )

        # 逐帧全局亮度的时间频谱：找下载版独有的周期性。
        mean_o = np.mean(frames_o, axis=(1, 2))
        mean_d = np.mean(frames_d, axis=(1, 2))
        mean_o -= np.polyval(np.polyfit(np.arange(count), mean_o, 1), np.arange(count))
        mean_d -= np.polyval(np.polyfit(np.arange(count), mean_d, 1), np.arange(count))
        spec_o = rfft(mean_o)
        spec_d = rfft(mean_d)
        temporal.append(
            {
                "segment": label,
                "orig_peaks": _top_peaks(spec_o),
                "download_peaks": _top_peaks(spec_d),
                "luma_std_orig": round(float(np.std(mean_o)), 5),
                "luma_std_download": round(float(np.std(mean_d)), 5),
            }
        )

    observations["dct_differential"] = dct_deltas
    observations["temporal_luma"] = temporal

    # 色度差分（段首 24 帧）。
    rgb_o, _ = ffmpeg.decode_video_range(original, int(fps * 15), 24, grayscale=False)
    rgb_d, _ = ffmpeg.decode_video_range(download, int(fps * 15), 24, grayscale=False)
    if scale is not None:
        rgb_d = ndimage.zoom(rgb_d, (1.0, scale, scale, 1.0), order=1)
    rgb_d = rgb_d[: rgb_o.shape[0], : rgb_o.shape[1], : rgb_o.shape[2], :]
    dy, dx, _ = _phase_shift(_to_gray(rgb_o[0])[::8, ::8], _to_gray(rgb_d[0])[::8, ::8])
    rgb_d = np.roll(np.roll(rgb_d, dy * 8, axis=1), dx * 8, axis=2)
    yuv_o = rgb_to_yuv(rgb_o)
    yuv_d = rgb_to_yuv(rgb_d)
    chroma = {
        "u_mean_abs_delta": round(float(np.mean(np.abs(yuv_o[..., 1] - yuv_d[..., 1]))), 4),
        "v_mean_abs_delta": round(float(np.mean(np.abs(yuv_o[..., 2] - yuv_d[..., 2]))), 4),
        "u_corr": round(float(np.corrcoef(yuv_o[..., 1].ravel(), yuv_d[..., 1].ravel())[0, 1]), 4),
        "v_corr": round(float(np.corrcoef(yuv_o[..., 2].ravel(), yuv_d[..., 2].ravel())[0, 1]), 4),
    }
    observations["chroma"] = chroma
    observations["audio"] = _audio_delta(original, download)

    out = Path(out_dir) / f"deep-diff-{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(observations, ensure_ascii=False, indent=2), encoding="utf-8")
    return observations


def rgb_to_yuv(rgb: np.ndarray) -> np.ndarray:
    """RGB(0~1) -> YUV，Y 之外另算，U/V 用于色度差分。"""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = -0.14713 * r - 0.28886 * g + 0.436 * b
    v = 0.615 * r - 0.51499 * g - 0.10001 * b
    return np.stack([y, u, v], axis=-1)
