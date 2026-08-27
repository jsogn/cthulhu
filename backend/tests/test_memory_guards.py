"""内存护栏与抽样等价性回归测试。"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.fftpack import dctn
from scipy.ndimage import median_filter

from cthulhu_backend import samples
from cthulhu_backend.media import ffmpeg
from cthulhu_backend.watermark import detect
from cthulhu_backend.watermark.qim import MID_BAND


def _frames(n: int = 12, h: int = 48, w: int = 64, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((n, h, w), dtype=np.float32)


def test_decode_video_rejects_oversized_estimate(monkeypatch: pytest.MonkeyPatch) -> None:
    """超大时长在真正解码前就被拒绝，不会启动 ffmpeg 也不会分配数组。"""
    monkeypatch.setattr(
        ffmpeg,
        "video_info",
        lambda path: {
            "width": 1920,
            "height": 1080,
            "fps": 30.0,
            "pix_fmt": "yuv420p",
            "codec": "h264",
            "format": "mov,mp4",
            "duration": 3600.0,
        },
    )
    with pytest.raises(ValueError, match="拒绝执行"):
        ffmpeg.decode_video("fake.mp4")


def test_encode_video_rejects_oversized_payload() -> None:
    """零拷贝伪造的超大帧缓冲同样在编码前被拦截。"""
    base = np.zeros(8, dtype=np.float32)
    fake = np.lib.stride_tricks.as_strided(base, shape=(1 << 31, 4, 4), strides=(0, 0, 0))
    with pytest.raises(ValueError, match="拒绝执行"):
        ffmpeg.encode_video(fake, "/tmp/never-written.mp4")


def test_ss_blind_incremental_matches_explicit_stack() -> None:
    """增量两遍实现与原残差栈公式逐位一致。"""
    frames = _frames()
    score = detect.ss_blind(frames)

    # 检测器在 float32 计算域内增量统计；参考实现必须同域、同累加精度。
    frames32 = frames.astype(np.float32)
    sum_res = np.zeros(frames32.shape[1:], dtype=np.float64)
    for frame in frames32:
        sum_res += frame - median_filter(frame, size=3)
    mean_res = sum_res / len(frames32)
    mean_energy = float(np.mean(mean_res**2))
    sum_sq = 0.0
    for frame in frames32:
        residual = frame - median_filter(frame, size=3)
        sum_sq += float(np.mean((residual - mean_res) ** 2))
    content_energy = sum_sq / len(frames32)
    ratio = mean_energy / max(content_energy, 1e-12)
    expected = detect._sigmoid(np.log10(1.0 + ratio), 0.15, 0.12)
    assert score == pytest.approx(expected, abs=1e-9)


def test_qim_blind_incremental_matches_concat() -> None:
    """逐帧计数实现与原全量 concat 公式逐位一致。"""
    frames = _frames()
    score = detect.qim_blind(frames)

    best = 0.0
    for delta in (4.0, 6.0, 8.0):
        distances: list[np.ndarray] = []
        for frame in frames:
            frame8 = np.asarray(frame, dtype=np.float64) * 255.0
            pad_h, pad_w = -frame8.shape[0] % 8, -frame8.shape[1] % 8
            padded = np.pad(frame8, ((0, pad_h), (0, pad_w)), mode="edge")
            blocks = padded.reshape(padded.shape[0] // 8, 8, padded.shape[1] // 8, 8)
            coeffs = dctn(blocks, axes=(1, 3), norm="ortho")
            bh, bw = coeffs.shape[0], coeffs.shape[2]
            flat = coeffs.transpose(0, 2, 1, 3).reshape(bh, bw, 64)
            mid = flat[:, :, MID_BAND]
            quantized = np.round(mid / delta)
            distances.append(np.abs(mid - quantized * delta).ravel())
        pooled = np.concatenate(distances)
        near_lattice = float(np.mean(pooled < delta * 0.08))
        best = max(best, detect._sigmoid(near_lattice, 0.35, 0.15))
    assert score == pytest.approx(best, abs=1e-12)


def test_windowed_scores_fallback_and_range() -> None:
    """短输入等价于整段打分；长输入逐窗口平均且各分数落在 [0,1]。"""
    short = _frames(n=6)
    assert detect.windowed_video_scores(short) == detect.video_scores(short)

    long = _frames(n=120)
    scores = detect.windowed_video_scores(long, window=50)
    assert set(scores) == {"ss", "qim", "lsb", "temporal", "dwt"}
    assert all(0.0 <= value <= 1.0 for value in scores.values())


def test_sampled_detect_path_is_bounded_and_deterministic(tmp_path) -> None:
    """detect-watermark 的抽样路径：帧数有界、结果可复现。"""
    video = samples.make_video_frames(120, 64, 64, seed=3)
    path = str(tmp_path / "clip.mp4")
    ffmpeg.encode_video(video, path, fps=30)

    sampled, _ = ffmpeg.decode_sampled(path, cap=60)
    assert len(sampled) <= 60
    first = detect.windowed_video_scores(sampled)
    second = detect.windowed_video_scores(sampled)
    assert first == second
    assert all(0.0 <= value <= 1.0 for value in first.values())
