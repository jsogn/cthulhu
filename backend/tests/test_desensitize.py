"""内容脱敏模块测试。"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from cthulhu_backend import samples
from cthulhu_backend.evaluate import metrics
from cthulhu_backend.media import ffmpeg
from cthulhu_backend.sample_prep import diff as diff_module
from cthulhu_backend.similarity import embedding
from cthulhu_backend.transform import audio as audio_transform
from cthulhu_backend.transform import shots
from cthulhu_backend.transform import video as video_transform

needs_ffmpeg = pytest.mark.skipif(not ffmpeg.has_ffmpeg(), reason="需要 ffmpeg")


def test_detect_cuts_finds_boundaries():
    video = samples.make_cut_video(segments=4, frames_per_segment=8, seed=6)
    assert shots.detect_cuts(video) == [0, 8, 16, 24, 32]


def test_reorder_keeps_content_but_changes_motion():
    video = samples.make_cut_video(segments=4, frames_per_segment=8, seed=7)
    reordered = video_transform.reorder_shots(video, shots.detect_cuts(video), np.random.default_rng(0))
    before = embedding.similarity_report(video, video)
    after = embedding.similarity_report(video, reordered)
    assert before["content_cosine"] > 0.999
    assert after["content_cosine"] > 0.9
    assert after["motion_cosine"] < before["motion_cosine"] - 0.1


def test_retime_and_recrop_shapes():
    video = samples.make_video_frames(20, 160, 120, seed=8)
    fast = video_transform.retime(video, 1.25)
    assert len(fast) == 16
    cropped = video_transform.recrop(video, 0.04)
    assert cropped.shape == video.shape


def test_overlay_banner_changes_pixels():
    video = samples.make_video_frames(3, 160, 120, seed=9)
    out = video_transform.overlay_banner(video, "DEMO")
    assert out.shape == video.shape
    assert float(np.mean(np.abs(out - video))) > 0.001


def test_similarity_bounds_and_regrade():
    video = samples.make_video_frames(4, 128, 128, seed=10)
    identical = embedding.similarity_report(video, video)
    assert identical["content_cosine"] > 0.999 and identical["dhash_agreement"] == 1.0
    recomposed = video_transform.recrop(video, 0.08)
    changed = embedding.similarity_report(video, recomposed)
    assert changed["content_cosine"] < 0.995
    regraded = video_transform.regrade(video, np.random.default_rng(0))
    regrade_report = embedding.similarity_report(video, regraded)
    assert regrade_report["content_cosine"] > 0.9  # 重调光不改变内容身份


def test_audio_remix_preserves_shape_and_peak():
    signal = samples.make_audio(1.0, seed=11)
    out = audio_transform.remix(signal, 16000, np.random.default_rng(0))
    assert out.shape == signal.shape
    assert abs(float(np.max(np.abs(out))) - 0.25) < 0.05


def test_audio_remix_keeps_content_timeline_aligned():
    """等长重混不应使内容提前或滞后：互相关峰值应停留在零延迟附近。"""
    from scipy.signal import correlate

    signal = samples.make_audio(2.0, seed=53)
    sample_rate = 16000
    rng = np.random.default_rng(1)
    outputs = (
        audio_transform.remix(signal, sample_rate, rng),
        audio_transform.remix_strong(signal, sample_rate, rng),
    )
    for out in outputs:
        corr = correlate(signal, out, mode="full", method="fft")
        lag_samples = len(signal) - 1 - int(np.argmax(corr))
        assert abs(lag_samples / sample_rate) < 0.01


def test_sharpen_increases_gradient_energy():
    frame = samples.make_video_frames(1, 128, 128, seed=41)[0]
    blurred = gaussian_filter(frame, sigma=1.5)

    def energy(img: np.ndarray) -> float:
        return float(np.mean(np.abs(np.diff(img, axis=0))) + np.mean(np.abs(np.diff(img, axis=1))))

    sharpened = video_transform.sharpen(np.asarray([blurred]))[0]
    assert energy(sharpened) > energy(blurred)


def test_color_restore_restores_mean_without_darkening():
    frames = samples.make_video_frames(4, 128, 128, seed=42)
    darkened = frames * 0.5 + 0.1
    restored = video_transform.color_restore(darkened, float(frames.mean()), float(frames.std()))
    assert abs(float(restored.mean()) - float(frames.mean())) < 0.01
    # 均值校正不应引入对比度扩张造成的暗部裁剪。
    assert float(restored.min()) >= 0.0 and float(restored.max()) <= 1.0


def test_parallel_transforms_match_sequential():
    """线程并行只改变执行方式，逐帧结果应与顺序循环一致。"""
    from scipy.signal import wiener

    from cthulhu_backend.attacks import spatial

    frames = samples.make_video_frames(12, 64, 64, seed=1)
    expected = np.stack([wiener(frame, (5, 5)) for frame in frames])
    assert np.allclose(expected, spatial.wiener_denoise(frames))
    assert video_transform.recrop(frames, 0.08).shape == frames.shape
    assert video_transform.sharpen(frames).shape == frames.shape


def test_midband_perturb_changes_midband_keeps_quality():
    frame = samples.make_video_frames(1, 128, 128, seed=43)[0]
    perturbed = video_transform.midband_perturb(np.asarray([frame]), strength=0.4, seed=0)[0]
    assert metrics.psnr(frame, perturbed) > 30
    baseline = diff_module._dct_midband_heatmap(frame, frame)
    changed = diff_module._dct_midband_heatmap(perturbed, frame)
    assert float(changed.mean()) > float(baseline.mean())


@needs_ffmpeg
def test_desensitize_flow_end_to_end(tmp_path):
    video = samples.make_cut_video(4, 8, 160, 120, seed=12)
    src = tmp_path / "src.mp4"
    dst = tmp_path / "out.mp4"
    ffmpeg.encode_video(video, str(src), fps=30)
    frames, info = ffmpeg.decode_video(str(src))
    rng = np.random.default_rng(1)
    out_frames = video_transform.reorder_shots(frames, shots.detect_cuts(frames), rng)
    out_frames = video_transform.regrade(out_frames, rng)
    ffmpeg.encode_video(out_frames, str(dst), fps=info["fps"])
    assert dst.exists()
    after = embedding.similarity_report(frames, ffmpeg.decode_video(str(dst))[0])
    assert after["motion_cosine"] < 0.95
