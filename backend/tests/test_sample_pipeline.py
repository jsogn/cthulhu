"""真实视频文件样本管线测试（部分用例依赖 ffmpeg）。"""

from __future__ import annotations

import numpy as np
import pytest

from cthulhu_backend import samples
from cthulhu_backend.evaluate import metrics
from cthulhu_backend.media import container, ffmpeg
from cthulhu_backend.sample_prep import diff as diff_module
from cthulhu_backend.sample_prep.align import align_pair, estimate_shift
from cthulhu_backend.watermark import common, qim

needs_ffmpeg = pytest.mark.skipif(not ffmpeg.has_ffmpeg(), reason="需要 ffmpeg/ffprobe")


def test_estimate_shift_recovers_translation():
    frame = samples.make_video_frames(1, 128, 128, seed=1)[0]
    shifted = np.roll(np.roll(frame, 3, axis=0), -5, axis=1)
    assert estimate_shift(frame, shifted) == (-3, 5)


def test_align_pair_crops_common_region():
    frame = samples.make_video_frames(1, 128, 128, seed=2)[0]
    moved = np.roll(np.roll(frame, 4, axis=0), 6, axis=1)
    ref, mov, shift = align_pair(frame, moved)
    assert shift == (-4, -6)
    assert ref.shape == mov.shape
    assert metrics.psnr(ref, mov) > 40


def test_aligned_psnr_recovers_after_translation():
    frame = samples.make_video_frames(1, 128, 128, seed=8)[0]
    moved = np.roll(np.roll(frame, 5, axis=0), 7, axis=1)
    assert metrics.aligned_psnr(frame, moved) > 40


def test_dct_heatmap_detects_qim_energy():
    frame = samples.make_video_frames(1, 128, 128, seed=3)[0]
    watermarked = qim.embed(frame, common.payload_bits(1, 32))
    heatmap = diff_module._dct_midband_heatmap(watermarked, frame)
    assert float(heatmap.mean()) > 1.0


@needs_ffmpeg
def test_encode_decode_roundtrip(tmp_path):
    frames = samples.make_video_frames(8, 160, 120, seed=4)
    path = tmp_path / "clean.mp4"
    ffmpeg.encode_video(frames, str(path), fps=30)
    decoded, info = ffmpeg.decode_video(str(path))
    assert decoded.shape == frames.shape
    assert info["width"] == 160 and info["height"] == 120
    assert metrics.psnr(frames, decoded) > 30


@needs_ffmpeg
def test_vmaf_identical_files(tmp_path):
    frames = samples.make_video_frames(8, 160, 120, seed=5)
    path = tmp_path / "a.mp4"
    ffmpeg.encode_video(frames, str(path))
    score = ffmpeg.vmaf_score(str(path), str(path))
    assert score is None or score > 95


@needs_ffmpeg
def test_sample_diff_end_to_end(tmp_path):
    clean = samples.make_video_frames(8, 160, 120, seed=6)
    watermarked = np.stack([qim.embed(f, common.payload_bits(1, 32)) for f in clean])
    clean_path = tmp_path / "c.mp4"
    wm_path = tmp_path / "w.mp4"
    ffmpeg.encode_video(clean, str(clean_path))
    ffmpeg.encode_video(watermarked, str(wm_path))
    out = tmp_path / "out"
    report = diff_module.build_report(
        ffmpeg.decode_video(str(clean_path))[0],
        ffmpeg.decode_video(str(wm_path))[0],
        str(out),
        "s",
    )
    assert report["dct_midband_diff_mean"] > 1.0
    assert (out / "s_report.json").exists()


@needs_ffmpeg
def test_container_scan_reports_tracks(tmp_path):
    frames = samples.make_video_frames(4, 160, 120, seed=7)
    path = tmp_path / "x.mp4"
    ffmpeg.encode_video(frames, str(path))
    scan = container.scan_mp4(str(path))
    assert scan["n_trak"] >= 1
