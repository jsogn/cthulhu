"""平台代理评估器测试：哈希自洽性、身份一致与扰动敏感性。"""

from __future__ import annotations

import numpy as np
import pytest

from cthulhu_backend import samples
from cthulhu_backend.fingerprint import hashes, proxy
from cthulhu_backend.media import ffmpeg
from cthulhu_backend.transform import audio as audio_transform
from cthulhu_backend.transform import video as video_transform

needs_ffmpeg = pytest.mark.skipif(not ffmpeg.has_ffmpeg(), reason="需要 ffmpeg/ffprobe")


def test_frame_hashes_are_deterministic_and_balanced():
    frame = samples.make_video_frames(1, 96, 64, seed=10)[0]
    for fn in (hashes.phash, hashes.ahash, hashes.dct_sign):
        bits = fn(frame)
        assert np.array_equal(bits, fn(frame))
        assert 0.1 < bits.mean() < 0.9


def test_edge_embedding_is_normalized():
    frame = samples.make_video_frames(1, 96, 64, seed=11)[0]
    vector = hashes.edge_embedding(frame)
    assert abs(float(np.linalg.norm(vector)) - 1.0) < 1e-9


def test_video_compare_identity_is_high_risk():
    frames = samples.make_cut_video(3, 8, 160, 120, seed=12)
    report = proxy.video_compare(frames, frames)
    for kind in proxy.HASH_KINDS:
        assert report[kind]["mean_min_hamming"] == 0
        assert report[kind]["flag_rate_le_2"] == 1.0
    assert report["content_cosine"] > 0.999
    assert report["dhash"]["bits"] == 72


def test_video_compare_detects_perturbation():
    frames = samples.make_cut_video(3, 8, 160, 120, seed=13)
    recropped = video_transform.recrop(frames, 0.06)
    regraded = video_transform.regrade(frames, np.random.default_rng(0), strength=0.12)
    changed = proxy.video_compare(frames, recropped)
    changed2 = proxy.video_compare(frames, regraded)
    assert changed["phash"]["mean_min_hamming"] > 0
    assert changed["content_cosine"] < 1.0
    assert changed2["phash"]["mean_min_hamming"] >= 0
    assert changed2["content_cosine"] > 0.85  # 调光不应改变内容身份


def test_audio_hashes_identity_and_remix():
    signal = samples.make_audio(3.0, seed=14)
    rng = np.random.default_rng(0)
    remixed = audio_transform.remix(signal, 16000, rng, speed_factor=0.97)
    same = proxy.audio_compare(signal, signal, 16000)
    changed = proxy.audio_compare(signal, remixed, 16000)
    assert same["mel_hash"]["hamming"] == 0
    assert same["mfcc_hash"]["hamming"] == 0
    assert changed["mel_hash"]["hamming"] > 0


@needs_ffmpeg
def test_compare_files_risk(tmp_path):
    video = samples.make_cut_video(3, 8, 160, 120, seed=15)
    audio = samples.make_audio(1.0, seed=15)
    path = tmp_path / "clip.mp4"
    ffmpeg.encode_video_with_audio(video, audio, str(path), fps=30)
    report = proxy.compare_files(str(path), str(path))
    assert report["risk"]["level"] == "高"
    assert report["risk"]["score"] > 0.9
