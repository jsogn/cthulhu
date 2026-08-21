"""平台代理评估器测试：哈希自洽性、身份一致与扰动敏感性。"""

from __future__ import annotations

import numpy as np
import pytest

from cthulhu_backend import samples
from cthulhu_backend.fingerprint import adversarial, hashes, proxy
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


def test_dct_matrix_matches_scipy_orthonormal():
    from scipy.fft import dctn

    rng = np.random.default_rng(3)
    block = rng.random((32, 32))
    expected = dctn(block, norm="ortho")
    dct = adversarial.dct_matrix(32)
    np.testing.assert_allclose(dct @ block @ dct.T, expected, atol=1e-10)


def test_attack_phash_flips_bits_within_budget():
    from scipy.ndimage import gaussian_filter

    frame = gaussian_filter(np.random.default_rng(18).random((256, 128)), sigma=8)
    attacked, _, flipped = adversarial.attack_phash(frame, epsilon=0.08)
    assert flipped >= 1
    assert float(np.abs(attacked - frame).max()) <= 0.08 + 1e-6
    assert float(np.abs(attacked).max()) <= 1.0
    assert float(np.abs(attacked).min()) >= 0.0


def test_attack_phash_uncapped_flips_many_bits():
    frame = samples.make_video_frames(1, 256, 128, seed=17)[0]
    _, _, flipped = adversarial.attack_phash(frame, epsilon=None)
    assert flipped >= 8


def test_multi_hash_attack_flips_bits_within_budget():
    from scipy.ndimage import gaussian_filter

    frame = gaussian_filter(np.random.default_rng(19).random((256, 128)), sigma=8)
    out, _, flips = adversarial.multi_hash_attack(frame, epsilon=0.05)
    assert set(flips) == {"phash", "ahash", "dhash", "dct_sign"}
    assert sum(flips.values()) >= 8
    assert float(np.abs(out - frame).max()) <= 0.05 + 1e-6
    assert float(np.abs(out).max()) <= 1.0 and float(np.abs(out).min()) >= 0.0


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
