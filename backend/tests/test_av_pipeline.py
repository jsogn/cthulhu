"""音画联合管线测试（依赖 ffmpeg）。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from cthulhu_backend import samples, services
from cthulhu_backend.evaluate import metrics
from cthulhu_backend.main import app
from cthulhu_backend.media import ffmpeg

needs_ffmpeg = pytest.mark.skipif(not ffmpeg.has_ffmpeg(), reason="需要 ffmpeg/ffprobe")

client = TestClient(app, headers={"X-CTHULHU-Token": "test-token"})


@needs_ffmpeg
def test_audio_decode_roundtrip(tmp_path):
    video = samples.make_cut_video(4, 8, 160, 120, seed=30)
    audio = samples.make_audio(1.0, seed=30)
    path = tmp_path / "av.mp4"
    ffmpeg.encode_video_with_audio(video, audio, str(path), fps=30)
    info = ffmpeg.probe(str(path))
    assert any(s.get("codec_type") == "audio" for s in info["streams"])
    decoded, sample_rate = ffmpeg.decode_audio(str(path))
    assert decoded is not None and sample_rate == 16000 and len(decoded) > 1000


@needs_ffmpeg
def test_decode_audio_none_without_track(tmp_path):
    video = samples.make_video_frames(8, 160, 120, seed=31)
    path = tmp_path / "v.mp4"
    ffmpeg.encode_video(video, str(path), fps=30)
    assert ffmpeg.decode_audio(str(path)) is None


@needs_ffmpeg
def test_desensitize_remixes_audio(tmp_path):
    video = samples.make_cut_video(4, 8, 160, 120, seed=32)
    audio = samples.make_audio(1.0, seed=32)
    src = tmp_path / "src.mp4"
    out = tmp_path / "out.mp4"
    ffmpeg.encode_video_with_audio(video, audio, str(src), fps=30)

    response = client.post(
        "/api/desensitize",
        json={"path": str(src), "output": str(out), "audio_remix": True},
    )
    assert response.status_code == 200, response.text
    out_audio, _ = ffmpeg.decode_audio(str(out))
    src_audio, _ = ffmpeg.decode_audio(str(src))
    assert out_audio is not None and src_audio is not None
    length = min(len(src_audio), len(out_audio))
    assert float(np.mean(np.abs(src_audio[:length] - out_audio[:length]))) > 1e-3


@needs_ffmpeg
def test_streaming_decoder_matches_range_decode(tmp_path):
    video = samples.make_cut_video(3, 40, 160, 120, seed=33)
    path = tmp_path / "src.mp4"
    ffmpeg.encode_video(video, str(path), fps=30)
    expected, _ = ffmpeg.decode_video_range(str(path), 30, 50)
    decoder = ffmpeg.StreamingDecoder(str(path), 30, 50)
    chunks = []
    while True:
        batch = decoder.read(17)
        if len(batch) == 0:
            break
        chunks.append(batch)
    decoder.close()
    actual = np.concatenate(chunks, axis=0) if chunks else np.empty_like(expected)
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected)


@needs_ffmpeg
def test_desensitize_fps_out(tmp_path):
    """输出帧率参数应真实改变产物帧率。"""
    video = samples.make_video_frames(24, 160, 120, seed=42)
    src = tmp_path / "src.mp4"
    out = tmp_path / "out.mp4"
    ffmpeg.encode_video(video, str(src), fps=30)
    response = client.post(
        "/api/desensitize",
        json={
            "path": str(src),
            "output": str(out),
            "audio_remix": False,
            "fps_out": 15,
            "reorder": False,
            "regrade": False,
            "sharpness": False,
            "color_restore": False,
        },
    )
    assert response.status_code == 200, response.text
    assert ffmpeg.video_info(str(out))["fps"] == 15


@needs_ffmpeg
def test_aligned_vmaf_recovers_spatial_shift(tmp_path):
    frames = samples.make_video_frames(16, 160, 120, seed=33)
    shifted = np.stack([np.roll(np.roll(frame, 2, axis=0), -3, axis=1) for frame in frames])
    reference = tmp_path / "r.mp4"
    distorted = tmp_path / "d.mp4"
    ffmpeg.encode_video(frames, str(reference), fps=30)
    ffmpeg.encode_video(shifted, str(distorted), fps=30)

    naive = ffmpeg.vmaf_score(str(distorted), str(reference))
    aligned = ffmpeg.aligned_vmaf(str(distorted), str(reference))
    assert naive is not None and aligned is not None
    assert aligned > naive + 10
    assert aligned > 70


@needs_ffmpeg
def test_hardware_decode_matches_software(tmp_path):
    frames = samples.make_video_frames(16, 160, 120, seed=34)
    path = tmp_path / "h.mp4"
    ffmpeg.encode_video(frames, str(path), fps=30)
    software, _ = ffmpeg.decode_video(str(path))
    try:
        hardware, _ = ffmpeg.decode_video(str(path), hwaccel=True)
    except Exception:  # noqa: BLE001 - 环境无硬解时跳过
        pytest.skip("当前环境无可用硬件解码")
    assert hardware.shape == software.shape
    assert metrics.psnr(software, hardware) > 20


@needs_ffmpeg
def test_hardware_encode_roundtrip(tmp_path):
    frames = samples.make_video_frames(16, 160, 120, seed=35)
    path = tmp_path / "hw.mp4"
    try:
        ffmpeg.encode_video(frames, str(path), fps=30, hardware=True)
    except Exception:  # noqa: BLE001 - 环境无硬编时跳过
        pytest.skip("当前环境无可用硬件编码")
    decoded, _ = ffmpeg.decode_video(str(path))
    assert decoded.shape == frames.shape


@needs_ffmpeg
def test_hardware_h265_encode_roundtrip(tmp_path):
    """H.265 在硬件模式下应映射到 hevc_videotoolbox 并可解码回读。"""
    import sys

    if sys.platform != "darwin":
        pytest.skip("hevc_videotoolbox 仅 macOS 可用")
    frames = samples.make_video_frames(8, 160, 120, seed=36)
    path = tmp_path / "h265.mp4"
    try:
        ffmpeg.encode_video(frames, str(path), fps=30, codec="libx265", hardware=True)
    except Exception:  # noqa: BLE001 - 环境无硬编时跳过
        pytest.skip("当前环境无可用硬件编码")
    info = ffmpeg.video_info(str(path))
    assert info["codec"] == "hevc"
    decoded, _ = ffmpeg.decode_video(str(path))
    assert decoded.shape == frames.shape


@needs_ffmpeg
def test_desensitize_stream_with_audio_remix(tmp_path):
    """流式清洗后音轨应独立重混并 mux 回产物。"""
    video = samples.make_video_frames(12, 160, 120, seed=5)
    audio = samples.make_audio(0.5, seed=5)
    source = tmp_path / "av.mp4"
    output = tmp_path / "out.mp4"
    ffmpeg.encode_video_with_audio(video, audio, str(source), fps=30)
    services.run_desensitize(
        str(source),
        str(output),
        reorder=False,
        speed=1.0,
        regrade=False,
        audio_remix=True,
        sharpness=False,
        color_restore=False,
        seed=1,
    )
    assert output.exists()
    streams = ffmpeg.probe(str(output))["streams"]
    assert any(stream.get("codec_type") == "audio" for stream in streams)


@needs_ffmpeg
def test_desensitize_accepts_camel_case_anti_options(tmp_path):
    """camelCase 参数应经别名映射生效，含指纹对抗与旋转。"""
    video = samples.make_cut_video(2, 8, 160, 120, seed=35)
    source = tmp_path / "src.mp4"
    output = tmp_path / "out.mp4"
    ffmpeg.encode_video(video, str(source), fps=30)
    response = client.post(
        "/api/desensitize",
        json={
            "path": str(source),
            "output": str(output),
            "audioRemix": False,
            "colorRestore": False,
            "sharpness": False,
            "rotate": 1.2,
            "phashAttack": True,
            "phashEpsilon": 0.03,
            "dctStep": 12,
            "requant": 32,
            "chromaLevels": 32,
            "dropEvery": 7,
            "subtractBeta": 1.2,
        },
    )
    assert response.status_code == 200, response.text
    report = response.json()
    assert report["transform_strategy"] == "fast"
    assert report["quality_metrics_na"] is True
    assert report["psnr_db"] is None
    assert output.exists()


@needs_ffmpeg
def test_desensitize_preserves_color(tmp_path):
    """彩色链路应保留三通道与色彩主序，且抗档 3 通道路径可跑通。"""
    rng = np.random.default_rng(6)
    frames = rng.random((8, 64, 48, 3), dtype=np.float32)
    frames[..., 0] = 0.9
    frames[..., 1] = 0.25
    frames[..., 2] = 0.1
    source = tmp_path / "color.mp4"
    ffmpeg.encode_video(frames, str(source), fps=30)

    plain = tmp_path / "plain.mp4"
    services.run_desensitize(
        str(source),
        str(plain),
        reorder=False,
        speed=1.0,
        regrade=False,
        audio_remix=False,
        sharpness=False,
        color_restore=False,
        denoise=False,
    )
    decoded, _ = ffmpeg.decode_video(str(plain), grayscale=False)
    assert decoded.shape[-1] == 3
    means = decoded.reshape(len(decoded), -1, 3).mean(axis=1).mean(axis=0)
    assert means[0] > means[1] > means[2]

    anti = tmp_path / "anti.mp4"
    services.run_desensitize(
        str(source),
        str(anti),
        reorder=False,
        speed=1.0,
        regrade=False,
        audio_remix=False,
        sharpness=False,
        color_restore=False,
        denoise=False,
        transform_strategy="fast",
        rotate=1.2,
        phash_attack=True,
        phash_epsilon=0.03,
    )
    decoded_anti, _ = ffmpeg.decode_video(str(anti), grayscale=False)
    assert decoded_anti.shape[-1] == 3
    anti_means = decoded_anti.reshape(len(decoded_anti), -1, 3).mean(axis=1).mean(axis=0)
    assert anti_means[0] > anti_means[1] > anti_means[2]


@needs_ffmpeg
def test_desensitize_transcode_chain(tmp_path):
    """编码域组合拳：二次转码后产物仍有效且内容保持。"""
    video = samples.make_cut_video(2, 8, 160, 120, seed=36)
    source = tmp_path / "src.mp4"
    output = tmp_path / "out.mp4"
    ffmpeg.encode_video(video, str(source), fps=30)
    services.run_desensitize(
        str(source),
        str(output),
        reorder=False,
        speed=1.0,
        regrade=False,
        audio_remix=False,
        sharpness=False,
        color_restore=False,
        denoise=False,
        transcode_chain=True,
    )
    assert output.exists()
    decoded, info = ffmpeg.decode_video(str(output))
    assert decoded.shape == video.shape
    assert info["codec"] == "h264"


@needs_ffmpeg
def test_make_preview_montage(tmp_path):
    """并排预览拼图应生成在原产物旁。"""
    video = samples.make_cut_video(2, 8, 160, 120, seed=37)
    source = tmp_path / "src.mp4"
    output = tmp_path / "out.mp4"
    ffmpeg.encode_video(video, str(source), fps=30)
    services.run_desensitize(
        str(source),
        str(output),
        reorder=False,
        speed=1.0,
        regrade=False,
        audio_remix=False,
        sharpness=False,
        color_restore=False,
        denoise=False,
    )
    preview = services.make_preview_montage(str(source), str(output))
    assert preview == str(output) + ".preview.png"
    assert Path(preview).exists() and Path(preview).stat().st_size > 0


@needs_ffmpeg
def test_generate_candidates_ranked(tmp_path):
    """多候选生成应产出按低损优选评分排序的差异化文件。"""
    video = samples.make_cut_video(2, 8, 160, 120, seed=38)
    source = tmp_path / "src.mp4"
    ffmpeg.encode_video(video, str(source), fps=30)
    report = services.generate_candidates(
        str(source),
        str(tmp_path / "candidates"),
        count=2,
        options={
            "reorder": False,
            "speed": 1.0,
            "regrade": False,
            "audio_remix": False,
            "sharpness": False,
            "color_restore": False,
            "denoise": False,
        },
    )
    assert len(report["candidates"]) == 2
    assert report["candidates"][0]["score"] >= report["candidates"][1]["score"]
    for candidate in report["candidates"]:
        assert Path(candidate["output"]).exists()
        assert candidate["stability_ratio"] is not None
        assert candidate["export_health"]["video_codec"] == "h264"
