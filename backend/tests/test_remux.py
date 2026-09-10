"""音轨保留与队列参数契约的回归测试。"""

from __future__ import annotations

import pytest

from cthulhu_backend import db, jobs, samples, services
from cthulhu_backend.media import ffmpeg


def test_reencode_without_audio_remix_keeps_audio(tmp_path) -> None:
    """重编码时即使音频重混关闭，音轨也必须原样保留，绝不能丢失。"""
    db.init_db()
    src = str(tmp_path / "in.mp4")
    out = str(tmp_path / "out.mp4")
    video = samples.make_video_frames(8, 64, 64, seed=4)
    audio = samples.make_audio(1.0, seed=5)
    ffmpeg.encode_video_with_audio(video, audio, src, fps=30)

    services.run_desensitize(src, out, audio_remix=False)

    assert any(
        stream.get("codec_type") == "audio"
        for stream in ffmpeg.probe(out).get("streams", [])
    )
    decoded = ffmpeg.decode_audio(out)
    assert decoded is not None and len(decoded[0]) > 0


def test_echo_defeat_without_audio_remix_keeps_audio(tmp_path) -> None:
    """只开音频回声扰动、不开音频重混时，音轨仍应保留并完成回声处理。"""
    db.init_db()
    src = str(tmp_path / "in.mp4")
    out = str(tmp_path / "out.mp4")
    video = samples.make_video_frames(8, 64, 64, seed=6)
    audio = samples.make_audio(1.0, seed=7)
    ffmpeg.encode_video_with_audio(video, audio, src, fps=30)

    services.run_desensitize(src, out, audio_remix=False, echo_defeat=True)

    decoded = ffmpeg.decode_audio(out)
    assert decoded is not None and len(decoded[0]) > 0


def test_echo_defeat_skips_slowdown_for_silent_source(tmp_path) -> None:
    """源无音轨时回声水印不存在：不得为它把成片放慢 3%（帧数/时长保持不变）。"""
    db.init_db()
    src = str(tmp_path / "silent.mp4")
    out = str(tmp_path / "silent_out.mp4")
    ffmpeg.encode_video(samples.make_video_frames(30, 64, 64, seed=11), src, fps=30)

    result = services.run_desensitize(src, out, audio_remix=True, echo_defeat=True)

    in_info = ffmpeg.video_info(src)
    out_info = ffmpeg.video_info(out)
    assert result["frames"] == 30
    assert result["out_frames"] == 30
    assert out_info["duration"] == pytest.approx(in_info["duration"], abs=0.1)


def test_echo_defeat_slows_down_when_audio_present(tmp_path) -> None:
    """有音轨时回声清除仍按设计同步放慢，并如实上报成片帧数。"""
    db.init_db()
    src = str(tmp_path / "with_audio.mp4")
    out = str(tmp_path / "with_audio_out.mp4")
    video = samples.make_video_frames(30, 64, 64, seed=12)
    audio = samples.make_audio(1.0, seed=13)
    ffmpeg.encode_video_with_audio(video, audio, src, fps=30)

    result = services.run_desensitize(src, out, audio_remix=True, echo_defeat=True)

    # 1/0.97 ≈ 1.031：30 帧入 → 31 帧出。
    assert result["frames"] == 30
    assert result["out_frames"] == 31
    assert ffmpeg.video_info(out)["duration"] > ffmpeg.video_info(src)["duration"]


def test_timing_change_without_audio_remix_aligns_audio(tmp_path) -> None:
    """音频处理关闭但视频变速时，音轨仍须保留（对齐逻辑由管线保证）。"""
    db.init_db()
    src = str(tmp_path / "in.mp4")
    out = str(tmp_path / "out.mp4")
    video = samples.make_video_frames(12, 64, 64, seed=8)
    audio = samples.make_audio(1.0, seed=9)
    ffmpeg.encode_video_with_audio(video, audio, src, fps=30)

    services.run_desensitize(src, out, audio_remix=False, speed=1.25)

    streams = ffmpeg.probe(out)["streams"]
    assert any(stream.get("codec_type") == "video" for stream in streams)
    assert any(stream.get("codec_type") == "audio" for stream in streams)
    decoded = ffmpeg.decode_audio(out)
    assert decoded is not None and len(decoded[0]) > 0


def test_queue_accepts_full_snake_case_options(tmp_path) -> None:
    """队列路径必须接受 UI 经 toSnakeOptions 生成的完整 snake_case 参数。"""
    db.init_db()
    src = str(tmp_path / "in.mp4")
    out = str(tmp_path / "out.mp4")
    ffmpeg.encode_video(samples.make_video_frames(8, 64, 64, seed=3), src, fps=30)

    options = {
        "output": out,
        "reorder": False,
        "speed": 1.0,
        "recrop": 0.0,
        "perturb": 0.15,
        "regrade": True,
        "audio_remix": False,
        "sharpness": False,
        "color_restore": False,
        "denoise": False,
        "anti_reembed": False,
        "seed": 0,
        "codec": "libx264",
        "lossless": False,
        "spoof": False,
        "skip_vmaf": True,
        "filter_scale": 0,
    }
    result = jobs._run_desensitize(src, options)
    assert result["output"] == out
