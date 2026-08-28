"""身份层输出（重封装）与队列参数契约的回归测试。"""

from __future__ import annotations

import subprocess

from cthulhu_backend import db, jobs, samples, services
from cthulhu_backend.media import ffmpeg


def _video_stream(path: str) -> bytes:
    cmd = [
        ffmpeg.FFMPEG_BIN, "-v", "error", "-i", path,
        "-map", "0:v", "-c:v", "copy", "-bsf:v", "h264_mp4toannexb", "-f", "h264", "-",
    ]
    return subprocess.run(cmd, capture_output=True, check=True).stdout


def _audio_stream(path: str) -> bytes:
    cmd = [
        ffmpeg.FFMPEG_BIN, "-v", "error", "-i", path,
        "-map", "0:a", "-c:a", "copy", "-f", "adts", "-",
    ]
    return subprocess.run(cmd, capture_output=True, check=True).stdout


def test_remux_keeps_audio_and_video_bit_identical(tmp_path) -> None:
    db.init_db()
    src = str(tmp_path / "in.mp4")
    out = str(tmp_path / "out.mp4")
    video = samples.make_video_frames(10, 64, 64, seed=1)
    audio = samples.make_audio(1.0, seed=2)
    ffmpeg.encode_video_with_audio(video, audio, src, fps=30)

    report = services.run_desensitize(src, out, output_mode="remux")

    assert report["mode"] == "remux"
    assert report["similarity_after"]["content_cosine"] == 1.0
    assert report["psnr_db"] is None
    assert _video_stream(src) == _video_stream(out)
    assert _audio_stream(src) == _audio_stream(out)


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
        "output_mode": "reencode",
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
        "detail_protect": 0.0,
        "seed": 0,
        "codec": "libx264",
        "lossless": False,
        "spoof": False,
        "skip_vmaf": True,
        "filter_scale": 0,
    }
    result = jobs._run_desensitize(src, options)
    assert result["output"] == out
