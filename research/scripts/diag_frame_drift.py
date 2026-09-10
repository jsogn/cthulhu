#!/usr/bin/env python3
"""诊断：逐阶段统计 run_desensitize 实际写出的帧数，定位成片帧数漂移。

用法：
    backend/.venv/bin/python research/scripts/diag_frame_drift.py \
        --input /tmp/clip.mp4 --output /tmp/out.mp4 --echo-defeat

输出：输入帧数/时长、管线写帧数、state（total_in/total_out/speed/分段）、
成片文件的帧数与时长。用于区分「设计内的同步变速」与「封装/拼接补帧」。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend" / "src"))

from cthulhu_backend import pipeline, services  # noqa: E402
from cthulhu_backend.media import ffmpeg  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--echo-defeat", action="store_true", help="打开音频回声扰动")
    parser.add_argument("--audio-remix", action="store_true", help="打开音频重混")
    parser.add_argument("--purify-strength", type=float, default=0.0)
    args = parser.parse_args()

    written = {"frames": 0, "segments": []}
    captured: dict[str, object] = {}
    orig_process = pipeline._process_segment_yuv
    orig_write = ffmpeg.StreamingEncoder.write
    orig_encode = services._encode_desensitize

    def spy_process(*call_args, **call_kwargs):
        count = orig_process(*call_args, **call_kwargs)
        written["segments"].append(count)
        return count

    def spy_write(encoder, frames):
        written["frames"] += len(frames)
        return orig_write(encoder, frames)

    def spy_encode(path, output, opts, state, progress_cb, should_stop, pause):
        captured["state"] = state
        return orig_encode(path, output, opts, state, progress_cb, should_stop, pause)

    pipeline._process_segment_yuv = spy_process
    ffmpeg.StreamingEncoder.write = spy_write
    services._encode_desensitize = spy_encode

    result = services.run_desensitize(
        args.input,
        args.output,
        compute_metrics=False,
        purify_strength=args.purify_strength,
        audio_remix=args.audio_remix,
        echo_defeat=args.echo_defeat,
        sharpness=True,
        color_restore=True,
        denoise=False,
    )

    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
            "-show_entries", "stream=nb_read_frames,duration,avg_frame_rate",
            "-of", "default=nw=1", args.output,
        ],
        capture_output=True,
        text=True,
    )
    info = ffmpeg.video_info(args.input)
    print(f"输入: 时长={info['duration']}s fps={info['fps']} 含音轨={ffmpeg.has_audio(args.input)}")
    print(f"管线写帧: {written['frames']} 段计数: {written['segments']}")
    state = captured.get("state")
    if state is not None:
        print(
            "state: total_in={} total_out={} speed={} audio_tempo={} segments={}".format(
                state.total_in, state.total_out, state.speed, state.audio_tempo, state.segments
            )
        )
    print(f"结果字段: frames={result.get('frames')} out_frames={result.get('out_frames')}")
    print("成片:", probe.stdout.strip().replace("\n", " "))


if __name__ == "__main__":
    main()
