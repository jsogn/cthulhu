#!/usr/bin/env python3
"""端到端测速：按 GUI 任务路径跑一遍「极速净化」档，量真实 s/帧。

与 `purify_speed_tiers.py` 只测扩散原语不同，这里走完整清洗管线
（解码 → 变换/净化 → 编码 → 封装），得到可直接换算成视频时长的数字。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend" / "src"))

from cthulhu_backend import services  # noqa: E402

ASSET = ROOT / "research" / "vendor" / "videoseal" / "assets" / "imgs" / "1.jpg"

TIERS = {
    "turbo": {
        "purify_strength": 0.10,
        "purify_steps": 20,
        "purify_detail": 0.5,
        "purify_max_edge": 256,
        "purify_batch": 8,
    },
    "fast": {
        "purify_strength": 0.15,
        "purify_steps": 20,
        "purify_detail": 0.0,
        "purify_max_edge": 512,
        "purify_batch": 4,
    },
    "original": {
        "purify_strength": 0.15,
        "purify_steps": 20,
        "purify_detail": 0.0,
        "purify_max_edge": 0,
        "purify_batch": 1,
    },
}


def make_clip(path: Path, frames: int, width: int, height: int, fps: int = 30) -> None:
    if path.exists():
        return
    crop_w = int(1080 * width / height)
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error", "-loop", "1", "-i", str(ASSET),
            "-vf", f"crop={crop_w}:1080:0:0,scale={width}:{height}",
            "-t", f"{frames / fps}", "-r", str(fps),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", str(path),
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", default="turbo", choices=sorted(TIERS))
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=1280)
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/cthulhu_tier_e2e"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    clip = args.out_dir / f"cover_{args.width}x{args.height}_{args.frames}.mp4"
    make_clip(clip, args.frames, args.width, args.height)
    output = args.out_dir / f"out_{args.tier}.mp4"

    last = {"note": "", "t": time.perf_counter()}

    def progress(fraction: float, note: str) -> None:
        now = time.perf_counter()
        if note != last["note"]:
            print(f"  [{fraction:5.1%}] {note}  (+{now - last['t']:.1f}s)", flush=True)
            last["note"] = note
            last["t"] = now

    options = dict(
        audio_remix=True,
        echo_defeat=True,
        audio_strong=True,
        embedding_attack="both",
        embedding_strength=0.0,
        auto_profile=True,
        sharpness=True,
        color_restore=True,
        denoise=True,
        **TIERS[args.tier],
    )
    start = time.perf_counter()
    result = services.run_desensitize(
        str(clip),
        str(output),
        compute_metrics=False,
        progress_cb=progress,
        **options,
    )
    elapsed = time.perf_counter() - start
    print(
        f"\n档位={args.tier} 帧数={args.frames} 分辨率={args.width}x{args.height}\n"
        f"总耗时 {elapsed:.1f}s → {elapsed / args.frames:.3f}s/帧\n"
        f"结果键：{sorted(result)[:8]}",
        flush=True,
    )


if __name__ == "__main__":
    main()
