#!/usr/bin/env python3
"""长片实机：3 分钟 720p 素材走产品潜空间引擎，量端到端耗时与画质。

用法：
    backend/.venv/bin/python research/scripts/measure_latent_long.py \
        --input /tmp/cthulhu_fast_e2e/long720.mp4 --output /tmp/cthulhu_fast_e2e/long_out.mp4
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend" / "src"))

from cthulhu_backend import services  # noqa: E402
from cthulhu_backend.media import ffmpeg  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--engine", default="latent", choices=["latent", "diffusion"])
    parser.add_argument("--edge", type=int, default=128)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--frames", type=int, default=0, help="限流：只处理前 N 帧（0=全片）")
    args = parser.parse_args()

    info = ffmpeg.video_info(args.input)
    marks: list[str] = []
    last = {"t": time.perf_counter(), "note": ""}

    def progress(fraction: float, note: str) -> None:
        now = time.perf_counter()
        if note != last["note"] or now - last["t"] > 30:
            marks.append(f"{fraction:.1%} {note} (+{now - last['t']:.0f}s)")
            last["note"], last["t"] = note, now

    start = time.perf_counter()
    result = services.run_desensitize(
        args.input,
        args.output,
        compute_metrics=True,
        progress_cb=progress,
        purify_strength=0.10,
        purify_engine=args.engine,
        purify_max_edge=args.edge,
        purify_batch=args.batch,
        purify_detail=0.5,
        audio_remix=True,
        echo_defeat=True,
        audio_strong=True,
        auto_profile=True,
        sharpness=True,
        color_restore=True,
        denoise=True,
        seed=5,
    )
    elapsed = time.perf_counter() - start
    frames = int(info["duration"] * info["fps"])
    print(f"输入 {info['duration']:.1f}s / {frames} 帧；处理 {elapsed:.1f}s")
    print(f"→ {elapsed / frames * 1000:.1f} ms/帧 · 处理/时长 = {elapsed / info['duration']:.2f}×")
    print(
        f"PSNR={result.get('psnr_db')} SSIM={result.get('ssim')} "
        f"VMAF={result.get('vmaf_aligned')} out_frames={result.get('out_frames')} "
        f"purify_note={result.get('purify_note')}"
    )
    print("阶段点：", " | ".join(marks[-6:]))


if __name__ == "__main__":
    main()
