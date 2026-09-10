#!/usr/bin/env python3
"""端到端验证：用 TAESD 快路径顶替扩散净化，量真实管线的 s/帧与画质。

不改产品代码：临时把 `purify.purify_frames` 换成 TAESD 候选实现，
其余（解码、编码、封装、指标）全走真实 `services.run_desensitize`。

用法：
    backend/.venv/bin/python research/scripts/measure_fast_e2e.py --frames 300 --edge 256
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for extra in (ROOT / "research" / "scripts", ROOT / "backend" / "src"):
    sys.path.insert(0, str(extra))

import purify_fast_candidates as fast  # noqa: E402

from cthulhu_backend import services  # noqa: E402
from cthulhu_backend.media import ffmpeg  # noqa: E402
from cthulhu_backend.transform import purify  # noqa: E402

ASSET = ROOT / "research" / "vendor" / "videoseal" / "assets" / "imgs" / "1.jpg"


def build_clip(path: Path, frames: int, width: int, height: int, fps: int = 30) -> None:
    if path.exists():
        return
    import subprocess

    crop_w = int(1080 * width / height)
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error", "-loop", "1", "-i", str(ASSET),
            "-vf", f"crop={crop_w}:1080:0:0,scale={width}:{height}",
            "-t", f"{frames / fps}", "-r", str(fps),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-pix_fmt", "yuv420p", str(path),
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--edge", type=int, default=256)
    parser.add_argument("--detail", type=float, default=0.5)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=1280)
    parser.add_argument("--no-metrics", action="store_true", help="按 GUI 任务路径关闭指标遍")
    parser.add_argument("--floor", action="store_true", help="只测管线地板（不开净化）")
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/cthulhu_fast_e2e"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    clip = args.out_dir / f"cover_{args.width}x{args.height}_{args.frames}.mp4"
    build_clip(clip, args.frames, args.width, args.height)
    output = args.out_dir / f"out_taesd{args.edge}.mp4"

    attacks = fast.make_attacks("mps", args.batch)
    key = f"taesd_fast{args.edge}"
    if key not in attacks:
        raise SystemExit(f"未知档位：{key}（可用：{sorted(attacks)}）")
    attack = attacks[key]
    # MPS 首次遇到新形状要编译 kernel：预热放在计时之外。
    attack(np.zeros((args.batch, 128, 128, 3), dtype=np.uint8))

    calls = {"count": 0, "seconds": 0.0}

    def patched(frames, **kwargs):  # noqa: ANN001, ARG001 - 接管净化入口
        start = time.perf_counter()
        result = attack(np.asarray(frames))
        calls["seconds"] += time.perf_counter() - start
        calls["count"] += len(frames)
        return result

    if not args.floor:
        purify.purify_frames = patched

    progress_marks: list[str] = []

    def progress(fraction: float, note: str) -> None:
        progress_marks.append(note)

    start = time.perf_counter()
    result = services.run_desensitize(
        str(clip),
        str(output),
        compute_metrics=not args.no_metrics,
        progress_cb=progress,
        purify_strength=0.0 if args.floor else 0.10,
        purify_steps=20,
        purify_detail=0.5,
        purify_max_edge=args.edge,
        purify_batch=args.batch,
        audio_remix=True,
        echo_defeat=False,
        auto_profile=True,
        sharpness=True,
        color_restore=True,
        denoise=True,
        seed=5,
    )
    elapsed = time.perf_counter() - start
    info = ffmpeg.video_info(str(output))
    print(
        f"\n候选=TAESD@{args.edge} detail={args.detail} batch={args.batch}\n"
        f"帧数={args.frames} 分辨率={args.width}x{args.height}\n"
        f"端到端 {elapsed:.1f}s → {elapsed / args.frames:.3f}s/帧 "
        f"（净化部分 {calls['seconds']:.1f}s / {calls['count']} 帧 = "
        f"{calls['seconds'] / max(calls['count'], 1):.3f}s/帧）\n"
        f"成片 {info['duration']:.3f}s（输入 {args.frames / 30:.3f}s）"
        f" → 处理/时长 = {elapsed / (args.frames / 30):.2f}×\n"
        f"PSNR={result.get('psnr_db')} SSIM={result.get('ssim')} "
        f"out_frames={result.get('out_frames')}",
        flush=True,
    )


if __name__ == "__main__":
    main()
