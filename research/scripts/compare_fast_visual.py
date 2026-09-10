#!/usr/bin/env python3
"""视觉对比：原帧 / 扩散净化 / TAESD 快路径，输出并排 PNG 供人工检查。

用法：
    backend/.venv/bin/python research/scripts/compare_fast_visual.py --edge 256
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
for extra in (ROOT / "research" / "scripts", ROOT / "backend" / "src"):
    sys.path.insert(0, str(extra))

import purify_fast_candidates as fast  # noqa: E402

from cthulhu_backend.transform import purify  # noqa: E402

ASSET = ROOT / "research" / "vendor" / "videoseal" / "assets" / "imgs" / "1.jpg"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edge", type=int, default=256)
    parser.add_argument("--content", default="kenburns", choices=["kenburns", "real"])
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "research" / "output" / "fast_visual_compare.png",
    )
    args = parser.parse_args()

    if args.content == "real":
        from purify_speed_tiers import read_frames

        frames = read_frames(ROOT / "research" / "data" / "real_clip_512.mp4", args.frames)
        height, width = frames.shape[1], frames.shape[2]
    else:
        image = Image.open(ASSET).convert("RGB")
        width, height = 720, 1280
        crop_w = int(image.size[1] * width / height)
        collected = []
        for index in range(args.frames):
            frac = index / max(args.frames - 1, 1)
            x0 = int((image.size[0] - crop_w) * frac)
            collected.append(
                np.asarray(
                    image.crop((x0, 0, x0 + crop_w, image.size[1])).resize(
                        (width, height), Image.BICUBIC
                    )
                )
            )
        frames = np.stack(collected)

    attacks = fast.make_attacks("mps", args.batch)
    attacks[f"taesd_fast{args.edge}"](np.zeros((args.batch, 64, 64, 3), dtype=np.uint8))
    fast_out = attacks[f"taesd_fast{args.edge}"](frames)
    diffusion = purify.purify_frames(
        frames, strength=0.10, steps=20, detail=0.5, max_edge=args.edge, batch=args.batch
    )

    # 取一帧的上半身区域放大对比（人脸/纹理最敏感）。
    pick = 0
    band_height = min(512, max(64, height))
    band = slice(int(height * 0.10), int(height * 0.10) + band_height)
    tiles = [
        frames[pick][band],
        diffusion[pick][band],
        fast_out[pick][band],
    ]
    strip = np.concatenate(tiles, axis=1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(strip).save(args.out)

    def stats(name: str, candidate: np.ndarray) -> None:
        diff = candidate.astype(np.float32) - frames.astype(np.float32)
        mse = float(np.mean(diff**2))
        psnr = 10 * np.log10(255.0**2 / mse) if mse > 0 else float("inf")
        print(f"{name}: PSNR={psnr:.2f}dB")

    stats("扩散净化", diffusion)
    stats(f"TAESD@{args.edge}", fast_out)
    print(f"对比图（原帧 | 扩散 | TAESD）：{args.out}")


if __name__ == "__main__":
    main()
