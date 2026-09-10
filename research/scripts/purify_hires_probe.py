#!/usr/bin/env python3
"""高分辨率下的净化画质/清除率探针：回答「1080p 素材该用多大长边」。

背景：产品默认把长边压到 192/256 再重建，在 512 宽素材上是 2 倍放大（可接受），
但在 1080×1920 上是 5~7 倍放大 → 画面不可看。本脚本在 1080×1920 竖直素材上
扫描长边取值，同时给出 BA@h264（清除率）与对原帧 PSNR/SSIM（保真度），并输出
字幕带对比图供肉眼确认。

用法：
    research/.venv/bin/python research/scripts/purify_hires_probe.py --edge 256 384 512 640
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
for extra in (ROOT / "research" / "scripts", ROOT / "backend" / "src"):
    sys.path.insert(0, str(extra))

ASSET = ROOT / "research" / "vendor" / "videoseal" / "assets" / "imgs" / "1.jpg"
FONT_CANDIDATES = (
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
)
SUBTITLE = "皇帝这才将女孩当成工具送出去和亲"


def _font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    raise SystemExit("缺少中文字体，无法生成字幕样本")


def build_frames(count: int, width: int, height: int) -> np.ndarray:
    """竖直 kenburns + 烧入清晰字幕：模拟真实短剧画面（有文字、有纹理）。"""
    image = Image.open(ASSET).convert("RGB")
    crop_w = int(image.size[1] * width / height)
    font = _font(int(height * 0.030))
    frames: list[np.ndarray] = []
    for index in range(count):
        frac = index / max(count - 1, 1)
        x0 = int((image.size[0] - crop_w) * frac)
        frame = image.crop((x0, 0, x0 + crop_w, image.size[1])).resize(
            (width, height), Image.LANCZOS
        )
        draw = ImageDraw.Draw(frame)
        box = draw.textbbox((0, 0), SUBTITLE, font=font)
        text_w = box[2] - box[0]
        x = (width - text_w) // 2
        y = int(height * 0.86)
        draw.text((x + 2, y + 2), SUBTITLE, font=font, fill=(0, 0, 0))
        draw.text((x, y), SUBTITLE, font=font, fill=(255, 255, 255))
        frames.append(np.asarray(frame))
    return np.stack(frames)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edges", type=int, nargs="+", default=[256, 384, 512, 640])
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--width", type=int, default=1080)
    parser.add_argument("--height", type=int, default=1920)
    parser.add_argument("--sigma", type=float, default=1.5)
    parser.add_argument("--detail", type=float, default=1.0)
    parser.add_argument(
        "--sigmas", type=float, nargs="+", default=None, help="σ 扫描（像素单位）"
    )
    parser.add_argument(
        "--details", type=float, nargs="+", default=None, help="回注强度扫描（可 >1 过冲）"
    )
    parser.add_argument("--schemes", nargs="+", default=["videoseal", "wam"])
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "research" / "output" / "hires_probe",
    )
    args = parser.parse_args()

    import torch

    from purify_speed_tiers import build_seal, build_wam_frames, h264_crf23, quality

    from cthulhu_backend.transform import purify

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    frames = build_frames(args.frames, args.width, args.height)
    print(f"素材 {args.width}x{args.height} × {args.frames} 帧，device={device}", flush=True)

    builders = {
        "videoseal": lambda: build_seal(device, "videoseal"),
        "pixelseal": lambda: build_seal(device, "pixelseal"),
        "wam": lambda: build_wam_frames(device),
    }
    sigmas = args.sigmas or [args.sigma]
    details = args.details or [args.detail]
    combos = [(edge, sigma, detail) for edge in args.edges for sigma in sigmas for detail in details]
    rows: list[dict] = []
    crops: list[np.ndarray] = []
    for scheme in args.schemes:
        builder = builders.get(scheme)
        if builder is None:
            continue
        _, embed_video, ba_video = builder()
        watermarked, message = embed_video(frames)
        clean_ba, _ = ba_video(watermarked, message)
        for edge, sigma, detail in combos:
            out = purify.purify_frames(
                watermarked,
                strength=0.15,
                max_edge=edge,
                batch=args.batch,
                detail=detail,
                detail_sigma=sigma,
            )
            ba, _ = ba_video(out, message)
            ba_h264, _ = ba_video(h264_crf23(out), message)
            psnr, ssim = quality(watermarked, out)
            psnr_src, ssim_src = quality(frames, out)
            rows.append(
                {
                    "scheme": scheme,
                    "edge": edge,
                    "sigma": sigma,
                    "detail": detail,
                    "clean_ba": round(clean_ba, 4),
                    "ba": round(ba, 4),
                    "ba_h264": round(ba_h264, 4),
                    "psnr": round(psnr, 2),
                    "ssim": round(ssim, 4),
                    "psnr_src": round(psnr_src, 2),
                    "ssim_src": round(ssim_src, 4),
                }
            )
            print(
                f"  {scheme:<10} edge={edge:<4} σ={sigma:<4} d={detail:<4} "
                f"BA={ba:.4f} BA@h264={ba_h264:.4f} "
                f"对原帧PSNR={psnr_src:5.2f} SSIM={ssim_src:.4f}",
                flush=True,
            )
            if scheme == args.schemes[0]:
                band = slice(int(args.height * 0.84), int(args.height * 0.90))
                crops.append(out[0][band])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    band = slice(int(args.height * 0.84), int(args.height * 0.90))
    strip = np.concatenate([frames[0][band], *crops], axis=1)
    Image.fromarray(strip).save(args.out.with_name(args.out.name + "_subtitle.png"))
    header = "scheme,edge,clean_ba,ba,ba_h264,psnr,ssim,psnr_src,ssim_src"
    lines = [header] + [
        ",".join(str(row[key]) for key in header.split(",")) for row in rows
    ]
    args.out.with_suffix(".csv").write_text("\n".join(lines) + "\n")
    print(f"字幕带对比（原帧 | " + " | ".join(f"@{e}" for e in args.edges) + f"）：{args.out}_subtitle.png")


if __name__ == "__main__":
    main()
