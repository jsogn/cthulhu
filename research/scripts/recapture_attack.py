#!/usr/bin/env python3
"""模拟录屏/重摄攻击链（研究用途）。

用 ffmpeg 模拟真实投放/传播中常见的再采集链路：
- 录屏：降分辨率 + 模糊 + 噪声 + 色彩偏移 + 重编码
- 重摄：轻微透视 + 摩尔纹网格 + 亮度/对比度 + 模糊 + 噪声 + 重编码
- 社交多次转码：H.264 CRF23 → H.265 CRF28 → H.264 CRF33

边界：自有/授权素材、公开预训练模型、本地离线实验，不接触任何平台线上系统。
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric

ROOT = Path(__file__).resolve().parents[2]
VSEAL_REPO = ROOT / "research" / "vendor" / "videoseal"
sys.path.insert(0, str(ROOT / "research" / "scripts"))
sys.path.insert(0, str(VSEAL_REPO))
sys.path.insert(0, str(ROOT / "backend" / "src"))
os.chdir(VSEAL_REPO)

import videoseal  # noqa: E402

from cthulhu_backend.media.ffmpeg import vmaf_score  # noqa: E402
from diffusion_natural_video import make_kenburns_video  # noqa: E402
from eot_pgd_video import aggregate_ba, detector_logits  # noqa: E402
from video_codec_attacks import read_video, run_ffmpeg, write_video  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-card", default="videoseal")
    parser.add_argument("--input-video", type=Path, default=None)
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument(
        "--out", type=Path, default=ROOT / "research" / "output" / "recapture_attack"
    )
    args = parser.parse_args()
    if args.input_video is not None and not args.input_video.is_absolute():
        args.input_video = ROOT / args.input_video

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = videoseal.load(args.model_card).to(device).eval()
    if args.input_video is not None:
        frames_np = read_video(args.input_video, size=512)
        cover = (
            torch.from_numpy(frames_np.astype(np.float32) / 255.0)
            .permute(0, 3, 1, 2)
            .to(device)
        )[: args.frames]
        print(f"loaded real video {tuple(cover.shape)} from {args.input_video}")
    else:
        cover = make_kenburns_video(
            VSEAL_REPO / "assets" / "imgs" / "1.jpg", args.frames
        ).to(device)
    with torch.no_grad():
        out = model.embed(cover, is_video=True)
    message = out["msgs"][0:1].float().to(device)
    watermarked = out["imgs_w"].detach()
    width = watermarked.shape[-1]
    wm_np = (
        (watermarked.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255)
        .round()
        .astype(np.uint8)
    )

    rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        lossless = tmpdir / "watermarked.mkv"
        write_video(wm_np, lossless, codec="ffv1")
        reference = tmpdir / "reference.mkv"
        write_video(wm_np, reference, codec="ffv1")

        def evaluate(name: str, steps: list[list[str]]) -> None:
            current = lossless
            for idx, extra in enumerate(steps):
                output = tmpdir / f"{name}_{idx}.mp4"
                run_ffmpeg(current, output, extra)
                current = output
            decoded = read_video(current, size=width)
            decoded_t = (
                torch.from_numpy(decoded.astype(np.float32) / 255.0)
                .permute(0, 3, 1, 2)
                .to(device)
            )
            with torch.no_grad():
                logits = detector_logits(model, decoded_t)
            ba, per_frame = aggregate_ba(logits, message)
            n = min(len(wm_np), len(decoded))
            row: dict[str, object] = {
                "attack": name,
                "ba": ba,
                "per_frame_ba": per_frame,
                "frames": len(decoded),
                "psnr": float(
                    np.mean(
                        [
                            psnr_metric(wm_np[i], decoded[i], data_range=255)
                            for i in range(n)
                        ]
                    )
                ),
                "ssim": float(
                    np.mean(
                        [
                            ssim_metric(
                                wm_np[i], decoded[i], channel_axis=2, data_range=255
                            )
                            for i in range(n)
                        ]
                    )
                ),
            }
            row["vmaf"] = vmaf_score(str(current), str(reference), subsample=2)
            rows.append(row)
            vmaf_value = row.get("vmaf")
            vmaf_text = f"{vmaf_value:.2f}" if vmaf_value is not None else "nan"
            print(
                f"{name:24s} BA={ba:.3f} per_frame={per_frame:.3f} "
                f"PSNR={row['psnr']:5.2f} SSIM={row['ssim']:.4f} VMAF={vmaf_text}"
            )

        h264 = ["-c:v", "libx264", "-preset", "medium", "-pix_fmt", "yuv420p"]
        h265 = ["-c:v", "libx265", "-preset", "medium", "-pix_fmt", "yuv420p"]

        evaluate(
            "screen_480",
            [
                [
                    "-vf",
                    "scale=480:480,gblur=sigma=0.8,"
                    "noise=alls=6:allf=t+u,"
                    "eq=brightness=0.03:contrast=1.05:saturation=0.95,"
                    "scale=512:512",
                    "-crf",
                    "23",
                    *h264,
                ]
            ],
        )
        evaluate(
            "screen_360",
            [
                [
                    "-vf",
                    "scale=360:360,gblur=sigma=1.2,"
                    "noise=alls=10:allf=t+u,"
                    "eq=brightness=0.05:contrast=1.1:saturation=0.9,"
                    "scale=512:512",
                    "-crf",
                    "28",
                    *h264,
                ]
            ],
        )
        evaluate(
            "camera_recapture",
            [
                [
                    "-vf",
                    "perspective=x0=2:y0=1:x1=W-3:y1=0:"
                    "x2=0:y2=H-2:x3=W:y3=H,"
                    "drawgrid=width=4:height=4:thickness=1:color=black@0.06,"
                    "gblur=sigma=1.0,noise=alls=10:allf=t+u,"
                    "eq=brightness=0.04:contrast=1.08:saturation=0.92",
                    "-crf",
                    "23",
                    *h264,
                ]
            ],
        )
        evaluate(
            "social_reupload",
            [
                ["-crf", "23", *h264],
                ["-vf", "scale=720:720", "-crf", "28", *h265],
                ["-vf", "scale=480:480", "-crf", "33", *h264],
            ],
        )
        evaluate(
            "recapture_heavy",
            [
                [
                    "-vf",
                    "perspective=x0=4:y0=2:x1=W-6:y1=1:"
                    "x2=1:y2=H-4:x3=W:y3=H,"
                    "drawgrid=width=5:height=5:thickness=1:color=black@0.08,"
                    "scale=360:360,gblur=sigma=1.5,"
                    "noise=alls=14:allf=t+u,"
                    "eq=brightness=0.06:contrast=1.12:saturation=0.88,"
                    "scale=512:512",
                    "-crf",
                    "28",
                    *h264,
                ]
            ],
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.out.with_suffix(".csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({k for r in rows for k in r}))
        writer.writeheader()
        writer.writerows(rows)
    md_path = args.out.with_suffix(".md")
    with md_path.open("w") as handle:
        handle.write("# 模拟录屏/重摄攻击链\n\n")
        handle.write(
            f"- 模型：`{args.model_card}`；视频：{args.frames} 帧 512x512；"
            f"载体：{'真实片段' if args.input_video else 'Ken Burns'}\n\n"
        )
        handle.write("| 攻击 | BA | 逐帧 BA | PSNR | SSIM | VMAF |\n")
        handle.write("| --- | ---: | ---: | ---: | ---: | ---: |\n")
        for row in rows:
            vmaf_value = row.get("vmaf")
            vmaf_text = f"{vmaf_value:.2f}" if vmaf_value is not None else "nan"
            handle.write(
                f"| {row['attack']} | {row['ba']:.3f} | {row['per_frame_ba']:.3f} | "
                f"{row['psnr']:.2f} | {row['ssim']:.4f} | {vmaf_text} |\n"
            )
    print(f"\nCSV: {csv_path}\nMarkdown: {md_path}")


if __name__ == "__main__":
    main()
