#!/usr/bin/env python3
"""VideoSeal 转码攻击复核：更有纹理/运动的合成视频。

用于验证 video_codec_attacks.py 在简单渐变视频上的结论是否受内容简单影响。
生成带纹理、相机平移和运动物体的合成视频，重测 H.264/H.265 转码后的 BA。

边界：合成样本、公开预训练模型、本地离线实验，不接触任何平台线上系统。
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
os.chdir(VSEAL_REPO)

import videoseal  # noqa: E402

from video_codec_attacks import read_video, run_ffmpeg, write_video  # noqa: E402


def make_textured_video(frames: int = 16, size: int = 512) -> torch.Tensor:
    rng = np.random.default_rng(20260909)
    yy, xx = np.mgrid[0:size, 0:size]
    base = np.zeros((size, size, 3), dtype=np.float32)
    base[..., 0] = 100 + 60 * np.sin(xx / 17.0) + 40 * np.sin(yy / 23.0)
    base[..., 1] = 80 + 50 * np.sin((xx + yy) / 29.0)
    base[..., 2] = 120 + 70 * np.cos(xx / 31.0)
    base += rng.normal(0, 12, base.shape)
    for _ in range(40):
        cy, cx = rng.integers(20, size - 20, size=2)
        radius = int(rng.integers(5, 30))
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 < radius**2
        base[mask] = rng.integers(0, 255, size=3)

    out = []
    for t in range(frames):
        frame = np.roll(base, 4 * t, axis=1)
        x0 = (40 + 9 * t) % (size - 170)
        frame[110:260, x0 : x0 + 150] = [230, 80, 60]
        out.append(np.clip(frame, 0, 255).astype(np.uint8))
    video = torch.from_numpy(np.stack(out)).permute(0, 3, 1, 2).float() / 255.0
    return video


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "research" / "output" / "video_codec_textured",
    )
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = videoseal.load("videoseal").to(device).eval()
    cover = make_textured_video(args.frames).to(device)
    with torch.no_grad():
        out = model.embed(cover, is_video=True)
    message = out["msgs"][0:1].float().to(device)
    watermarked = out["imgs_w"]
    wm_np = (
        (watermarked.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255)
        .round()
        .astype(np.uint8)
    )

    attacks = [
        ("h264_crf18", ["-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p"]),
        ("h264_crf23", ["-c:v", "libx264", "-crf", "23", "-preset", "medium", "-pix_fmt", "yuv420p"]),
        ("h264_crf28", ["-c:v", "libx264", "-crf", "28", "-preset", "medium", "-pix_fmt", "yuv420p"]),
        ("h264_crf33", ["-c:v", "libx264", "-crf", "33", "-preset", "medium", "-pix_fmt", "yuv420p"]),
        ("h265_crf23", ["-c:v", "libx265", "-crf", "23", "-preset", "medium", "-pix_fmt", "yuv420p"]),
        ("h265_crf28", ["-c:v", "libx265", "-crf", "28", "-preset", "medium", "-pix_fmt", "yuv420p"]),
        ("h265_crf33", ["-c:v", "libx265", "-crf", "33", "-preset", "medium", "-pix_fmt", "yuv420p"]),
    ]

    rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        lossless = tmpdir / "watermarked.mkv"
        write_video(wm_np, lossless, codec="ffv1")
        for name, extra in attacks:
            attacked_path = tmpdir / f"{name}.mp4"
            run_ffmpeg(lossless, attacked_path, extra)
            attacked_np = read_video(attacked_path, size=512)
            attacked_t = (
                torch.from_numpy(attacked_np.astype(np.float32) / 255.0)
                .permute(0, 3, 1, 2)
                .to(device)
            )
            with torch.no_grad():
                preds = model.detect(attacked_t, is_video=True)["preds"]
            aggregated = (preds[:, 1:].mean(dim=0) > 0).float()
            ba = float((aggregated == message).float().mean())
            n = min(len(wm_np), len(attacked_np))
            psnr = float(
                np.mean(
                    [
                        psnr_metric(wm_np[i], attacked_np[i], data_range=255)
                        for i in range(n)
                    ]
                )
            )
            ssim = float(
                np.mean(
                    [
                        ssim_metric(
                            wm_np[i], attacked_np[i], channel_axis=2, data_range=255
                        )
                        for i in range(n)
                    ]
                )
            )
            rows.append(
                {
                    "attack": name,
                    "ba": ba,
                    "psnr_vs_watermarked": psnr,
                    "ssim_vs_watermarked": ssim,
                }
            )
            print(
                f"{name:16s} BA={ba:.3f} PSNR={psnr:5.2f} SSIM={ssim:.4f}"
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.out.with_suffix(".csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    md_path = args.out.with_suffix(".md")
    with md_path.open("w") as handle:
        handle.write("# VideoSeal 转码攻击复核（纹理+运动视频）\n\n")
        handle.write(f"- 视频：{args.frames} 帧 512x512，带纹理/相机平移/运动物体\n\n")
        handle.write("| 攻击 | BA | PSNR | SSIM |\n")
        handle.write("| --- | ---: | ---: | ---: |\n")
        for row in rows:
            handle.write(
                f"| {row['attack']} | {row['ba']:.3f} | "
                f"{row['psnr_vs_watermarked']:.2f} | "
                f"{row['ssim_vs_watermarked']:.4f} |\n"
            )
    print(f"\nCSV: {csv_path}\nMarkdown: {md_path}")


if __name__ == "__main__":
    main()
