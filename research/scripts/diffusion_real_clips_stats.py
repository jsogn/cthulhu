#!/usr/bin/env python3
"""多段真实视频上的扩散净化统计评估（研究用途）。

从自有/授权样本视频中提取 N 段短片段，逐段嵌入 VideoSeal 水印并做逐帧
扩散净化，统计 BA、PSNR、SSIM、LPIPS、帧间一致性与 FNR。

边界：自有/授权素材、公开预训练模型、本地离线实验，不接触任何平台线上系统。
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path

import lpips
import numpy as np
import torch
from diffusers import AutoPipelineForImage2Image
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[2]
VSEAL_REPO = ROOT / "research" / "vendor" / "videoseal"
sys.path.insert(0, str(ROOT / "research" / "scripts"))
sys.path.insert(0, str(VSEAL_REPO))
os.chdir(VSEAL_REPO)

import videoseal  # noqa: E402

from eot_pgd_video import aggregate_ba, detector_logits  # noqa: E402
from video_codec_attacks import read_video  # noqa: E402

FFMPEG = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"


def extract_clip(source: Path, out_path: Path, start: float, frames: int) -> None:
    duration = frames / 30.0 + 0.2
    cmd = [
        FFMPEG, "-y",
        "-ss", str(start),
        "-i", str(source),
        "-t", str(duration),
        "-vf",
        "scale=512:512:force_original_aspect_ratio=decrease,"
        "pad=512:512:(ow-iw)/2:(oh-ih)/2",
        "-r", "30", "-an",
        "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips", type=int, default=8)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--strength", type=float, default=0.15)
    parser.add_argument("--diffusion-steps", type=int, default=20)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=ROOT / "backend" / "data" / "library",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "research" / "output" / "diffusion_real_clips_stats",
    )
    parser.add_argument("--model", default="stabilityai/sd-turbo")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32
    pipe = AutoPipelineForImage2Image.from_pretrained(
        args.model, torch_dtype=dtype, variant="fp16" if dtype == torch.float16 else None
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    model = videoseal.load("videoseal").to(device).eval()
    lpips_net = lpips.LPIPS(net="alex").to(device).eval()

    clip_dir = ROOT / "research" / "data" / "real_clips"
    clip_dir.mkdir(parents=True, exist_ok=True)
    sources = sorted(glob.glob(str(args.source_dir / "*.mp4")))
    if not sources:
        raise SystemExit(f"no source videos in {args.source_dir}")
    clips = []
    for i in range(args.clips):
        source = Path(sources[i % len(sources)])
        start = 5.0 + (i // len(sources)) * 30.0
        clip_path = clip_dir / f"clip_{i:02d}.mp4"
        extract_clip(source, clip_path, start, args.frames)
        clips.append((source.name, start, clip_path))

    rows: list[dict[str, object]] = []
    for i, (source_name, start, clip_path) in enumerate(clips):
        frames_np = read_video(clip_path, size=512)[: args.frames]
        cover = (
            torch.from_numpy(frames_np.astype(np.float32) / 255.0)
            .permute(0, 3, 1, 2)
            .to(device)
        )
        with torch.no_grad():
            out = model.embed(cover, is_video=True)
        message = out["msgs"][0:1].float().to(device)
        watermarked = out["imgs_w"].detach()
        wm_np = (
            (watermarked.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255)
            .round()
            .astype(np.uint8)
        )
        purified = []
        for t in range(len(watermarked)):
            pil = transforms.ToPILImage()(watermarked[t].clamp(0, 1).cpu())
            generator = torch.Generator(device="cpu").manual_seed(20260909 + i * 100 + t)
            attacked = pipe(
                prompt="",
                image=pil,
                strength=args.strength,
                guidance_scale=0.0,
                num_inference_steps=args.diffusion_steps,
                generator=generator,
            ).images[0]
            purified.append(transforms.ToTensor()(attacked))
        purified_t = torch.stack(purified).to(device)
        with torch.no_grad():
            logits = detector_logits(model, purified_t)
        ba, per_frame_ba = aggregate_ba(logits, message)

        attacked_np = (
            (purified_t.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255)
            .round()
            .astype(np.uint8)
        )
        n = min(len(wm_np), len(attacked_np))
        per_frame_psnr = [
            psnr_metric(wm_np[t], attacked_np[t], data_range=255) for t in range(n)
        ]
        psnr = float(np.mean(per_frame_psnr))
        ssim = float(
            np.mean(
                [
                    ssim_metric(
                        wm_np[t], attacked_np[t], channel_axis=2, data_range=255
                    )
                    for t in range(n)
                ]
            )
        )
        with torch.no_grad():
            lpips_values = []
            for t in range(n):
                a = watermarked[t : t + 1] * 2 - 1
                b = purified_t[t : t + 1] * 2 - 1
                lpips_values.append(float(lpips_net(a, b).mean()))
        rows.append(
            {
                "clip": i,
                "source": source_name,
                "start": start,
                "ba": ba,
                "per_frame_ba": per_frame_ba,
                "psnr": psnr,
                "ssim": ssim,
                "lpips": float(np.mean(lpips_values)),
                "frame_psnr_std": float(np.std(per_frame_psnr)),
            }
        )
        print(
            f"[{i}] {source_name} @{start:.0f}s BA={ba:.3f} "
            f"PSNR={psnr:5.2f} SSIM={ssim:.4f} LPIPS={rows[-1]['lpips']:.4f} "
            f"frame_std={rows[-1]['frame_psnr_std']:.2f}"
        )

    bas = [r["ba"] for r in rows]
    fnr = float(np.mean([1.0 if r["ba"] < 0.6 else 0.0 for r in rows]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.out.with_suffix(".csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    md_path = args.out.with_suffix(".md")
    with md_path.open("w") as handle:
        handle.write("# 多段真实视频：扩散净化统计评估\n\n")
        handle.write(
            f"- 片段：{args.clips} 段 × {args.frames} 帧 512x512；"
            f"净化 strength={args.strength}，{args.diffusion_steps} steps\n"
        )
        handle.write(
            f"- 聚合：BA {np.mean(bas):.3f} ± {np.std(bas):.3f}；"
            f"PSNR {np.mean([r['psnr'] for r in rows]):.2f}；"
            f"SSIM {np.mean([r['ssim'] for r in rows]):.4f}；"
            f"LPIPS {np.mean([r['lpips'] for r in rows]):.4f}；"
            f"帧间 PSNR 标准差 {np.mean([r['frame_psnr_std'] for r in rows]):.2f}；"
            f"FNR(BA<0.6) {fnr:.2f}\n\n"
        )
        handle.write(
            "| clip | 来源 | 起始 | BA | PSNR | SSIM | LPIPS | 帧间 PSNR std |\n"
        )
        handle.write("| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for r in rows:
            handle.write(
                f"| {r['clip']} | {r['source']} | {r['start']:.0f}s | "
                f"{r['ba']:.3f} | {r['psnr']:.2f} | {r['ssim']:.4f} | "
                f"{r['lpips']:.4f} | {r['frame_psnr_std']:.2f} |\n"
            )
    print(f"\nAggregate BA {np.mean(bas):.3f} ± {np.std(bas):.3f}; FNR {fnr:.2f}")
    print(f"CSV: {csv_path}\nMarkdown: {md_path}")


if __name__ == "__main__":
    main()
