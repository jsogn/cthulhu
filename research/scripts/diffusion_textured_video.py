#!/usr/bin/env python3
"""扩散净化在有纹理/运动视频上的复核（研究用途）。

用于验证扩散净化对 VideoSeal 的攻击是否受内容复杂度影响。视频使用
纹理背景 + 相机平移 + 运动物体，逐帧净化后做时序聚合检测。

边界：合成样本、公开预训练模型、本地离线实验，不接触任何平台线上系统。
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

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

from video_codec_textured import make_textured_video  # noqa: E402


def to_pil(tensor: torch.Tensor) -> Image.Image:
    return transforms.ToPILImage()(tensor.clamp(0.0, 1.0).cpu())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "research" / "output" / "diffusion_textured_video",
    )
    parser.add_argument("--model", default="stabilityai/sd-turbo")
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = videoseal.load("videoseal").to(device).eval()
    dtype = torch.float16 if device == "mps" else torch.float32
    pipe = AutoPipelineForImage2Image.from_pretrained(
        args.model, torch_dtype=dtype, variant="fp16" if dtype == torch.float16 else None
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

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

    def diffuse(img: Image.Image, strength: float, seed: int) -> Image.Image:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        return pipe(
            prompt="",
            image=img,
            strength=strength,
            guidance_scale=0.0,
            num_inference_steps=args.steps,
            generator=generator,
        ).images[0]

    rows: list[dict[str, object]] = []
    for name, strength, same_seed in [
        ("identity", 0.0, False),
        ("per_frame_s0.15", 0.15, False),
        ("per_frame_s0.25", 0.25, False),
        ("per_frame_s0.35", 0.35, False),
        ("correlated_s0.25", 0.25, True),
    ]:
        if strength == 0:
            attacked_np = wm_np.copy()
        else:
            frames = []
            for i in range(len(watermarked)):
                seed = 20260909 if same_seed else 20260909 + i
                frames.append(to_pil(watermarked[i]))
                frames[-1] = diffuse(frames[-1], strength, seed)
            attacked_np = (
                torch.stack([transforms.ToTensor()(f).to(device) for f in frames])
                .clamp(0, 1)
                .permute(0, 2, 3, 1)
                .cpu()
                .numpy()
                * 255
            ).round().astype(np.uint8)

        attacked_t = (
            torch.from_numpy(attacked_np.astype(np.float32) / 255.0)
            .permute(0, 3, 1, 2)
            .to(device)
        )
        with torch.no_grad():
            preds = model.detect(attacked_t, is_video=True)["preds"]
        aggregated = (preds[:, 1:].mean(dim=0) > 0).float()
        per_frame = (preds[:, 1:] > 0).float()
        ba = float((aggregated == message).float().mean())
        per_frame_ba = float((per_frame == message).float().mean())
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
                "per_frame_ba": per_frame_ba,
                "psnr_vs_watermarked": psnr,
                "ssim_vs_watermarked": ssim,
            }
        )
        print(
            f"{name:20s} BA={ba:.3f} per_frame={per_frame_ba:.3f} "
            f"PSNR={psnr:5.2f} SSIM={ssim:.4f}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.out.with_suffix(".csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    md_path = args.out.with_suffix(".md")
    with md_path.open("w") as handle:
        handle.write("# 扩散净化 vs VideoSeal（纹理+运动视频复核）\n\n")
        handle.write(f"- 视频：{args.frames} 帧 512x512，带纹理/相机平移/运动物体\n")
        handle.write(f"- 攻击：`{args.model}` img2img，{args.steps} steps\n\n")
        handle.write("| 攻击 | 聚合 BA | 逐帧 BA | PSNR | SSIM |\n")
        handle.write("| --- | ---: | ---: | ---: | ---: |\n")
        for row in rows:
            handle.write(
                f"| {row['attack']} | {row['ba']:.3f} | {row['per_frame_ba']:.3f} | "
                f"{row['psnr_vs_watermarked']:.2f} | "
                f"{row['ssim_vs_watermarked']:.4f} |\n"
            )
    print(f"\nCSV: {csv_path}\nMarkdown: {md_path}")


if __name__ == "__main__":
    main()
