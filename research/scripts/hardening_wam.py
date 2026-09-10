#!/usr/bin/env python3
"""WAM 加固实验：嵌入强度 + 多副本冗余（研究用途）。

在扩散净化攻击（sd-turbo strength 0.15）下比较：
- 提高 WAM 嵌入强度 scaling_w（2 / 4 / 8）
- 同一消息嵌入多张图，解码时对逐比特概率求平均（模拟视频时序冗余）

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
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric

ROOT = Path(__file__).resolve().parents[2]
WAM_REPO = ROOT / "research" / "vendor" / "watermark-anything"
sys.path.insert(0, str(ROOT / "research" / "scripts"))
sys.path.insert(0, str(WAM_REPO))
os.chdir(WAM_REPO)

from notebooks.inference_utils import (  # noqa: E402
    default_transform,
    load_model_from_checkpoint,
    unnormalize_img,
)

from baseline_trustmark import synthetic_image  # noqa: E402


def tensor_to_pil(tensor: torch.Tensor):
    from PIL import Image

    arr = (
        unnormalize_img(tensor)
        .clamp(0.0, 1.0)
        .squeeze(0)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    return Image.fromarray((arr * 255.0).round().astype(np.uint8))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=int, default=4)
    parser.add_argument("--attack-strength", type=float, default=0.15)
    parser.add_argument(
        "--out", type=Path, default=ROOT / "research" / "output" / "hardening_wam"
    )
    parser.add_argument("--model", default="stabilityai/sd-turbo")
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    wam = (
        load_model_from_checkpoint(
            str(WAM_REPO / "checkpoints" / "params.json"),
            str(WAM_REPO / "checkpoints" / "wam_mit.pth"),
        )
        .to(device)
        .eval()
    )

    dtype = torch.float16 if device == "mps" else torch.float32
    pipe = AutoPipelineForImage2Image.from_pretrained(
        args.model, torch_dtype=dtype, variant="fp16" if dtype == torch.float16 else None
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    torch.manual_seed(20260909)
    message = torch.randint(0, 2, (1, 32)).float().to(device)
    rows: list[dict[str, object]] = []

    for scaling_w in (2.0, 4.0, 8.0):
        wam.scaling_w = scaling_w
        per_image_probs: list[np.ndarray] = []
        for image_id in range(args.images):
            cover = synthetic_image(image_id)
            cover_pt = default_transform(cover).unsqueeze(0).to(device)
            with torch.no_grad():
                watermarked_pt = wam.embed(cover_pt, message)["imgs_w"]
            watermarked = tensor_to_pil(watermarked_pt)

            generator = torch.Generator(device="cpu").manual_seed(20260909 + image_id)
            attacked = pipe(
                prompt="",
                image=watermarked,
                strength=args.attack_strength,
                guidance_scale=0.0,
                num_inference_steps=args.steps,
                generator=generator,
            ).images[0]

            attacked_pt = default_transform(attacked).unsqueeze(0).to(device)
            with torch.no_grad():
                preds = wam.detect(attacked_pt)["preds"].cpu()
            mask = torch.sigmoid(preds[:, 0, :, :])
            bits = preds[:, 1:, :, :]
            # soft 加权平均：用定位 mask 作为逐像素权重
            weight = mask.reshape(1, -1)
            prob = (bits.reshape(1, 32, -1) * weight.unsqueeze(1)).sum(-1) / (
                weight.sum(-1, keepdim=True) + 1e-6
            )
            per_image_probs.append(prob.numpy()[0])

            cover_arr = np.asarray(cover).astype(np.float64)
            wm_arr = np.asarray(watermarked).astype(np.float64)
            att_arr = np.asarray(attacked).astype(np.float64)
            rows.append(
                {
                    "scaling_w": scaling_w,
                    "image_id": image_id,
                    "bit_accuracy": float(
                        ((prob > 0.5).float().numpy()[0] == message.cpu().numpy()[0]).mean()
                    ),
                    "psnr_vs_cover": psnr_metric(cover_arr, att_arr, data_range=255),
                    "ssim_vs_cover": ssim_metric(
                        cover_arr, att_arr, channel_axis=2, data_range=255
                    ),
                }
            )

        stacked = np.stack(per_image_probs, axis=0)  # (images, 32)
        avg_prob = stacked.mean(axis=0)
        avg_ba = float(
            ((avg_prob > 0.5) == message.cpu().numpy()[0]).mean()
        )
        subset = [r for r in rows if r["scaling_w"] == scaling_w]
        print(
            f"scaling_w={scaling_w:.1f} single_BA="
            f"{np.mean([r['bit_accuracy'] for r in subset]):.3f} "
            f"multi_BA={avg_ba:.3f} "
            f"PSNR={np.mean([r['psnr_vs_cover'] for r in subset]):.2f} "
            f"SSIM={np.mean([r['ssim_vs_cover'] for r in subset]):.4f}"
        )
        for row in subset:
            row["multi_copy_ba"] = avg_ba

    args.out.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.out.with_suffix(".csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md_path = args.out.with_suffix(".md")
    with md_path.open("w") as handle:
        handle.write("# WAM 加固实验：嵌入强度 vs 多副本冗余\n\n")
        handle.write(
            f"- 攻击：sd-turbo img2img strength={args.attack_strength}，"
            f"{args.steps} steps\n"
        )
        handle.write(f"- 样本：{args.images} 张合成图；消息：固定 32-bit\n")
        handle.write("- single=单副本 BA；multi=多副本逐比特概率平均后的 BA\n\n")
        handle.write(
            "| scaling_w | single BA | multi-copy BA | PSNR vs 原图 | SSIM vs 原图 |\n"
        )
        handle.write("| ---: | ---: | ---: | ---: | ---: |\n")
        for scaling_w in (2.0, 4.0, 8.0):
            subset = [r for r in rows if r["scaling_w"] == scaling_w]
            handle.write(
                f"| {scaling_w:.1f} | "
                f"{np.mean([r['bit_accuracy'] for r in subset]):.3f} | "
                f"{subset[0]['multi_copy_ba']:.3f} | "
                f"{np.mean([r['psnr_vs_cover'] for r in subset]):.2f} | "
                f"{np.mean([r['ssim_vs_cover'] for r in subset]):.4f} |\n"
            )

    print(f"\nCSV: {csv_path}\nMarkdown: {md_path}")


if __name__ == "__main__":
    main()
