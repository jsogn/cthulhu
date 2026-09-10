#!/usr/bin/env python3
"""扩散净化攻击对 WAM（Watermark Anything）的迁移验证（研究用途）。

WAM 是分割式深度水印（ICLR 2025，公开 MIT 权重 wam_mit.pth），带定位模块
和 32-bit 局部消息，是架构上最接近 DWSF 的公开方案。本脚本用与 TrustMark/
MBRS 相同的 sd-turbo 扩散净化攻击，检验攻击是否威胁分割式水印。

边界：合成样本、公开预训练模型、本地离线实验，不接触任何平台线上系统。
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import sys
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoPipelineForImage2Image
from PIL import Image
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
from watermark_anything.data.metrics import msg_predict_inference  # noqa: E402

from baseline_trustmark import synthetic_image  # noqa: E402


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    arr = (
        unnormalize_img(tensor)
        .clamp(0.0, 1.0)
        .squeeze(0)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    return Image.fromarray((arr * 255.0).round().astype(np.uint8))


def jpeg_roundtrip(img: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=int, default=4)
    parser.add_argument(
        "--out", type=Path, default=ROOT / "research" / "output" / "wam_transfer"
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

    rng = np.random.default_rng(20260909)
    rows: list[dict[str, object]] = []
    for image_id in range(args.images):
        cover = synthetic_image(image_id)
        cover_pt = default_transform(cover).unsqueeze(0).to(device)
        torch.manual_seed(20260909 + image_id)
        message = torch.randint(0, 2, (1, 32)).float().to(device)
        with torch.no_grad():
            watermarked_pt = wam.embed(cover_pt, message)["imgs_w"]
        watermarked = tensor_to_pil(watermarked_pt)

        def decode_ba(img: Image.Image) -> float:
            attacked_pt = default_transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                preds = wam.detect(attacked_pt)["preds"].cpu()
            mask = torch.sigmoid(preds[:, 0, :, :])
            bits = preds[:, 1:, :, :]
            pred = msg_predict_inference(bits, mask)
            return float((pred == message.cpu()).float().mean().item())

        clean_ba = decode_ba(watermarked)
        cover_arr = np.asarray(cover).astype(np.float64)
        wm_arr = np.asarray(watermarked).astype(np.float64)
        print(f"[image {image_id}] clean BA={clean_ba:.3f}")

        def diffusion(img: Image.Image, strength: float) -> Image.Image:
            generator = torch.Generator(device="cpu").manual_seed(20260909 + image_id)
            return pipe(
                prompt="",
                image=img,
                strength=strength,
                guidance_scale=0.0,
                num_inference_steps=args.steps,
                generator=generator,
            ).images[0]

        variants = {
            "identity": lambda im: im,
            "jpeg_50": lambda im: jpeg_roundtrip(im, 50),
            "sd_turbo_s0.15": lambda im: diffusion(im, 0.15),
            "sd_turbo_s0.25": lambda im: diffusion(im, 0.25),
            "sd_turbo_s0.35": lambda im: diffusion(im, 0.35),
            "sd_turbo_s0.45": lambda im: diffusion(im, 0.45),
            "sd_turbo_s0.55": lambda im: diffusion(im, 0.55),
            "sd_turbo_s0.65": lambda im: diffusion(im, 0.65),
        }

        for name, fn in variants.items():
            attacked = fn(watermarked)
            attacked_arr = np.asarray(attacked).astype(np.float64)
            if name == "identity":
                attack_psnr = float("inf")
                attack_ssim = 1.0
            else:
                attack_psnr = psnr_metric(wm_arr, attacked_arr, data_range=255)
                attack_ssim = ssim_metric(
                    wm_arr, attacked_arr, channel_axis=2, data_range=255
                )
            rows.append(
                {
                    "image_id": image_id,
                    "attack": name,
                    "bit_accuracy": decode_ba(attacked),
                    "psnr_vs_watermarked": attack_psnr,
                    "ssim_vs_watermarked": attack_ssim,
                    "psnr_vs_cover": psnr_metric(
                        cover_arr, attacked_arr, data_range=255
                    ),
                    "ssim_vs_cover": ssim_metric(
                        cover_arr, attacked_arr, channel_axis=2, data_range=255
                    ),
                }
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.out.with_suffix(".csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    attacks = list(dict.fromkeys(r["attack"] for r in rows))
    md_path = args.out.with_suffix(".md")
    with md_path.open("w") as handle:
        handle.write("# 扩散净化攻击 vs WAM（迁移验证）\n\n")
        handle.write(
            f"- 样本：{args.images} 张合成图；载荷：32-bit；模型：WAM MIT 权重\n"
        )
        handle.write(f"- 攻击：`{args.model}` img2img（{device}，{args.steps} steps）\n")
        handle.write("- BA=比特准确率（0.5 为随机猜测）\n\n")
        handle.write(
            "| 攻击 | BA 均值 | BA 标准差 | PSNR vs 原图 | SSIM vs 原图 |\n"
        )
        handle.write("| --- | ---: | ---: | ---: | ---: |\n")
        for attack in attacks:
            subset = [r for r in rows if r["attack"] == attack]
            handle.write(
                f"| {attack} | "
                f"{np.mean([r['bit_accuracy'] for r in subset]):.3f} | "
                f"{np.std([r['bit_accuracy'] for r in subset]):.3f} | "
                f"{np.mean([r['psnr_vs_cover'] for r in subset]):.2f} | "
                f"{np.mean([r['ssim_vs_cover'] for r in subset]):.4f} |\n"
            )

    print(f"\nCSV: {csv_path}\nMarkdown: {md_path}")
    for attack in attacks:
        subset = [r for r in rows if r["attack"] == attack]
        print(
            f"{attack:20s} BA={np.mean([r['bit_accuracy'] for r in subset]):.3f} "
            f"PSNR={np.mean([r['psnr_vs_cover'] for r in subset]):5.2f} "
            f"SSIM={np.mean([r['ssim_vs_cover'] for r in subset]):.4f}"
        )


if __name__ == "__main__":
    main()
