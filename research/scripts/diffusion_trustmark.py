#!/usr/bin/env python3
"""扩散净化攻击对深度盲水印的基线（研究用途）。

使用 sd-turbo 的 img2img 在低 strength 下重建图像，模拟文献中的
"regeneration / purification" 攻击；并测试先加高斯噪声再扩散重建的组合。

边界：合成样本、公开预训练模型、本地离线实验，不接触任何平台线上系统。
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoPipelineForImage2Image
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "scripts"))
sys.path.insert(0, str(ROOT / "backend" / "src"))

from baseline_trustmark import bit_accuracy, random_bits, synthetic_image  # noqa: E402
from trustmark import TrustMark  # noqa: E402


def add_gaussian_noise(img: Image.Image, sigma: float, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = np.asarray(img).astype(np.float32) / 255.0
    noisy = np.clip(arr + rng.normal(0.0, sigma, arr.shape), 0.0, 1.0)
    return Image.fromarray((noisy * 255.0).round().astype(np.uint8))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=int, default=4)
    parser.add_argument(
        "--out", type=Path, default=ROOT / "research" / "output" / "diffusion_trustmark"
    )
    parser.add_argument("--model", default="stabilityai/sd-turbo")
    parser.add_argument("--steps", type=int, default=8)
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32
    pipe = AutoPipelineForImage2Image.from_pretrained(
        args.model, torch_dtype=dtype, variant="fp16" if dtype == torch.float16 else None
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    tm = TrustMark(use_ECC=False, verbose=True, model_type="Q", loadRemover=False)
    rng = np.random.default_rng(20260909)

    variants: list[tuple[str, float, float]] = []
    for strength in (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.45, 0.60):
        variants.append((f"sd_turbo_s{strength:.2f}", 0.0, strength))
    for noise in (0.05, 0.10, 0.15):
        variants.append((f"noise{noise:.2f}_sd_turbo_s0.35", noise, 0.35))

    rows: list[dict[str, object]] = []
    for image_id in range(args.images):
        cover = synthetic_image(image_id)
        payload = random_bits(rng)
        watermarked = tm.encode(cover, payload, MODE="binary")
        clean_pred, _, _ = tm.decode(watermarked, MODE="binary")
        print(f"[image {image_id}] clean BA={bit_accuracy(clean_pred, payload):.3f}")

        cover_arr = np.asarray(cover).astype(np.float64)
        wm_arr = np.asarray(watermarked).astype(np.float64)
        for name, noise, strength in variants:
            base = (
                add_gaussian_noise(watermarked, noise, seed=1000 + image_id)
                if noise > 0
                else watermarked
            )
            generator = torch.Generator(device="cpu").manual_seed(20260909 + image_id)
            attacked = pipe(
                prompt="",
                image=base,
                strength=strength,
                guidance_scale=0.0,
                num_inference_steps=args.steps,
                generator=generator,
            ).images[0]
            pred, _, _ = tm.decode(attacked, MODE="binary")
            attacked_arr = np.asarray(attacked).astype(np.float64)
            rows.append(
                {
                    "image_id": image_id,
                    "attack": name,
                    "bit_accuracy": bit_accuracy(pred, payload),
                    "psnr_vs_watermarked": psnr_metric(
                        wm_arr, attacked_arr, data_range=255
                    ),
                    "ssim_vs_watermarked": ssim_metric(
                        wm_arr, attacked_arr, channel_axis=2, data_range=255
                    ),
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
        handle.write("# 扩散净化攻击 vs TrustMark Q（研究基线）\n\n")
        handle.write(
            f"- 样本：{args.images} 张合成图；载荷：100-bit 原始二进制（关闭 ECC）\n"
        )
        handle.write(
            f"- 模型：`{args.model}`（{device}，{args.steps} steps，guidance=0）\n"
        )
        handle.write("- BA=比特准确率（0.5 为随机猜测）\n\n")
        handle.write(
            "| 攻击 | BA 均值 | BA 标准差 | PSNR vs 水印图 | SSIM vs 水印图 | "
            "PSNR vs 原图 | SSIM vs 原图 |\n"
        )
        handle.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for attack in attacks:
            subset = [r for r in rows if r["attack"] == attack]
            handle.write(
                f"| {attack} | "
                f"{np.mean([r['bit_accuracy'] for r in subset]):.3f} | "
                f"{np.std([r['bit_accuracy'] for r in subset]):.3f} | "
                f"{np.mean([r['psnr_vs_watermarked'] for r in subset]):.2f} | "
                f"{np.mean([r['ssim_vs_watermarked'] for r in subset]):.4f} | "
                f"{np.mean([r['psnr_vs_cover'] for r in subset]):.2f} | "
                f"{np.mean([r['ssim_vs_cover'] for r in subset]):.4f} |\n"
            )

    print(f"\nCSV: {csv_path}\nMarkdown: {md_path}")
    for attack in attacks:
        subset = [r for r in rows if r["attack"] == attack]
        print(
            f"{attack:28s} BA={np.mean([r['bit_accuracy'] for r in subset]):.3f} "
            f"PSNR_wm={np.mean([r['psnr_vs_watermarked'] for r in subset]):5.2f} "
            f"SSIM_wm={np.mean([r['ssim_vs_watermarked'] for r in subset]):.4f} "
            f"PSNR_cover={np.mean([r['psnr_vs_cover'] for r in subset]):5.2f}"
        )


if __name__ == "__main__":
    main()
