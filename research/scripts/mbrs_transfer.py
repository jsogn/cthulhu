#!/usr/bin/env python3
"""扩散净化攻击对 MBRS 的迁移验证（研究用途）。

MBRS（ACM MM 2021）是公开权重的深度盲水印方案（256x256、256-bit 载荷、
EC_42.pth）。本脚本复用与 TrustMark 相同的 sd-turbo 扩散净化攻击，检验
攻击是否跨深度水印架构通用。

边界：合成样本、公开预训练模型、本地离线实验，不接触任何平台线上系统。
"""

from __future__ import annotations

import argparse
import csv
import io
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
sys.path.insert(0, str(ROOT / "research" / "vendor" / "MBRS"))

from baseline_trustmark import random_bits, synthetic_image  # noqa: E402
from network.Encoder_MP_Decoder import EncoderDecoder  # noqa: E402


def to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def to_pil(tensor: torch.Tensor) -> Image.Image:
    arr = (
        (tensor.clamp(0.0, 1.0).squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0)
        .round()
        .astype(np.uint8)
    )
    return Image.fromarray(arr)


def bit_accuracy(logits: torch.Tensor, message: torch.Tensor) -> float:
    pred = (logits > 0.5).float()
    return float((pred == message).float().mean().item())


def jpeg_roundtrip(img: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=int, default=4)
    parser.add_argument(
        "--out", type=Path, default=ROOT / "research" / "output" / "mbrs_transfer"
    )
    parser.add_argument("--model", default="stabilityai/sd-turbo")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument(
        "--weights",
        type=Path,
        default=ROOT
        / "research"
        / "vendor"
        / "MBRS"
        / "results"
        / "MBRS_256_256"
        / "EC_42.pth",
    )
    args = parser.parse_args()

    model = EncoderDecoder(256, 256, 256, noise_layers=[])
    state = torch.load(args.weights, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"MBRS weights loaded (missing={len(missing)}, unexpected={len(unexpected)})")
    model.eval()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32
    pipe = AutoPipelineForImage2Image.from_pretrained(
        args.model, torch_dtype=dtype, variant="fp16" if dtype == torch.float16 else None
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    rng = np.random.default_rng(20260909)
    rows: list[dict[str, object]] = []
    for image_id in range(args.images):
        cover_512 = synthetic_image(image_id)
        cover = cover_512.resize((256, 256), Image.BICUBIC)
        message_np = np.array([int(x) for x in random_bits(rng, 256)], dtype=np.float32)
        message = torch.from_numpy(message_np).unsqueeze(0)
        with torch.no_grad():
            encoded = model.encoder(to_tensor(cover), message).clamp(0.0, 1.0)
            clean_logits = model.decoder(encoded)
        watermarked = to_pil(encoded)
        print(
            f"[image {image_id}] clean BA={bit_accuracy(clean_logits, message):.3f} "
            f"PSNR={psnr_metric(np.asarray(cover), np.asarray(watermarked), data_range=255):.2f}"
        )

        cover_arr = np.asarray(cover).astype(np.float64)
        wm_arr = np.asarray(watermarked).astype(np.float64)

        def decode_ba(img: Image.Image) -> float:
            with torch.no_grad():
                logits = model.decoder(to_tensor(img))
            return bit_accuracy(logits, message)

        def resize_only(img: Image.Image) -> Image.Image:
            return img.resize((512, 512), Image.BICUBIC).resize((256, 256), Image.BICUBIC)

        def diffusion(img: Image.Image, strength: float) -> Image.Image:
            upscaled = img.resize((512, 512), Image.BICUBIC)
            generator = torch.Generator(device="cpu").manual_seed(20260909 + image_id)
            out = pipe(
                prompt="",
                image=upscaled,
                strength=strength,
                guidance_scale=0.0,
                num_inference_steps=args.steps,
                generator=generator,
            ).images[0]
            return out.resize((256, 256), Image.BICUBIC)

        def diffusion_x2(img: Image.Image, strength: float) -> Image.Image:
            return diffusion(diffusion(img, strength), strength)

        def add_noise(img: Image.Image, sigma: float) -> Image.Image:
            arr = np.asarray(img).astype(np.float32) / 255.0
            noisy = np.clip(
                arr + rng.normal(0.0, sigma, arr.shape), 0.0, 1.0
            )
            return Image.fromarray((noisy * 255.0).round().astype(np.uint8))

        variants = {
            "resize_only": resize_only,
            "jpeg_50": lambda im: jpeg_roundtrip(im, 50),
            "sd_turbo_s0.15": lambda im: diffusion(im, 0.15),
            "sd_turbo_s0.25": lambda im: diffusion(im, 0.25),
            "sd_turbo_s0.35": lambda im: diffusion(im, 0.35),
            "sd_turbo_s0.45": lambda im: diffusion(im, 0.45),
            "sd_turbo_s0.55": lambda im: diffusion(im, 0.55),
            "sd_turbo_s0.65": lambda im: diffusion(im, 0.65),
            "sd_turbo_s0.45_x2": lambda im: diffusion_x2(im, 0.45),
            "noise0.10_sd_turbo_s0.45": lambda im: diffusion(
                add_noise(im, 0.10), 0.45
            ),
        }

        for name, fn in variants.items():
            attacked = fn(watermarked)
            attacked_arr = np.asarray(attacked).astype(np.float64)
            rows.append(
                {
                    "image_id": image_id,
                    "attack": name,
                    "bit_accuracy": decode_ba(attacked),
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
        handle.write("# 扩散净化攻击 vs MBRS（迁移验证）\n\n")
        handle.write(
            f"- 样本：{args.images} 张合成图；载荷：256-bit；模型：MBRS EC_42\n"
        )
        handle.write(f"- 攻击：`{args.model}` img2img（{device}，{args.steps} steps）\n")
        handle.write("- BA=比特准确率（0.5 为随机猜测）\n\n")
        handle.write("| 攻击 | BA 均值 | BA 标准差 | PSNR vs 原图 | SSIM vs 原图 |\n")
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
