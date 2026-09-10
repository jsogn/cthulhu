#!/usr/bin/env python3
"""MBRS 加固实验：嵌入强度 + 多副本冗余（研究用途）。

在扩散净化攻击（sd-turbo）下比较：
- 提高 MBRS 嵌入强度 strength_factor（1.0 / 1.5 / 2.0）
- 同一消息嵌入多张图，解码时对逐比特概率求平均（模拟视频时序冗余）

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=int, default=4)
    parser.add_argument("--attack-strength", type=float, default=0.65)
    parser.add_argument(
        "--out", type=Path, default=ROOT / "research" / "output" / "hardening_mbrs"
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
    model.load_state_dict(state, strict=False)
    model.eval()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32
    pipe = AutoPipelineForImage2Image.from_pretrained(
        args.model, torch_dtype=dtype, variant="fp16" if dtype == torch.float16 else None
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    rng = np.random.default_rng(20260909)
    message_np = np.array([int(x) for x in random_bits(rng, 256)], dtype=np.float32)
    message = torch.from_numpy(message_np).unsqueeze(0)
    rows: list[dict[str, object]] = []

    for factor in (1.0, 1.5, 2.0):
        per_image_probs: list[np.ndarray] = []
        for image_id in range(args.images):
            cover = synthetic_image(image_id).resize((256, 256), Image.BICUBIC)
            cover_tensor = to_tensor(cover)
            with torch.no_grad():
                encoded = model.encoder(cover_tensor, message)
            encoded = (cover_tensor + (encoded - cover_tensor) * factor).clamp(0.0, 1.0)
            watermarked = to_pil(encoded)

            upscaled = watermarked.resize((512, 512), Image.BICUBIC)
            generator = torch.Generator(device="cpu").manual_seed(20260909 + image_id)
            attacked = pipe(
                prompt="",
                image=upscaled,
                strength=args.attack_strength,
                guidance_scale=0.0,
                num_inference_steps=args.steps,
                generator=generator,
            ).images[0].resize((256, 256), Image.BICUBIC)

            with torch.no_grad():
                logits = model.decoder(to_tensor(attacked))
            prob = torch.sigmoid(logits).numpy()[0]
            per_image_probs.append(prob)

            cover_arr = np.asarray(cover).astype(np.float64)
            att_arr = np.asarray(attacked).astype(np.float64)
            rows.append(
                {
                    "strength_factor": factor,
                    "image_id": image_id,
                    "bit_accuracy": float(((prob > 0.5) == message_np).mean()),
                    "psnr_vs_cover": psnr_metric(cover_arr, att_arr, data_range=255),
                    "ssim_vs_cover": ssim_metric(
                        cover_arr, att_arr, channel_axis=2, data_range=255
                    ),
                }
            )

        avg_prob = np.stack(per_image_probs, axis=0).mean(axis=0)
        avg_ba = float(((avg_prob > 0.5) == message_np).mean())
        subset = [r for r in rows if r["strength_factor"] == factor]
        for row in subset:
            row["multi_copy_ba"] = avg_ba
        print(
            f"strength_factor={factor:.1f} single_BA="
            f"{np.mean([r['bit_accuracy'] for r in subset]):.3f} "
            f"multi_BA={avg_ba:.3f} "
            f"PSNR={np.mean([r['psnr_vs_cover'] for r in subset]):.2f} "
            f"SSIM={np.mean([r['ssim_vs_cover'] for r in subset]):.4f}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.out.with_suffix(".csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md_path = args.out.with_suffix(".md")
    with md_path.open("w") as handle:
        handle.write("# MBRS 加固实验：嵌入强度 vs 多副本冗余\n\n")
        handle.write(
            f"- 攻击：sd-turbo img2img strength={args.attack_strength}，"
            f"{args.steps} steps\n"
        )
        handle.write(f"- 样本：{args.images} 张合成图；消息：固定 256-bit\n\n")
        handle.write(
            "| strength_factor | single BA | multi-copy BA | PSNR vs 原图 | SSIM vs 原图 |\n"
        )
        handle.write("| ---: | ---: | ---: | ---: | ---: |\n")
        for factor in (1.0, 1.5, 2.0):
            subset = [r for r in rows if r["strength_factor"] == factor]
            handle.write(
                f"| {factor:.1f} | "
                f"{np.mean([r['bit_accuracy'] for r in subset]):.3f} | "
                f"{subset[0]['multi_copy_ba']:.3f} | "
                f"{np.mean([r['psnr_vs_cover'] for r in subset]):.2f} | "
                f"{np.mean([r['ssim_vs_cover'] for r in subset]):.4f} |\n"
            )

    print(f"\nCSV: {csv_path}\nMarkdown: {md_path}")


if __name__ == "__main__":
    main()
