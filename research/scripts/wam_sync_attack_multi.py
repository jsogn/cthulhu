#!/usr/bin/env python3
"""WAM 同步/定位模块定向攻击：多图 × 多 ε（研究用途）。

用 sd-turbo 生成自然图像，逐张测试：
- mask 定向 PGD（打同步/定位）
- message 定向 PGD（打消息，双尾）
- 扩散净化（sd-turbo）
输出每张图的 mask IoU / 消息 BA / PSNR / SSIM，并聚合均值与标准差。

边界：生成图像、公开预训练模型、本地离线实验，不接触任何平台线上系统。
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoPipelineForImage2Image, AutoPipelineForText2Image
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric

ROOT = Path(__file__).resolve().parents[2]
WAM_REPO = ROOT / "research" / "vendor" / "watermark-anything"
sys.path.insert(0, str(ROOT / "research" / "scripts"))
sys.path.insert(0, str(WAM_REPO))
os.chdir(WAM_REPO)

from notebooks.inference_utils import (  # noqa: E402
    create_random_mask,
    default_transform,
    load_model_from_checkpoint,
    unnormalize_img,
)
from watermark_anything.data.transforms import normalize_img  # noqa: E402

from wam_sync_attack import evaluate, to_pil  # noqa: E402


PROMPTS = [
    "a natural landscape photo with mountains and a lake",
    "a city street photo in the daytime",
    "a portrait photo of a person in natural light",
    "a close-up photo of food on a table",
    "a photo of an animal in a forest",
    "a photo of a modern indoor room",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=int, default=6)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--epsilons", nargs="+", type=float, default=[4 / 255, 8 / 255])
    parser.add_argument(
        "--modes", nargs="+", default=["mask", "message"]
    )
    parser.add_argument("--purify-strengths", nargs="+", type=float, default=[0.15])
    parser.add_argument("--model", default="stabilityai/sd-turbo")
    parser.add_argument("--diffusion-steps", type=int, default=4)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "research" / "output" / "wam_sync_attack_multi",
    )
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32
    txt2img = AutoPipelineForText2Image.from_pretrained(
        args.model, torch_dtype=dtype, variant="fp16" if dtype == torch.float16 else None
    ).to(device)
    txt2img.set_progress_bar_config(disable=True)
    img2img = AutoPipelineForImage2Image.from_pretrained(
        args.model, torch_dtype=dtype, variant="fp16" if dtype == torch.float16 else None
    ).to(device)
    img2img.set_progress_bar_config(disable=True)
    wam = (
        load_model_from_checkpoint(
            str(WAM_REPO / "checkpoints" / "params.json"),
            str(WAM_REPO / "checkpoints" / "wam_mit.pth"),
        )
        .to(device)
        .eval()
    )

    rows: list[dict[str, object]] = []
    for image_id in range(args.images):
        prompt = PROMPTS[image_id % len(PROMPTS)]
        generator = torch.Generator(device="cpu").manual_seed(20260909 + image_id)
        image = txt2img(
            prompt=prompt,
            num_inference_steps=args.diffusion_steps,
            guidance_scale=0.0,
            generator=generator,
            height=512,
            width=512,
        ).images[0]
        img_pt = default_transform(image).unsqueeze(0).to(device)
        torch.manual_seed(20260909 + image_id)
        message = torch.randint(0, 2, (1, 32)).float().to(device)
        with torch.no_grad():
            embedded = wam.embed(img_pt, message)["imgs_w"]
        mask = create_random_mask(
            img_pt, num_masks=1, mask_percentage=0.5
        ).to(device)
        watermarked = embedded * mask + img_pt * (1 - mask)
        true_mask_256 = F.interpolate(
            mask, size=(256, 256), mode="nearest"
        ).clamp(0, 1)
        base_pixel = unnormalize_img(watermarked).clamp(0, 1)

        clean_iou, clean_ba = evaluate(wam, watermarked, true_mask_256, message)
        rows.append(
            {
                "image_id": image_id,
                "prompt": prompt,
                "attack": "clean",
                "epsilon": 0.0,
                "mask_iou": clean_iou,
                "message_ba": clean_ba,
                "psnr": float("inf"),
                "ssim": 1.0,
            }
        )
        print(f"[{image_id}] clean IoU={clean_iou:.3f} BA={clean_ba:.3f}")

        def record(name: str, eps: float, adv_pixel: torch.Tensor) -> None:
            adv_pixel = adv_pixel.detach().clamp(0, 1)
            adv_pt = normalize_img(adv_pixel)
            iou, ba = evaluate(wam, adv_pt, true_mask_256, message)
            adv_np = (
                (adv_pixel.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255)
                .round()
                .astype(np.uint8)
            )
            base_np = (
                (base_pixel.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255)
                .round()
                .astype(np.uint8)
            )
            rows.append(
                {
                    "image_id": image_id,
                    "prompt": prompt,
                    "attack": name,
                    "epsilon": eps,
                    "mask_iou": iou,
                    "message_ba": ba,
                    "psnr": psnr_metric(base_np, adv_np, data_range=255),
                    "ssim": ssim_metric(
                        base_np, adv_np, channel_axis=2, data_range=255
                    ),
                }
            )
            print(
                f"  {name:16s} eps={eps*255:.0f}/255 IoU={iou:.3f} BA={ba:.3f} "
                f"PSNR={rows[-1]['psnr']:.2f} SSIM={rows[-1]['ssim']:.4f}"
            )

        for eps in args.epsilons:
            for mode in args.modes:
                delta = torch.zeros_like(base_pixel, requires_grad=True)
                alpha = eps / 5.0
                for _ in range(args.steps):
                    adv_pixel = (base_pixel + delta).clamp(0, 1)
                    adv_pt = normalize_img(adv_pixel)
                    preds = wam.detect(adv_pt)["preds"]
                    mask_logits = preds[:, 0:1]
                    bit_logits = preds[:, 1:]
                    mask_loss = F.binary_cross_entropy_with_logits(
                        mask_logits, true_mask_256
                    )
                    msg_loss = bit_logits.pow(2).mean()
                    if mode == "mask":
                        loss = mask_loss
                        direction = 1.0
                    else:
                        loss = msg_loss
                        direction = -1.0
                    grad = torch.autograd.grad(loss, delta)[0]
                    delta = (
                        delta + direction * alpha * grad.sign()
                    ).detach().clamp(-eps, eps)
                    delta.requires_grad_(True)
                record(f"pgd_{mode}", eps, base_pixel + delta)

        base_pil = to_pil(watermarked)
        for strength in args.purify_strengths:
            generator = torch.Generator(device="cpu").manual_seed(20260909 + image_id)
            purified = img2img(
                prompt="",
                image=base_pil,
                strength=strength,
                guidance_scale=0.0,
                num_inference_steps=50,
                generator=generator,
            ).images[0]
            purified_pt = default_transform(purified).unsqueeze(0).to(device)
            iou, ba = evaluate(wam, purified_pt, true_mask_256, message)
            pur_np = np.asarray(purified).astype(np.float64)
            base_np = np.asarray(base_pil).astype(np.float64)
            rows.append(
                {
                    "image_id": image_id,
                    "prompt": prompt,
                    "attack": f"purify_s{strength:.2f}",
                    "epsilon": 0.0,
                    "mask_iou": iou,
                    "message_ba": ba,
                    "psnr": psnr_metric(base_np, pur_np, data_range=255),
                    "ssim": ssim_metric(
                        base_np, pur_np, channel_axis=2, data_range=255
                    ),
                }
            )
            print(
                f"  purify_s{strength:.2f}   IoU={iou:.3f} BA={ba:.3f} "
                f"PSNR={rows[-1]['psnr']:.2f} SSIM={rows[-1]['ssim']:.4f}"
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.out.with_suffix(".csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    groups: dict[tuple[str, float], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["attack"]), float(row["epsilon"]))
        groups.setdefault(key, []).append(row)
    md_path = args.out.with_suffix(".md")
    with md_path.open("w") as handle:
        handle.write("# WAM 同步/定位模块定向攻击：多图 × 多 ε\n\n")
        handle.write(
            f"- 图像：{args.images} 张 sd-turbo 生成的自然图（512x512）；"
            f"局部水印 mask 50%；PGD {args.steps} 步\n\n"
        )
        handle.write(
            "| 攻击 | ε (/255) | mask IoU | 消息 BA | PSNR | SSIM |\n"
        )
        handle.write("| --- | ---: | ---: | ---: | ---: | ---: |\n")
        for (attack, eps), subset in groups.items():
            handle.write(
                f"| {attack} | {eps*255:.0f} | "
                f"{np.mean([r['mask_iou'] for r in subset]):.3f} ± "
                f"{np.std([r['mask_iou'] for r in subset]):.3f} | "
                f"{np.mean([r['message_ba'] for r in subset]):.3f} ± "
                f"{np.std([r['message_ba'] for r in subset]):.3f} | "
                f"{np.mean([r['psnr'] for r in subset]):.2f} | "
                f"{np.mean([r['ssim'] for r in subset]):.4f} |\n"
            )
    print(f"\nCSV: {csv_path}\nMarkdown: {md_path}")


if __name__ == "__main__":
    main()
