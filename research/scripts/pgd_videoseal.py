#!/usr/bin/env python3
"""VideoSeal 对抗扰动攻击：白盒 PGD（研究用途）。

在公开的 VideoSeal 模型上做白盒 PGD，最大化提取器的比特错误，测量
不同 L∞ 预算下的 BA 与画质，并检查扰动经过 JPEG 重压后是否存活。

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
import torch.nn.functional as F
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

from baseline_trustmark import synthetic_image  # noqa: E402


def jpeg_roundtrip(img: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def detector_logits(model, imgs: torch.Tensor) -> torch.Tensor:
    """绕过 Videoseal.detect 的 @torch.no_grad，保留 PGD 梯度。"""
    imgs_res = imgs
    if imgs.shape[-2:] != (model.img_size, model.img_size):
        imgs_res = F.interpolate(
            imgs,
            size=(model.img_size, model.img_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
    return model.detector(imgs_res)[:, 1:]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=int, default=4)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument(
        "--mode",
        choices=["invert", "random_target", "dual_tail"],
        default="random_target",
        help="invert=翻转比特；random_target=指向随机消息；dual_tail=把 logits 推向 0",
    )
    parser.add_argument(
        "--epsilons", type=float, nargs="+", default=[2 / 255, 4 / 255, 8 / 255]
    )
    parser.add_argument(
        "--out", type=Path, default=ROOT / "research" / "output" / "pgd_videoseal"
    )
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = videoseal.load("videoseal").to(device).eval()
    rows: list[dict[str, object]] = []

    for image_id in range(args.images):
        cover = synthetic_image(image_id)
        cover_t = transforms.ToTensor()(cover).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model.embed(cover_t, is_video=False)
        message = out["msgs"].float().to(device)
        watermarked = out["imgs_w"].detach()

        for eps in args.epsilons:
            if args.mode == "invert":
                target = 1.0 - message
            elif args.mode == "random_target":
                torch.manual_seed(20260909 + image_id)
                target = torch.randint(0, 2, message.shape).float().to(device)
            else:
                target = None
            alpha = eps / 5.0
            delta = torch.zeros_like(watermarked, requires_grad=True)
            for _ in range(args.steps):
                x_adv = (watermarked + delta).clamp(0.0, 1.0)
                logits = detector_logits(model, x_adv)
                if args.mode == "dual_tail":
                    loss = logits.pow(2).mean()
                else:
                    loss = F.binary_cross_entropy_with_logits(logits, target)
                grad = torch.autograd.grad(loss, delta)[0]
                delta = (delta + alpha * grad.sign()).detach().clamp(-eps, eps)
                delta.requires_grad_(True)

            x_adv = (watermarked + delta).clamp(0.0, 1.0)
            with torch.no_grad():
                bits = (detector_logits(model, x_adv) > 0).float()
            ba = float((bits == message).float().mean())
            ba_target = (
                float((bits == target).float().mean()) if target is not None else float("nan")
            )
            adv_pil = transforms.ToPILImage()(x_adv[0].cpu())
            jpeg_pil = jpeg_roundtrip(adv_pil, 70)
            jpeg_t = transforms.ToTensor()(jpeg_pil).unsqueeze(0).to(device)
            with torch.no_grad():
                jpeg_bits = (detector_logits(model, jpeg_t) > 0).float()
            ba_jpeg = float((jpeg_bits == message).float().mean())

            cover_arr = np.asarray(cover).astype(np.float64)
            adv_arr = np.asarray(adv_pil).astype(np.float64)
            rows.append(
                {
                    "image_id": image_id,
                    "mode": args.mode,
                    "epsilon": eps,
                    "ba": ba,
                    "ba_target": ba_target,
                    "ba_after_jpeg70": ba_jpeg,
                    "psnr_vs_cover": psnr_metric(
                        cover_arr, adv_arr, data_range=255
                    ),
                    "ssim_vs_cover": ssim_metric(
                        cover_arr, adv_arr, channel_axis=2, data_range=255
                    ),
                }
            )
            print(
                f"[image {image_id}] eps={eps*255:.1f}/255 BA={ba:.3f} "
                f"BA_target={ba_target:.3f} "
                f"JPEG70 BA={ba_jpeg:.3f} "
                f"PSNR={rows[-1]['psnr_vs_cover']:.2f} "
                f"SSIM={rows[-1]['ssim_vs_cover']:.4f}"
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.out.with_suffix(".csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md_path = args.out.with_suffix(".md")
    with md_path.open("w") as handle:
        handle.write("# VideoSeal 白盒 PGD 对抗攻击\n\n")
        handle.write(
            f"- 样本：{args.images} 张合成图；PGD {args.steps} 步；"
            f"mode=`{args.mode}`\n"
        )
        handle.write(
            "- BA=对原消息的比特准确率；BA_target=对攻击目标消息的准确率；"
            "JPEG70=扰动后再过 JPEG q70 的 BA\n\n"
        )
        handle.write(
            "| epsilon (/255) | BA | BA_target | JPEG70 BA | PSNR vs 原图 | "
            "SSIM vs 原图 |\n"
        )
        handle.write("| ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for eps in args.epsilons:
            subset = [r for r in rows if abs(r["epsilon"] - eps) < 1e-9]
            handle.write(
                f"| {eps*255:.1f} | "
                f"{np.mean([r['ba'] for r in subset]):.3f} | "
                f"{np.mean([r['ba_target'] for r in subset]):.3f} | "
                f"{np.mean([r['ba_after_jpeg70'] for r in subset]):.3f} | "
                f"{np.mean([r['psnr_vs_cover'] for r in subset]):.2f} | "
                f"{np.mean([r['ssim_vs_cover'] for r in subset]):.4f} |\n"
            )

    print(f"\nCSV: {csv_path}\nMarkdown: {md_path}")


if __name__ == "__main__":
    main()
