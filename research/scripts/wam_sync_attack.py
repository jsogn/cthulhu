#!/usr/bin/env python3
"""WAM 同步/定位模块定向攻击（研究用途）。

WAM 的检测器输出 (1, 1+32, 256, 256)：通道 0 是定位 mask，通道 1..32 是
逐像素消息比特。本脚本对比三种攻击：
- mask：最大化定位 mask 与真实 mask 的 BCE（打同步/定位）
- message：最小化比特 logits²（打消息，双尾）
- both：两者联合
并与扩散净化（sd-turbo）对比，判断同步模块是否比消息编码更脆弱。

边界：公开许可素材、公开预训练模型、本地离线实验，不接触任何平台线上系统。
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
from diffusers import AutoPipelineForImage2Image
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric

ROOT = Path(__file__).resolve().parents[2]
WAM_REPO = ROOT / "research" / "vendor" / "watermark-anything"
VSEAL_REPO = ROOT / "research" / "vendor" / "videoseal"
sys.path.insert(0, str(ROOT / "research" / "scripts"))
sys.path.insert(0, str(WAM_REPO))
os.chdir(WAM_REPO)

from notebooks.inference_utils import (  # noqa: E402
    create_random_mask,
    default_transform,
    load_model_from_checkpoint,
    unnormalize_img,
)
from watermark_anything.data.metrics import msg_predict_inference  # noqa: E402
from watermark_anything.data.transforms import normalize_img  # noqa: E402


def to_pil(tensor: torch.Tensor) -> Image.Image:
    arr = (
        unnormalize_img(tensor)
        .clamp(0.0, 1.0)
        .squeeze(0)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    return Image.fromarray((arr * 255.0).round().astype(np.uint8))


def evaluate(
    wam,
    img_pt: torch.Tensor,
    true_mask_256: torch.Tensor,
    message: torch.Tensor,
) -> tuple[float, float]:
    with torch.no_grad():
        preds = wam.detect(img_pt)["preds"]
    mask_logits = preds[:, 0:1]
    bit_logits = preds[:, 1:]
    pred_mask = (torch.sigmoid(mask_logits) > 0.5).float()
    inter = (pred_mask * true_mask_256).sum().item()
    union = ((pred_mask + true_mask_256) > 0).float().sum().item()
    iou = inter / max(union, 1.0)
    pred_msg = msg_predict_inference(
        bit_logits.cpu(), torch.sigmoid(mask_logits).cpu()
    )
    ba = float((pred_msg == message.cpu()).float().mean().item())
    return iou, ba


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--eps", type=float, default=4 / 255)
    parser.add_argument(
        "--modes", nargs="+", default=["mask", "message", "both"]
    )
    parser.add_argument("--purify-strengths", nargs="+", type=float, default=[0.15, 0.25])
    parser.add_argument("--model", default="stabilityai/sd-turbo")
    parser.add_argument("--diffusion-steps", type=int, default=50)
    parser.add_argument(
        "--out", type=Path, default=ROOT / "research" / "output" / "wam_sync_attack"
    )
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

    source = Image.open(VSEAL_REPO / "assets" / "imgs" / "1.jpg").convert("RGB")
    source = source.resize((512, 512), Image.BICUBIC)
    img_pt = default_transform(source).unsqueeze(0).to(device)
    torch.manual_seed(20260909)
    np.random.seed(20260909)
    message = torch.randint(0, 2, (1, 32)).float().to(device)
    with torch.no_grad():
        embedded = wam.embed(img_pt, message)["imgs_w"]
    mask = create_random_mask(img_pt, num_masks=1, mask_percentage=0.5).to(device)
    watermarked = embedded * mask + img_pt * (1 - mask)
    true_mask_256 = F.interpolate(
        mask, size=(256, 256), mode="nearest"
    ).clamp(0, 1)

    clean_iou, clean_ba = evaluate(wam, watermarked, true_mask_256, message)
    print(f"clean: mask IoU={clean_iou:.3f} message BA={clean_ba:.3f}")

    base_pixel = unnormalize_img(watermarked).clamp(0, 1)
    rows: list[dict[str, object]] = []

    def record(name: str, adv_pixel: torch.Tensor) -> None:
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
                "attack": name,
                "mask_iou": iou,
                "message_ba": ba,
                "psnr": psnr_metric(base_np, adv_np, data_range=255),
                "ssim": ssim_metric(base_np, adv_np, channel_axis=2, data_range=255),
            }
        )
        print(
            f"{name:22s} mask IoU={iou:.3f} message BA={ba:.3f} "
            f"PSNR={rows[-1]['psnr']:.2f} SSIM={rows[-1]['ssim']:.4f}"
        )

    record("clean", base_pixel)

    for mode in args.modes:
        delta = torch.zeros_like(base_pixel, requires_grad=True)
        alpha = args.eps / 5.0
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
            elif mode == "message":
                loss = msg_loss
                direction = -1.0
            else:
                loss = msg_loss - 1.0 * mask_loss
                direction = -1.0
            grad = torch.autograd.grad(loss, delta)[0]
            delta = (delta + direction * alpha * grad.sign()).detach().clamp(
                -args.eps, args.eps
            )
            delta.requires_grad_(True)
        record(f"pgd_{mode}", (base_pixel + delta).clamp(0, 1))

    base_pil = to_pil(watermarked)
    for strength in args.purify_strengths:
        generator = torch.Generator(device="cpu").manual_seed(20260909)
        purified = pipe(
            prompt="",
            image=base_pil,
            strength=strength,
            guidance_scale=0.0,
            num_inference_steps=args.diffusion_steps,
            generator=generator,
        ).images[0]
        purified_pt = default_transform(purified).unsqueeze(0).to(device)
        iou, ba = evaluate(wam, purified_pt, true_mask_256, message)
        pur_np = np.asarray(purified).astype(np.float64)
        base_np = np.asarray(base_pil).astype(np.float64)
        rows.append(
            {
                "attack": f"purify_s{strength:.2f}",
                "mask_iou": iou,
                "message_ba": ba,
                "psnr": psnr_metric(base_np, pur_np, data_range=255),
                "ssim": ssim_metric(base_np, pur_np, channel_axis=2, data_range=255),
            }
        )
        print(
            f"purify_s{strength:.2f}       mask IoU={iou:.3f} message BA={ba:.3f} "
            f"PSNR={rows[-1]['psnr']:.2f} SSIM={rows[-1]['ssim']:.4f}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.out.with_suffix(".csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    md_path = args.out.with_suffix(".md")
    with md_path.open("w") as handle:
        handle.write("# WAM 同步/定位模块定向攻击\n\n")
        handle.write(
            f"- 模型：WAM MIT 权重；素材：VideoSeal 自带自然图像（512x512）；"
            f"局部水印 mask 50%；ε={args.eps*255:.1f}/255；{args.steps} 步\n\n"
        )
        handle.write("| 攻击 | mask IoU | 消息 BA | PSNR | SSIM |\n")
        handle.write("| --- | ---: | ---: | ---: | ---: |\n")
        for row in rows:
            handle.write(
                f"| {row['attack']} | {row['mask_iou']:.3f} | "
                f"{row['message_ba']:.3f} | {row['psnr']:.2f} | "
                f"{row['ssim']:.4f} |\n"
            )
    print(f"\nCSV: {csv_path}\nMarkdown: {md_path}")


if __name__ == "__main__":
    main()
