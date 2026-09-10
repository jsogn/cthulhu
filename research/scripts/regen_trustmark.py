#!/usr/bin/env python3
"""再生/净化攻击对深度盲水印的基线（研究用途）。

在 TrustMark Q（100-bit 原始载荷、关闭 ECC）上评估：
- VAE 再生（encode -> decode，1/2/4 轮）
- 加噪 + VAE 净化（diffusion purification 的轻量代理）
- Real-ESRGAN x4 超分再生（本机 ONNX 权重）
- Real-ESRGAN + VAE 组合

边界：合成样本、公开预训练模型、本地离线实验，不接触任何平台线上系统。
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from diffusers import AutoencoderKL
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "scripts"))
sys.path.insert(0, str(ROOT / "backend" / "src"))

from baseline_trustmark import bit_accuracy, random_bits, synthetic_image  # noqa: E402
from trustmark import TrustMark  # noqa: E402


def to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB")).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def to_pil(tensor: torch.Tensor) -> Image.Image:
    arr = (
        ((tensor.clamp(-1.0, 1.0) + 1.0) * 127.5)
        .round()
        .to(torch.uint8)
        .squeeze(0)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    return Image.fromarray(arr)


@torch.no_grad()
def vae_roundtrip(
    vae: AutoencoderKL,
    img: Image.Image,
    device: str,
    *,
    mode: str = "mode",
    noise: float = 0.0,
    passes: int = 1,
) -> Image.Image:
    x = to_tensor(img).to(device)
    if noise > 0:
        x = (x + noise * torch.randn_like(x)).clamp(-1.0, 1.0)
    for _ in range(passes):
        posterior = vae.encode(x)
        z = (
            posterior.latent_dist.mode()
            if mode == "mode"
            else posterior.latent_dist.sample()
        )
        x = vae.decode(z).sample.clamp(-1.0, 1.0)
    return to_pil(x)


class RealEsrgan:
    def __init__(self, model_path: Path) -> None:
        self.session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )

    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.asarray(img.convert("RGB")).astype(np.float32) / 255.0
        inp = arr.transpose(2, 0, 1)[None]
        out = self.session.run(None, {"input": inp})[0][0]
        out = np.clip(out, 0.0, 1.0).transpose(1, 2, 0)
        upscaled = Image.fromarray((out * 255.0).round().astype(np.uint8))
        return upscaled.resize(img.size, Image.BICUBIC)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=int, default=4)
    parser.add_argument(
        "--out", type=Path, default=ROOT / "research" / "output" / "regen_trustmark"
    )
    parser.add_argument(
        "--vae",
        default="stabilityai/sd-vae-ft-mse",
        help="diffusers AutoencoderKL 模型 ID",
    )
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(20260909)
    vae = AutoencoderKL.from_pretrained(args.vae, torch_dtype=torch.float32).to(device)
    vae.eval()
    sr = RealEsrgan(ROOT / "backend" / "models" / "realesr-general-x4v3.onnx")
    tm = TrustMark(use_ECC=False, verbose=True, model_type="Q", loadRemover=False)
    rng = np.random.default_rng(20260909)

    rows: list[dict[str, object]] = []
    for image_id in range(args.images):
        cover = synthetic_image(image_id)
        payload = random_bits(rng)
        watermarked = tm.encode(cover, payload, MODE="binary")
        clean_pred, _, _ = tm.decode(watermarked, MODE="binary")
        print(f"[image {image_id}] clean BA={bit_accuracy(clean_pred, payload):.3f}")

        variants = {
            "vae_mode": lambda im: vae_roundtrip(vae, im, device, mode="mode"),
            "vae_sample": lambda im: vae_roundtrip(vae, im, device, mode="sample"),
            "vae_mode_x2": lambda im: vae_roundtrip(
                vae, im, device, mode="mode", passes=2
            ),
            "vae_mode_x4": lambda im: vae_roundtrip(
                vae, im, device, mode="mode", passes=4
            ),
            "noise0.05_vae": lambda im: vae_roundtrip(
                vae, im, device, mode="mode", noise=0.05
            ),
            "noise0.10_vae": lambda im: vae_roundtrip(
                vae, im, device, mode="mode", noise=0.10
            ),
            "noise0.20_vae": lambda im: vae_roundtrip(
                vae, im, device, mode="mode", noise=0.20
            ),
            "noise0.30_vae": lambda im: vae_roundtrip(
                vae, im, device, mode="mode", noise=0.30
            ),
            "real_esrgan_x4": lambda im: sr(im),
            "real_esrgan_x4_vae": lambda im: vae_roundtrip(
                vae, sr(im), device, mode="mode"
            ),
        }

        cover_arr = np.asarray(cover).astype(np.float64)
        wm_arr = np.asarray(watermarked).astype(np.float64)
        for name, fn in variants.items():
            attacked = fn(watermarked)
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
        handle.write("# 再生/净化攻击 vs TrustMark Q（研究基线）\n\n")
        handle.write(
            f"- 样本：{args.images} 张合成图；载荷：100-bit 原始二进制（关闭 ECC）\n"
        )
        handle.write(f"- VAE：`{args.vae}`（{device}）\n")
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
            f"{attack:20s} BA={np.mean([r['bit_accuracy'] for r in subset]):.3f} "
            f"PSNR_wm={np.mean([r['psnr_vs_watermarked'] for r in subset]):5.2f} "
            f"PSNR_cover={np.mean([r['psnr_vs_cover'] for r in subset]):5.2f}"
        )


if __name__ == "__main__":
    main()
