#!/usr/bin/env python3
"""TrustMark 深度盲水印基线评估（研究用途）。

目标：在合成样本上，用 cthulhu 现有的经典攻击原语评估一个公开预训练的
深度盲水印方案（Adobe TrustMark Q，100-bit 原始载荷、关闭 ECC），得到
每族攻击的比特准确率（BA）与画质代价（PSNR/SSIM）基线。

边界：只使用合成样本与公开模型权重，不接触任何平台线上系统，不使用
未授权素材。结果只代表本地实验条件，不代表平台真实判定。
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend" / "src"))

from cthulhu_backend.attacks import dct as dct_attack  # noqa: E402
from cthulhu_backend.attacks import geometric, spatial  # noqa: E402
from cthulhu_backend.transform import regenerate  # noqa: E402
from trustmark import TrustMark  # noqa: E402


def synthetic_image(seed: int, size: int = 512) -> Image.Image:
    """生成确定性的合成测试图，避免使用任何第三方素材。"""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    img = np.zeros((size, size, 3), dtype=np.float64)
    mode = seed % 4

    if mode == 0:
        img[..., 0] = 70 + 130 * xx / size
        img[..., 1] = 55 + 110 * yy / size
        img[..., 2] = 150 - 90 * xx / size
        img[80:220, 60:280] = [225, 95, 55]
        img[280:470, 230:470] = [35, 175, 120]
    elif mode == 1:
        checker = (((xx // 32).astype(int) + (yy // 32).astype(int)) % 2) * 120 + 60
        img[:] = checker[..., None]
        circle = (xx - size / 2) ** 2 + (yy - size / 2) ** 2 < (size / 4) ** 2
        img[circle] = [230, 80, 80]
    elif mode == 2:
        radius = np.sqrt((xx - size / 2) ** 2 + (yy - size / 2) ** 2)
        img[..., 0] = 220 - 160 * radius / radius.max()
        img[..., 1] = 90 + 120 * xx / size
        img[..., 2] = 60 + 150 * yy / size
        img[::40, :, :] = [245, 245, 245]
    else:
        img[:] = [40, 90, 150]
        for _ in range(12):
            cy, cx = rng.integers(60, size - 60, size=2)
            rad = int(rng.integers(20, 70))
            mask = (yy - cy) ** 2 + (xx - cx) ** 2 < rad**2
            color = rng.integers(40, 235, size=3)
            img[mask] = color

    img += rng.normal(0, 2.5, img.shape)
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def random_bits(rng: np.random.Generator, length: int = 100) -> str:
    return "".join(str(int(x)) for x in rng.integers(0, 2, size=length))


def bit_accuracy(pred: str, truth: str) -> float:
    if len(pred) != len(truth):
        return 0.0
    return float(sum(p == t for p, t in zip(pred, truth)) / len(truth))


def jpeg_roundtrip(img: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def apply_attack(name: str, img: Image.Image, rng: np.random.Generator) -> Image.Image:
    """把 cthulhu 现有攻击原语作用到单张 RGB 图上。"""
    arr = np.asarray(img).astype(np.float32) / 255.0

    if name == "identity":
        out = arr
    elif name.startswith("jpeg_"):
        return jpeg_roundtrip(img, int(name.split("_")[1]))
    elif name == "median3":
        out = spatial.median(arr[None, ...], size=3)[0]
    elif name == "median5":
        out = spatial.median(arr[None, ...], size=5)[0]
    elif name == "gaussian_0.8":
        out = spatial.gaussian(arr[None, ...], sigma=0.8)[0]
    elif name == "gaussian_1.5":
        out = spatial.gaussian(arr[None, ...], sigma=1.5)[0]
    elif name == "wiener5":
        out = spatial.wiener_denoise(arr[None, ...], size=5)[0]
    elif name == "pixel_requant_32":
        out = spatial.requant_pixels(arr[None, ...], levels=32)[0]
    elif name == "pixel_requant_16":
        out = spatial.requant_pixels(arr[None, ...], levels=16)[0]
    elif name == "dct_requant_12":
        out = np.stack(
            [dct_attack.requant_dct(arr[..., c], step=12.0) for c in range(3)], axis=-1
        )
    elif name == "dct_requant_8":
        out = np.stack(
            [dct_attack.requant_dct(arr[..., c], step=8.0) for c in range(3)], axis=-1
        )
    elif name == "geo_light":
        out = np.stack(
            [
                geometric.crop_rotate_rescale(
                    arr[None, ..., c], crop_frac=0.05, angle=0.5, rescale=0.99
                )[0]
                for c in range(3)
            ],
            axis=-1,
        )
    elif name == "geo_strong":
        out = np.stack(
            [
                geometric.crop_rotate_rescale(
                    arr[None, ..., c], crop_frac=0.10, angle=1.5, rescale=0.95
                )[0]
                for c in range(3)
            ],
            axis=-1,
        )
    elif name == "fft_phase_0.5":
        out = regenerate.fft_phase(arr[None, ...], strength=0.5, rng=rng)[0]
    elif name == "fft_phase_1.0":
        out = regenerate.fft_phase(arr[None, ...], strength=1.0, rng=rng)[0]
    elif name == "dwt_detail_0.5":
        out = regenerate.dwt_detail(arr[None, ...], strength=0.5, rng=rng)[0]
    elif name == "dwt_detail_1.0":
        out = regenerate.dwt_detail(arr[None, ...], strength=1.0, rng=rng)[0]
    else:
        raise ValueError(f"unknown attack: {name}")

    out = np.clip(out, 0.0, 1.0)
    return Image.fromarray((out * 255.0).round().astype(np.uint8))


ATTACKS = [
    "identity",
    "jpeg_95",
    "jpeg_90",
    "jpeg_70",
    "jpeg_50",
    "median3",
    "median5",
    "gaussian_0.8",
    "gaussian_1.5",
    "wiener5",
    "pixel_requant_32",
    "pixel_requant_16",
    "dct_requant_12",
    "dct_requant_8",
    "geo_light",
    "geo_strong",
    "fft_phase_0.5",
    "fft_phase_1.0",
    "dwt_detail_0.5",
    "dwt_detail_1.0",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=int, default=4, help="合成样本数量")
    parser.add_argument(
        "--out", type=Path, default=ROOT / "research" / "output" / "baseline_trustmark"
    )
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tm = TrustMark(use_ECC=False, verbose=True, model_type="Q", loadRemover=False)
    rng = np.random.default_rng(20260909)

    rows: list[dict[str, object]] = []
    for image_id in range(args.images):
        cover = synthetic_image(image_id)
        payload = random_bits(rng)
        watermarked = tm.encode(cover, payload, MODE="binary")
        clean_pred, _, _ = tm.decode(watermarked, MODE="binary")
        clean_ba = bit_accuracy(clean_pred, payload)
        cover_arr = np.asarray(cover).astype(np.float64)
        wm_arr = np.asarray(watermarked).astype(np.float64)
        print(
            f"[image {image_id}] clean BA={clean_ba:.3f} "
            f"PSNR={psnr_metric(cover_arr, wm_arr, data_range=255):.2f} "
            f"SSIM={ssim_metric(cover_arr, wm_arr, channel_axis=2, data_range=255):.4f}"
        )

        for attack_name in ATTACKS:
            attacked = apply_attack(attack_name, watermarked, rng)
            pred, _, _ = tm.decode(attacked, MODE="binary")
            attacked_arr = np.asarray(attacked).astype(np.float64)
            if attack_name == "identity":
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
                    "attack": attack_name,
                    "bit_accuracy": bit_accuracy(pred, payload),
                    "psnr_vs_watermarked": attack_psnr,
                    "ssim_vs_watermarked": attack_ssim,
                }
            )

    csv_path = args.out.with_suffix(".csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary: dict[str, dict[str, float]] = {}
    for attack_name in ATTACKS:
        subset = [r for r in rows if r["attack"] == attack_name]
        summary[attack_name] = {
            "ba_mean": float(np.mean([r["bit_accuracy"] for r in subset])),
            "ba_std": float(np.std([r["bit_accuracy"] for r in subset])),
            "psnr_mean": float(np.mean([r["psnr_vs_watermarked"] for r in subset])),
            "ssim_mean": float(np.mean([r["ssim_vs_watermarked"] for r in subset])),
        }

    md_path = args.out.with_suffix(".md")
    with md_path.open("w") as handle:
        handle.write("# TrustMark Q 深度盲水印：经典攻击基线\n\n")
        handle.write(
            f"- 样本：{args.images} 张合成图；载荷：100-bit 原始二进制（关闭 ECC）\n"
        )
        handle.write("- 指标：BA=比特准确率（0.5 为随机猜测）；PSNR/SSIM 相对带水印图\n")
        handle.write("- 模型：Adobe TrustMark Q（公开预训练权重）\n\n")
        handle.write("| 攻击 | BA 均值 | BA 标准差 | PSNR dB | SSIM | 判定 |\n")
        handle.write("| --- | ---: | ---: | ---: | ---: | --- |\n")
        for attack_name, values in summary.items():
            if values["ba_mean"] >= 0.90:
                verdict = "水印存活"
            elif values["ba_mean"] <= 0.55:
                verdict = "水印被破坏"
            else:
                verdict = "部分削弱"
            handle.write(
                f"| {attack_name} | {values['ba_mean']:.3f} | {values['ba_std']:.3f} | "
                f"{values['psnr_mean']:.2f} | {values['ssim_mean']:.4f} | {verdict} |\n"
            )

    print(f"\nCSV: {csv_path}\nMarkdown: {md_path}")
    for attack_name, values in summary.items():
        print(
            f"{attack_name:20s} BA={values['ba_mean']:.3f} "
            f"PSNR={values['psnr_mean']:5.2f} SSIM={values['ssim_mean']:.4f}"
        )


if __name__ == "__main__":
    main()
