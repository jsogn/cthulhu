#!/usr/bin/env python3
"""跨帧一致 + 时变分量的视频 EOT-PGD（研究用途）。

设计动机：纯静态扰动在长视频上会被时序聚合部分恢复（64 帧 H.264 CRF23
后 BA 回升到 0.68）。本脚本在静态基底之外叠加一个跨帧零均值的时变分量：
- 静态基底保证聚合 logits 被推向 0；
- 零均值时变分量不改变跨帧平均，但干扰检测器的逐帧/时序特征。

对比模式：`static`（纯静态）vs `temporal`（基底 + 零均值时变）。

边界：公开许可素材、公开预训练模型、本地离线实验，不接触任何平台线上系统。
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from kornia.enhance import jpeg_codec_differentiable
from kornia.filters import gaussian_blur2d
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric

ROOT = Path(__file__).resolve().parents[2]
VSEAL_REPO = ROOT / "research" / "vendor" / "videoseal"
sys.path.insert(0, str(ROOT / "research" / "scripts"))
sys.path.insert(0, str(VSEAL_REPO))
sys.path.insert(0, str(ROOT / "backend" / "src"))
os.chdir(VSEAL_REPO)

import videoseal  # noqa: E402

from cthulhu_backend.media.ffmpeg import vmaf_score  # noqa: E402
from diffusion_natural_video import make_kenburns_video  # noqa: E402
from eot_pgd_video import aggregate_ba, detector_logits  # noqa: E402
from video_codec_attacks import read_video, run_ffmpeg, write_video  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-card", default="videoseal")
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--eps", type=float, default=3 / 255)
    parser.add_argument(
        "--mode", choices=["static", "temporal", "modulated"], default="temporal"
    )
    parser.add_argument("--temporal-weight", type=float, default=0.0)
    parser.add_argument("--mod-lr", type=float, default=0.02)
    parser.add_argument("--tv-weight", type=float, default=0.0)
    parser.add_argument("--vmaf", action="store_true")
    parser.add_argument(
        "--out", type=Path, default=ROOT / "research" / "output" / "eot_pgd_temporal"
    )
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = videoseal.load(args.model_card).to(device).eval()
    cover = make_kenburns_video(
        VSEAL_REPO / "assets" / "imgs" / "1.jpg", args.frames
    ).to(device)
    with torch.no_grad():
        out = model.embed(cover, is_video=True)
    message = out["msgs"][0:1].float().to(device)
    watermarked = out["imgs_w"].detach()
    frames, _, height, width = watermarked.shape
    wm_np = (
        (watermarked.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255)
        .round()
        .astype(np.uint8)
    )

    delta_base = torch.zeros((1, 3, height, width), device=device, requires_grad=True)
    delta_var = None
    if args.mode == "temporal":
        delta_var = torch.zeros(
            (frames, 3, height, width), device=device, requires_grad=True
        )
    mod = None
    if args.mode == "modulated":
        mod = torch.ones((frames,), device=device, requires_grad=True)

    alpha = args.eps / 5.0
    for _ in range(args.steps):
        if delta_var is None and mod is None:
            delta = delta_base
        elif mod is not None:
            mod_centered = mod - mod.mean() + 1.0
            delta = delta_base * mod_centered[:, None, None, None]
        else:
            delta = delta_base + (delta_var - delta_var.mean(dim=0, keepdim=True))
        delta = delta.clamp(-args.eps, args.eps)
        x_adv = (watermarked + delta).clamp(0.0, 1.0)

        scale = float(np.random.uniform(0.5, 0.9))
        x_eot = F.interpolate(
            x_adv, scale_factor=scale, mode="bilinear", align_corners=False, antialias=True
        )
        x_eot = F.interpolate(
            x_eot,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        sigma = float(np.random.uniform(0.3, 1.2))
        x_eot = gaussian_blur2d(x_eot, (5, 5), (sigma, sigma))
        quality = float(np.random.uniform(50.0, 90.0))
        x_eot = jpeg_codec_differentiable(
            x_eot.clamp(0, 1),
            torch.full((x_eot.shape[0],), quality, device=device),
        )
        keep = max(4, int(frames * 0.75))
        idx = torch.randperm(frames, device=device)[:keep]
        logits = detector_logits(model, x_eot[idx])
        loss = logits.pow(2).mean()
        if delta_var is not None and args.temporal_weight > 0:
            loss = loss + args.temporal_weight * (
                (delta_var[1:] - delta_var[:-1]).pow(2).mean()
            )
        if mod is not None and args.temporal_weight > 0:
            loss = loss + args.temporal_weight * (
                (mod[1:] - mod[:-1]).pow(2).mean()
            )
        if args.tv_weight > 0:
            loss = loss + args.tv_weight * (
                (delta[..., :, 1:] - delta[..., :, :-1]).abs().mean()
                + (delta[..., 1:, :] - delta[..., :-1, :]).abs().mean()
            )

        if delta_var is None and mod is None:
            grad = torch.autograd.grad(loss, delta_base)[0]
            delta_base = (delta_base - alpha * grad.sign()).detach().clamp(
                -args.eps, args.eps
            )
            delta_base.requires_grad_(True)
        elif mod is not None:
            grad_base, grad_mod = torch.autograd.grad(loss, [delta_base, mod])
            delta_base = (delta_base - alpha * grad_base.sign()).detach().clamp(
                -args.eps, args.eps
            )
            mod = (mod - args.mod_lr * grad_mod.sign()).detach().clamp(0.5, 1.5)
            delta_base.requires_grad_(True)
            mod.requires_grad_(True)
        else:
            grad_base, grad_var = torch.autograd.grad(
                loss, [delta_base, delta_var]
            )
            delta_base = (delta_base - alpha * grad_base.sign()).detach().clamp(
                -args.eps, args.eps
            )
            delta_var = (delta_var - alpha * grad_var.sign()).detach().clamp(
                -args.eps, args.eps
            )
            delta_base.requires_grad_(True)
            delta_var.requires_grad_(True)

    with torch.no_grad():
        if delta_var is None and mod is None:
            delta_final = delta_base.clamp(-args.eps, args.eps)
        elif mod is not None:
            mod_centered = mod - mod.mean() + 1.0
            delta_final = (
                delta_base * mod_centered[:, None, None, None]
            ).clamp(-args.eps, args.eps)
        else:
            delta_final = (
                delta_base + (delta_var - delta_var.mean(dim=0, keepdim=True))
            ).clamp(-args.eps, args.eps)
        attacked = (watermarked + delta_final).clamp(0.0, 1.0)

    with torch.no_grad():
        logits = detector_logits(model, attacked)
    ba_no_codec, per_frame_no_codec = aggregate_ba(logits, message)
    attacked_np = (
        (attacked.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255)
        .round()
        .astype(np.uint8)
    )
    n = min(len(wm_np), len(attacked_np))
    psnr = float(
        np.mean(
            [
                psnr_metric(wm_np[i], attacked_np[i], data_range=255)
                for i in range(n)
            ]
        )
    )
    ssim = float(
        np.mean(
            [
                ssim_metric(wm_np[i], attacked_np[i], channel_axis=2, data_range=255)
                for i in range(n)
            ]
        )
    )

    row: dict[str, object] = {
        "mode": args.mode,
        "eps": args.eps,
        "steps": args.steps,
        "ba_no_codec": ba_no_codec,
        "per_frame_no_codec": per_frame_no_codec,
        "psnr": psnr,
        "ssim": ssim,
    }
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        lossless = tmpdir / "attacked.mkv"
        write_video(attacked_np, lossless, codec="ffv1")
        if args.vmaf:
            ref = tmpdir / "reference.mkv"
            write_video(wm_np, ref, codec="ffv1")
            row["vmaf"] = vmaf_score(str(lossless), str(ref), subsample=2)
        for codec_name, extra in [
            ("h264_crf23", ["-c:v", "libx264", "-crf", "23", "-preset", "medium", "-pix_fmt", "yuv420p"]),
            ("h264_crf28", ["-c:v", "libx264", "-crf", "28", "-preset", "medium", "-pix_fmt", "yuv420p"]),
            ("h265_crf23", ["-c:v", "libx265", "-crf", "23", "-preset", "medium", "-pix_fmt", "yuv420p"]),
            ("h265_crf28", ["-c:v", "libx265", "-crf", "28", "-preset", "medium", "-pix_fmt", "yuv420p"]),
        ]:
            encoded = tmpdir / f"{codec_name}.mp4"
            run_ffmpeg(lossless, encoded, extra)
            decoded = read_video(encoded, size=width)
            decoded_t = (
                torch.from_numpy(decoded.astype(np.float32) / 255.0)
                .permute(0, 3, 1, 2)
                .to(device)
            )
            with torch.no_grad():
                codec_logits = detector_logits(model, decoded_t)
            codec_ba, codec_pf = aggregate_ba(codec_logits, message)
            row[f"ba_{codec_name}"] = codec_ba
            row[f"per_frame_{codec_name}"] = codec_pf

    args.out.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.out.with_suffix(".csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    md_path = args.out.with_suffix(".md")
    with md_path.open("w") as handle:
        handle.write("# 跨帧一致 + 时变分量 EOT-PGD\n\n")
        handle.write(
            f"- 模型：`{args.model_card}`；视频：{args.frames} 帧 512x512；"
            f"模式：`{args.mode}`；ε={args.eps*255:.1f}/255；{args.steps} 步\n"
        )
        handle.write(
            f"- 时变平滑权重：{args.temporal_weight}；TV 权重：{args.tv_weight}\n\n"
        )
        handle.write(
            "| 无编码 | H.264 CRF23 | H.264 CRF28 | H.265 CRF23 | H.265 CRF28 | "
            "PSNR | SSIM | VMAF |\n"
        )
        handle.write("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        handle.write(
            f"| {row['ba_no_codec']:.3f} | "
            f"{row.get('ba_h264_crf23', float('nan')):.3f} | "
            f"{row.get('ba_h264_crf28', float('nan')):.3f} | "
            f"{row.get('ba_h265_crf23', float('nan')):.3f} | "
            f"{row.get('ba_h265_crf28', float('nan')):.3f} | "
            f"{row['psnr']:.2f} | {row['ssim']:.4f} | "
            f"{row.get('vmaf', float('nan')):.2f} |\n"
        )

    print(
        f"mode={args.mode} no_codec={row['ba_no_codec']:.3f} "
        f"h264_23={row.get('ba_h264_crf23', float('nan')):.3f} "
        f"h264_28={row.get('ba_h264_crf28', float('nan')):.3f} "
        f"h265_23={row.get('ba_h265_crf23', float('nan')):.3f} "
        f"h265_28={row.get('ba_h265_crf28', float('nan')):.3f} "
        f"PSNR={row['psnr']:.2f} SSIM={row['ssim']:.4f} "
        f"VMAF={row.get('vmaf', float('nan')):.2f}"
    )
    print(f"\nCSV: {csv_path}\nMarkdown: {md_path}")


if __name__ == "__main__":
    main()
