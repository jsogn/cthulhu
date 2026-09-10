#!/usr/bin/env python3
"""视频 EOT 鲁棒 PGD（研究用途）。

在自然内容视频上做白盒 PGD，EOT 代理包含可微 JPEG、缩放、模糊和随机抽帧；
优化完成后再用真实 H.264/H.265 编码管线验收，比较静态扰动与逐帧扰动。

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
from kornia.metrics import ssim
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
from video_codec_attacks import read_video, run_ffmpeg, write_video  # noqa: E402


def detector_logits(model, video: torch.Tensor) -> torch.Tensor:
    resized = F.interpolate(
        video,
        size=(model.img_size, model.img_size),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    return model.detector(resized)[:, 1:]


def aggregate_ba(logits: torch.Tensor, message: torch.Tensor) -> tuple[float, float]:
    bits = (logits > 0).float()
    per_frame = float((bits == message).float().mean())
    aggregated = (logits.mean(dim=0, keepdim=True) > 0).float()
    agg = float((aggregated == message).float().mean())
    return agg, per_frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-card", default="videoseal")
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--eps", type=float, default=4 / 255)
    parser.add_argument(
        "--direction",
        choices=["descend", "ascend"],
        default="descend",
        help="descend=最小化 logits²（推向随机）；ascend=最大化（旧口径）",
    )
    parser.add_argument("--ssim-weight", type=float, default=0.0)
    parser.add_argument("--tv-weight", type=float, default=0.0)
    parser.add_argument("--vmaf", action="store_true")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["plain_static", "eot_static"],
        choices=["plain_static", "eot_static", "plain_per_frame", "eot_per_frame"],
    )
    parser.add_argument(
        "--out", type=Path, default=ROOT / "research" / "output" / "eot_pgd_video"
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
    wm_np = (
        (watermarked.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255)
        .round()
        .astype(np.uint8)
    )
    frames, _, height, width = watermarked.shape

    rows: list[dict[str, object]] = []

    vmaf_ref_path: Path | None = None
    if args.vmaf:
        vmaf_dir = Path(tempfile.mkdtemp(prefix="eot_vmaf_"))
        vmaf_ref_path = vmaf_dir / "reference.mkv"
        write_video(wm_np, vmaf_ref_path, codec="ffv1")

    def evaluate_codecs(name: str, attacked: torch.Tensor) -> None:
        with torch.no_grad():
            clean_logits = detector_logits(model, attacked)
        clean_ba, clean_pf = aggregate_ba(clean_logits, message)
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
                    ssim_metric(
                        wm_np[i], attacked_np[i], channel_axis=2, data_range=255
                    )
                    for i in range(n)
                ]
            )
        )
        row: dict[str, object] = {
            "variant": name,
            "ba_no_codec": clean_ba,
            "per_frame_ba_no_codec": clean_pf,
            "psnr": psnr,
            "ssim": ssim,
        }
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            lossless = tmpdir / "attacked.mkv"
            write_video(attacked_np, lossless, codec="ffv1")
            if vmaf_ref_path is not None:
                row["vmaf"] = vmaf_score(
                    str(lossless), str(vmaf_ref_path), subsample=2
                )
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
                    logits = detector_logits(model, decoded_t)
                ba, per_frame = aggregate_ba(logits, message)
                row[f"ba_{codec_name}"] = ba
                row[f"per_frame_{codec_name}"] = per_frame
        rows.append(row)
        print(
            f"{name:18s} no_codec={clean_ba:.3f} "
            f"h264_23={row.get('ba_h264_crf23', float('nan')):.3f} "
            f"h264_28={row.get('ba_h264_crf28', float('nan')):.3f} "
            f"h265_23={row.get('ba_h265_crf23', float('nan')):.3f} "
            f"h265_28={row.get('ba_h265_crf28', float('nan')):.3f} "
            f"PSNR={psnr:5.2f} SSIM={ssim:.4f}"
        )

    for variant in args.variants:
        static = variant.endswith("static")
        use_eot = variant.startswith("eot")
        delta = torch.zeros(
            (1, 3, height, width) if static else (frames, 3, height, width),
            device=device,
            requires_grad=True,
        )
        alpha = args.eps / 5.0
        for step in range(args.steps):
            x_adv = (watermarked + delta).clamp(0.0, 1.0)
            x_eot = x_adv
            wm_eot = watermarked
            if use_eot:
                scale = float(np.random.uniform(0.5, 0.9))
                x_eot = F.interpolate(
                    x_eot,
                    scale_factor=scale,
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
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
                quality_t = torch.full((x_eot.shape[0],), quality, device=device)
                x_eot = jpeg_codec_differentiable(x_eot.clamp(0, 1), quality_t)
                keep = max(4, int(frames * 0.75))
                idx = torch.randperm(frames, device=device)[:keep]
                x_eot = x_eot[idx]
                wm_eot = watermarked[idx]
            logits = detector_logits(model, x_eot)
            loss = logits.pow(2).mean()
            if args.ssim_weight > 0:
                loss = loss + args.ssim_weight * (
                    1.0 - ssim(x_eot, wm_eot, window_size=11).mean()
                )
            if args.tv_weight > 0:
                tv = (
                    (delta[..., :, 1:] - delta[..., :, :-1]).abs().mean()
                    + (delta[..., 1:, :] - delta[..., :-1, :]).abs().mean()
                )
                loss = loss + args.tv_weight * tv
            grad = torch.autograd.grad(loss, delta)[0]
            if args.direction == "descend":
                delta = (delta - alpha * grad.sign()).detach().clamp(
                    -args.eps, args.eps
                )
            else:
                delta = (delta + alpha * grad.sign()).detach().clamp(
                    -args.eps, args.eps
                )
            delta.requires_grad_(True)

        with torch.no_grad():
            attacked = (watermarked + delta).clamp(0.0, 1.0)
        evaluate_codecs(variant, attacked)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.out.with_suffix(".csv")
    with csv_path.open("w", newline="") as handle:
        fieldnames = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    md_path = args.out.with_suffix(".md")
    with md_path.open("w") as handle:
        handle.write(
            f"# 视频 EOT-PGD（{args.model_card}，自然内容，ε={args.eps*255:.1f}/255）\n\n"
        )
        handle.write(
            f"- 视频：{args.frames} 帧 512x512 Ken Burns；PGD {args.steps} 步\n"
        )
        handle.write(
            f"- 方向：`{args.direction}`；SSIM 权重：{args.ssim_weight}；"
            f"TV 权重：{args.tv_weight}\n"
        )
        handle.write("- EOT 代理：可微 JPEG + 缩放 + 高斯模糊 + 随机抽帧\n")
        handle.write("- 验收：真实 H.264/H.265 编码后解码再检测\n\n")
        handle.write(
            "| 变体 | 无编码 BA | H.264 CRF23 | H.264 CRF28 | H.265 CRF23 | "
            "H.265 CRF28 | PSNR | SSIM | VMAF |\n"
        )
        handle.write(
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
        )
        for row in rows:
            handle.write(
                f"| {row['variant']} | {row['ba_no_codec']:.3f} | "
                f"{row.get('ba_h264_crf23', float('nan')):.3f} | "
                f"{row.get('ba_h264_crf28', float('nan')):.3f} | "
                f"{row.get('ba_h265_crf23', float('nan')):.3f} | "
                f"{row.get('ba_h265_crf28', float('nan')):.3f} | "
                f"{row['psnr']:.2f} | {row['ssim']:.4f} | "
                f"{row.get('vmaf', float('nan')):.2f} |\n"
            )
    print(f"\nCSV: {csv_path}\nMarkdown: {md_path}")


if __name__ == "__main__":
    main()
