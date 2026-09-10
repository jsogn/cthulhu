#!/usr/bin/env python3
"""视频 EOT-PGD 黑盒迁移测试（研究用途）。

在源模型（默认 VideoSeal）上优化静态扰动，然后把同一扰动迁移到目标模型
（默认 PixelSeal）的水印视频上，比较源/目标模型在真实 H.264/H.265 编码
后的 BA。用于判断抗编码扰动是否依赖白盒模型访问。

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
os.chdir(VSEAL_REPO)

import videoseal  # noqa: E402

from diffusion_natural_video import make_kenburns_video  # noqa: E402
from eot_pgd_video import aggregate_ba, detector_logits  # noqa: E402
from video_codec_attacks import read_video, run_ffmpeg, write_video  # noqa: E402


def evaluate(
    model, video: torch.Tensor, message: torch.Tensor, width: int
) -> dict[str, float]:
    with torch.no_grad():
        logits = detector_logits(model, video)
    ba, per_frame = aggregate_ba(logits, message)
    out: dict[str, float] = {"ba": ba, "per_frame_ba": per_frame}
    video_np = (
        (video.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255)
        .round()
        .astype(np.uint8)
    )
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        lossless = tmpdir / "video.mkv"
        write_video(video_np, lossless, codec="ffv1")
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
                .to(video.device)
            )
            with torch.no_grad():
                codec_logits = detector_logits(model, decoded_t)
            codec_ba, codec_pf = aggregate_ba(codec_logits, message)
            out[f"ba_{codec_name}"] = codec_ba
            out[f"per_frame_{codec_name}"] = codec_pf
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-card", default="videoseal")
    parser.add_argument("--target-card", default="pixelseal")
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--eps", type=float, default=3 / 255)
    parser.add_argument(
        "--out", type=Path, default=ROOT / "research" / "output" / "eot_pgd_transfer"
    )
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    source = videoseal.load(args.source_card).to(device).eval()
    target = videoseal.load(args.target_card).to(device).eval()
    cover = make_kenburns_video(
        VSEAL_REPO / "assets" / "imgs" / "1.jpg", args.frames
    ).to(device)
    torch.manual_seed(20260909)
    message = torch.randint(0, 2, (1, 256)).float().to(device)
    with torch.no_grad():
        src_wm = source.embed(cover, msgs=message, is_video=True)["imgs_w"]
        tgt_wm = target.embed(cover, msgs=message, is_video=True)["imgs_w"]
    frames, _, height, width = src_wm.shape

    delta = torch.zeros((1, 3, height, width), device=device, requires_grad=True)
    alpha = args.eps / 5.0
    for _ in range(args.steps):
        x_adv = (src_wm + delta).clamp(0.0, 1.0)
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
        logits = detector_logits(source, x_eot[idx])
        loss = logits.pow(2).mean()
        grad = torch.autograd.grad(loss, delta)[0]
        delta = (delta - alpha * grad.sign()).detach().clamp(-args.eps, args.eps)
        delta.requires_grad_(True)

    with torch.no_grad():
        src_attacked = (src_wm + delta).clamp(0.0, 1.0)
        tgt_attacked = (tgt_wm + delta).clamp(0.0, 1.0)

    src_result = evaluate(source, src_attacked, message, width)
    tgt_result = evaluate(target, tgt_attacked, message, width)
    tgt_clean = evaluate(target, tgt_wm, message, width)

    tgt_wm_np = (
        (tgt_wm.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255)
        .round()
        .astype(np.uint8)
    )
    tgt_att_np = (
        (tgt_attacked.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255)
        .round()
        .astype(np.uint8)
    )
    n = min(len(tgt_wm_np), len(tgt_att_np))
    psnr = float(
        np.mean(
            [
                psnr_metric(tgt_wm_np[i], tgt_att_np[i], data_range=255)
                for i in range(n)
            ]
        )
    )
    ssim = float(
        np.mean(
            [
                ssim_metric(
                    tgt_wm_np[i], tgt_att_np[i], channel_axis=2, data_range=255
                )
                for i in range(n)
            ]
        )
    )

    row = {
        "source_card": args.source_card,
        "target_card": args.target_card,
        "epsilon": args.eps,
        "steps": args.steps,
        "source_ba": src_result["ba"],
        "source_h264_23": src_result.get("ba_h264_crf23", float("nan")),
        "source_h264_28": src_result.get("ba_h264_crf28", float("nan")),
        "source_h265_23": src_result.get("ba_h265_crf23", float("nan")),
        "source_h265_28": src_result.get("ba_h265_crf28", float("nan")),
        "target_clean_ba": tgt_clean["ba"],
        "target_ba": tgt_result["ba"],
        "target_h264_23": tgt_result.get("ba_h264_crf23", float("nan")),
        "target_h264_28": tgt_result.get("ba_h264_crf28", float("nan")),
        "target_h265_23": tgt_result.get("ba_h265_crf23", float("nan")),
        "target_h265_28": tgt_result.get("ba_h265_crf28", float("nan")),
        "target_psnr": psnr,
        "target_ssim": ssim,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.out.with_suffix(".csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    md_path = args.out.with_suffix(".md")
    with md_path.open("w") as handle:
        handle.write("# 视频 EOT-PGD 黑盒迁移\n\n")
        handle.write(
            f"- 源模型：`{args.source_card}`；目标模型：`{args.target_card}`\n"
        )
        handle.write(
            f"- 扰动：静态、ε={args.eps*255:.1f}/255、{args.steps} 步、EOT\n\n"
        )
        handle.write("| 模型 | 无编码 | H.264 23 | H.264 28 | H.265 23 | H.265 28 |\n")
        handle.write("| --- | ---: | ---: | ---: | ---: | ---: |\n")
        handle.write(
            f"| 源 {args.source_card} | {row['source_ba']:.3f} | "
            f"{row['source_h264_23']:.3f} | {row['source_h264_28']:.3f} | "
            f"{row['source_h265_23']:.3f} | {row['source_h265_28']:.3f} |\n"
        )
        handle.write(
            f"| 目标 {args.target_card}（干净） | {row['target_clean_ba']:.3f} | — | — | — | — |\n"
        )
        handle.write(
            f"| 目标 {args.target_card}（迁移） | {row['target_ba']:.3f} | "
            f"{row['target_h264_23']:.3f} | {row['target_h264_28']:.3f} | "
            f"{row['target_h265_23']:.3f} | {row['target_h265_28']:.3f} |\n\n"
        )
        handle.write(f"目标视频画质：PSNR {psnr:.2f} dB / SSIM {ssim:.4f}\n")

    print(
        f"source={args.source_card} BA={row['source_ba']:.3f} "
        f"(H264_23={row['source_h264_23']:.3f})\n"
        f"target={args.target_card} clean={row['target_clean_ba']:.3f} "
        f"transfer BA={row['target_ba']:.3f} "
        f"(H264_23={row['target_h264_23']:.3f}, H265_28={row['target_h265_28']:.3f}) "
        f"PSNR={psnr:.2f} SSIM={ssim:.4f}"
    )
    print(f"\nCSV: {csv_path}\nMarkdown: {md_path}")


if __name__ == "__main__":
    main()
