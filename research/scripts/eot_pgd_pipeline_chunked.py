#!/usr/bin/env python3
"""内存安全的分块 EOT-PGD 管线验收（研究用途）。

与 eot_pgd_pipeline.py 相同的攻击目标，但 PGD 阶段按 chunk 分块前向/反向并
累积梯度，避免 128 帧整段反向传播导致的 MPS 内存峰值。每块结束清理缓存。

边界：公开许可素材、公开预训练模型、本地离线实验，不接触任何平台线上系统。
"""

from __future__ import annotations

import argparse
import csv
import gc
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


def mps_mem() -> str:
    if not torch.backends.mps.is_available():
        return "cpu"
    try:
        cur = torch.mps.current_allocated_memory() / 1024**3
        drv = torch.mps.driver_allocated_memory() / 1024**3
        return f"mps cur={cur:.2f}GB drv={drv:.2f}GB"
    except Exception:
        return "mps n/a"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-card", default="videoseal")
    parser.add_argument(
        "--input-video",
        type=Path,
        default=None,
        help="可选：使用真实视频片段作为载体，而不是 Ken Burns 生成视频",
    )
    parser.add_argument("--frames", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--eps", type=float, default=4 / 255)
    parser.add_argument("--jpeg-min", type=float, default=25.0)
    parser.add_argument("--jpeg-max", type=float, default=60.0)
    parser.add_argument("--scale-min", type=float, default=0.3)
    parser.add_argument("--scale-max", type=float, default=0.9)
    parser.add_argument("--drop-min", type=float, default=0.4)
    parser.add_argument("--drop-max", type=float, default=1.0)
    parser.add_argument("--blur-max", type=float, default=2.0)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "research" / "output" / "eot_pgd_pipeline_chunked",
    )
    args = parser.parse_args()
    if args.input_video is not None and not args.input_video.is_absolute():
        args.input_video = ROOT / args.input_video

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = videoseal.load(args.model_card).to(device).eval()
    if args.input_video is not None:
        real_frames = read_video(args.input_video, size=512)
        cover = (
            torch.from_numpy(real_frames.astype(np.float32) / 255.0)
            .permute(0, 3, 1, 2)
            .to(device)
        )[: args.frames]
        print(f"loaded real video {tuple(cover.shape)} from {args.input_video}")
    else:
        cover = make_kenburns_video(
            VSEAL_REPO / "assets" / "imgs" / "1.jpg", args.frames
        ).to(device)
    with torch.no_grad():
        out = model.embed(cover, is_video=True)
    message = out["msgs"][0:1].float().to(device)
    watermarked = out["imgs_w"].detach()
    frames, _, height, width = watermarked.shape
    print(f"loaded video {frames} frames {height}x{width}; {mps_mem()}")

    delta_base = torch.zeros((1, 3, height, width), device=device, requires_grad=True)
    mod = torch.ones((frames,), device=device, requires_grad=True)
    alpha = args.eps / 5.0

    for step in range(args.steps):
        grad_base_accum = torch.zeros_like(delta_base)
        grad_mod_accum = torch.zeros_like(mod)
        for start in range(0, frames, args.chunk_size):
            end = min(start + args.chunk_size, frames)
            mod_centered = mod - mod.mean() + 1.0
            delta_chunk = (
                delta_base * mod_centered[start:end, None, None, None]
            ).clamp(-args.eps, args.eps)
            x_adv = (watermarked[start:end] + delta_chunk).clamp(0.0, 1.0)

            scale = float(np.random.uniform(args.scale_min, args.scale_max))
            x_eot = F.interpolate(
                x_adv,
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
            sigma = float(np.random.uniform(0.3, args.blur_max))
            x_eot = gaussian_blur2d(x_eot, (5, 5), (sigma, sigma))
            quality = float(np.random.uniform(args.jpeg_min, args.jpeg_max))
            x_eot = jpeg_codec_differentiable(
                x_eot.clamp(0, 1),
                torch.full((x_eot.shape[0],), quality, device=device),
            )
            keep = max(4, int((end - start) * np.random.uniform(
                args.drop_min, args.drop_max
            )))
            idx = torch.randperm(end - start, device=device)[:keep]
            logits = detector_logits(model, x_eot[idx])
            loss = logits.pow(2).mean()
            grad_base, grad_mod = torch.autograd.grad(loss, [delta_base, mod])
            grad_base_accum += grad_base.detach()
            grad_mod_accum += grad_mod.detach()
            del x_adv, x_eot, logits, loss, grad_base, grad_mod
            if device == "mps":
                torch.mps.empty_cache()

        delta_base = (delta_base - alpha * grad_base_accum.sign()).detach().clamp(
            -args.eps, args.eps
        )
        mod = (mod - 0.02 * grad_mod_accum.sign()).detach().clamp(0.5, 1.5)
        delta_base.requires_grad_(True)
        mod.requires_grad_(True)
        if step % 5 == 0:
            print(
                f"step {step:3d} mod_mean={float(mod.mean()):.3f} {mps_mem()}"
            )

    with torch.no_grad():
        mod_centered = mod - mod.mean() + 1.0
        delta_final = (delta_base * mod_centered[:, None, None, None]).clamp(
            -args.eps, args.eps
        )
        attacked = (watermarked + delta_final).clamp(0.0, 1.0)
        logits = detector_logits(model, attacked)
    ba_no_codec, pf_no_codec = aggregate_ba(logits, message)
    attacked_np = (
        (attacked.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255)
        .round()
        .astype(np.uint8)
    )
    wm_np = (
        (watermarked.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255)
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
    del watermarked, cover
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()

    rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        lossless = tmpdir / "attacked.mkv"
        write_video(attacked_np, lossless, codec="ffv1")
        reference = tmpdir / "reference.mkv"
        write_video(wm_np, reference, codec="ffv1")

        def evaluate(name: str, extra: list[str], *, align_step: int = 1) -> None:
            output = tmpdir / f"{name}.mp4"
            run_ffmpeg(lossless, output, extra)
            decoded = read_video(output, size=width)
            decoded_t = (
                torch.from_numpy(decoded.astype(np.float32) / 255.0)
                .permute(0, 3, 1, 2)
                .to(device)
            )
            with torch.no_grad():
                codec_logits = detector_logits(model, decoded_t)
            ba, pf = aggregate_ba(codec_logits, message)
            ref = wm_np[::align_step]
            m = min(len(ref), len(decoded))
            row = {
                "pipeline": name,
                "ba": ba,
                "per_frame_ba": pf,
                "frames": len(decoded),
                "psnr": float(
                    np.mean(
                        [
                            psnr_metric(ref[i], decoded[i], data_range=255)
                            for i in range(m)
                        ]
                    )
                ),
                "ssim": float(
                    np.mean(
                        [
                            ssim_metric(
                                ref[i], decoded[i], channel_axis=2, data_range=255
                            )
                            for i in range(m)
                        ]
                    )
                ),
            }
            if align_step == 1:
                row["vmaf"] = vmaf_score(str(output), str(reference), subsample=2)
            rows.append(row)
            del decoded, decoded_t, codec_logits
            if device == "mps":
                torch.mps.empty_cache()
            vmaf_value = row.get("vmaf")
            vmaf_text = f"{vmaf_value:.2f}" if vmaf_value is not None else "nan"
            print(
                f"{name:24s} BA={ba:.3f} PSNR={row['psnr']:5.2f} "
                f"SSIM={row['ssim']:.4f} VMAF={vmaf_text}"
            )

        h264 = ["-c:v", "libx264", "-preset", "medium", "-pix_fmt", "yuv420p"]
        h265 = ["-c:v", "libx265", "-preset", "medium", "-pix_fmt", "yuv420p"]
        evaluate("h264_crf23_512", ["-crf", "23", *h264])
        evaluate("h264_crf28_512", ["-crf", "28", *h264])
        evaluate("h265_crf23_512", ["-crf", "23", *h265])
        evaluate("h265_crf28_512", ["-crf", "28", *h265])
        evaluate("h264_crf23_360", ["-vf", "scale=360:360", "-crf", "23", *h264])
        evaluate("h264_crf28_360", ["-vf", "scale=360:360", "-crf", "28", *h264])
        evaluate("fps15_h264_crf23", ["-vf", "fps=15", "-crf", "23", *h264], align_step=2)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.out.with_suffix(".csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({k for r in rows for k in r}))
        writer.writeheader()
        writer.writerows(rows)
    md_path = args.out.with_suffix(".md")
    with md_path.open("w") as handle:
        handle.write("# 分块 EOT-PGD 管线验收（内存安全）\n\n")
        handle.write(
            f"- 模型：`{args.model_card}`；视频：{args.frames} 帧 512x512；"
            f"chunk={args.chunk_size}；ε={args.eps*255:.1f}/255；{args.steps} 步\n"
        )
        if args.input_video is not None:
            handle.write(f"- 载体：真实视频片段 `{args.input_video}`\n")
        handle.write(
            f"- 攻击前 BA={ba_no_codec:.3f}；PSNR {psnr:.2f} / SSIM {ssim:.4f}\n\n"
        )
        handle.write("| 管线 | BA | 逐帧 BA | 帧数 | PSNR | SSIM | VMAF |\n")
        handle.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for row in rows:
            vmaf_value = row.get("vmaf")
            vmaf_text = f"{vmaf_value:.2f}" if vmaf_value is not None else "nan"
            handle.write(
                f"| {row['pipeline']} | {row['ba']:.3f} | "
                f"{row['per_frame_ba']:.3f} | {row['frames']} | "
                f"{row['psnr']:.2f} | {row['ssim']:.4f} | {vmaf_text} |\n"
            )
    print(f"\nCSV: {csv_path}\nMarkdown: {md_path}")


if __name__ == "__main__":
    main()
