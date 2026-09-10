#!/usr/bin/env python3
"""正均值调制 EOT-PGD + 帧率/分辨率变化管线验收（研究用途）。

在 64 帧自然内容视频上优化正均值调制扰动，EOT 代理额外包含更强的随机
降分辨率与随机抽帧；优化后用真实管线验收：
- 512p H.264/H.265 CRF23/28
- 360p 降分辨率后 H.264/H.265 编码再升回 512p
- 15fps 降帧后 H.264 编码
- 15fps + 360p 组合

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
    parser.add_argument("--eps", type=float, default=4 / 255)
    parser.add_argument("--jpeg-min", type=float, default=45.0)
    parser.add_argument("--jpeg-max", type=float, default=90.0)
    parser.add_argument("--scale-min", type=float, default=0.4)
    parser.add_argument("--scale-max", type=float, default=0.9)
    parser.add_argument("--drop-min", type=float, default=0.5)
    parser.add_argument("--drop-max", type=float, default=1.0)
    parser.add_argument("--blur-max", type=float, default=1.5)
    parser.add_argument(
        "--out", type=Path, default=ROOT / "research" / "output" / "eot_pgd_pipeline"
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
    mod = torch.ones((frames,), device=device, requires_grad=True)
    alpha = args.eps / 5.0
    for _ in range(args.steps):
        mod_centered = mod - mod.mean() + 1.0
        delta = (delta_base * mod_centered[:, None, None, None]).clamp(
            -args.eps, args.eps
        )
        x_adv = (watermarked + delta).clamp(0.0, 1.0)

        # 更强的 EOT：降分辨率 + 随机抽帧 + JPEG + 模糊
        scale = float(np.random.uniform(args.scale_min, args.scale_max))
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
        sigma = float(np.random.uniform(0.3, args.blur_max))
        x_eot = gaussian_blur2d(x_eot, (5, 5), (sigma, sigma))
        quality = float(np.random.uniform(args.jpeg_min, args.jpeg_max))
        x_eot = jpeg_codec_differentiable(
            x_eot.clamp(0, 1),
            torch.full((x_eot.shape[0],), quality, device=device),
        )
        keep = max(
            8, int(frames * np.random.uniform(args.drop_min, args.drop_max))
        )
        idx = torch.randperm(frames, device=device)[:keep]
        logits = detector_logits(model, x_eot[idx])
        loss = logits.pow(2).mean()
        grad_base, grad_mod = torch.autograd.grad(loss, [delta_base, mod])
        delta_base = (delta_base - alpha * grad_base.sign()).detach().clamp(
            -args.eps, args.eps
        )
        mod = (mod - 0.02 * grad_mod.sign()).detach().clamp(0.5, 1.5)
        delta_base.requires_grad_(True)
        mod.requires_grad_(True)

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

    rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        lossless = tmpdir / "attacked.mkv"
        write_video(attacked_np, lossless, codec="ffv1")
        reference = tmpdir / "reference.mkv"
        write_video(wm_np, reference, codec="ffv1")

        def evaluate(
            name: str,
            extra: list[str],
            *,
            align_step: int = 1,
            with_vmaf: bool = True,
        ) -> None:
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
            n = min(len(ref), len(decoded))
            psnr = float(
                np.mean(
                    [
                        psnr_metric(ref[i], decoded[i], data_range=255)
                        for i in range(n)
                    ]
                )
            )
            ssim = float(
                np.mean(
                    [
                        ssim_metric(
                            ref[i], decoded[i], channel_axis=2, data_range=255
                        )
                        for i in range(n)
                    ]
                )
            )
            row: dict[str, object] = {
                "pipeline": name,
                "ba": ba,
                "per_frame_ba": pf,
                "frames": len(decoded),
                "psnr": psnr,
                "ssim": ssim,
            }
            if with_vmaf and align_step == 1:
                row["vmaf"] = vmaf_score(str(output), str(reference), subsample=2)
            rows.append(row)
            vmaf_value = row.get("vmaf")
            vmaf_text = f"{vmaf_value:.2f}" if vmaf_value is not None else "nan"
            print(
                f"{name:28s} BA={ba:.3f} per_frame={pf:.3f} "
                f"frames={len(decoded):3d} PSNR={psnr:5.2f} SSIM={ssim:.4f} "
                f"VMAF={vmaf_text}"
            )

        base_h264 = ["-c:v", "libx264", "-preset", "medium", "-pix_fmt", "yuv420p"]
        base_h265 = ["-c:v", "libx265", "-preset", "medium", "-pix_fmt", "yuv420p"]

        evaluate("h264_crf23_512", ["-crf", "23", *base_h264])
        evaluate("h264_crf28_512", ["-crf", "28", *base_h264])
        evaluate("h265_crf23_512", ["-crf", "23", *base_h265])
        evaluate("h265_crf28_512", ["-crf", "28", *base_h265])
        evaluate(
            "h264_crf23_360",
            ["-vf", "scale=360:360", "-crf", "23", *base_h264],
            with_vmaf=False,
        )
        evaluate(
            "h264_crf28_360",
            ["-vf", "scale=360:360", "-crf", "28", *base_h264],
            with_vmaf=False,
        )
        evaluate(
            "h265_crf23_360",
            ["-vf", "scale=360:360", "-crf", "23", *base_h265],
            with_vmaf=False,
        )
        evaluate(
            "fps15_h264_crf23",
            ["-vf", "fps=15", "-crf", "23", *base_h264],
            align_step=2,
            with_vmaf=False,
        )
        evaluate(
            "fps15_360_h264_crf23",
            ["-vf", "fps=15,scale=360:360", "-crf", "23", *base_h264],
            align_step=2,
            with_vmaf=False,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.out.with_suffix(".csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({k for r in rows for k in r}))
        writer.writeheader()
        writer.writerows(rows)
    md_path = args.out.with_suffix(".md")
    with md_path.open("w") as handle:
        handle.write("# 正均值调制 EOT-PGD + 帧率/分辨率管线\n\n")
        handle.write(
            f"- 模型：`{args.model_card}`；视频：{args.frames} 帧 512x512；"
            f"ε={args.eps*255:.1f}/255；{args.steps} 步\n"
        )
        handle.write(
            f"- EOT 代理：随机降分辨率 {args.scale_min}~{args.scale_max}、"
            f"随机抽帧 {args.drop_min}~{args.drop_max}、"
            f"可微 JPEG {args.jpeg_min}~{args.jpeg_max}、"
            f"模糊到 {args.blur_max}\n\n"
        )
        handle.write(
            "| 管线 | BA | 逐帧 BA | 输出帧数 | PSNR | SSIM | VMAF |\n"
        )
        handle.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for row in rows:
            vmaf_value = row.get("vmaf")
            vmaf_text = f"{vmaf_value:.2f}" if vmaf_value is not None else "nan"
            handle.write(
                f"| {row['pipeline']} | {row['ba']:.3f} | "
                f"{row['per_frame_ba']:.3f} | {row['frames']} | "
                f"{row['psnr']:.2f} | {row['ssim']:.4f} | "
                f"{vmaf_text} |\n"
            )
    print(f"\nno_codec BA={ba_no_codec:.3f} per_frame={pf_no_codec:.3f}")
    print(f"CSV: {csv_path}\nMarkdown: {md_path}")


if __name__ == "__main__":
    main()
