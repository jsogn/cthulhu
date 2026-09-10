#!/usr/bin/env python3
"""H.264-in-the-loop + STE 的视频 PGD（研究用途）。

动机：可微 JPEG 代理无法完全模拟 H.264 CRF28，导致 CRF28 是当前唯一
没完全打透的管线。本脚本把真实 ffmpeg H.264 编码放进优化循环，用直通
估计（STE）把解码帧的梯度传回扰动：

    x_ste = x_adv + (codec(x_adv) - x_adv).detach()

优化后用真实 H.264/H.265 管线验收，比较是否能突破 CRF28。

边界：公开许可素材、公开预训练模型、本地离线实验，不接触任何平台线上系统。
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric

ROOT = Path(__file__).resolve().parents[2]
VSEAL_REPO = ROOT / "research" / "vendor" / "videoseal"
sys.path.insert(0, str(ROOT / "research" / "scripts"))
sys.path.insert(0, str(VSEAL_REPO))
sys.path.insert(0, str(ROOT / "backend" / "src"))
os.chdir(VSEAL_REPO)

import videoseal  # noqa: E402

from diffusion_natural_video import make_kenburns_video  # noqa: E402
from eot_pgd_video import aggregate_ba, detector_logits  # noqa: E402
from video_codec_attacks import read_video, run_ffmpeg, write_video  # noqa: E402

FFMPEG = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"


def h264_roundtrip(
    frames: torch.Tensor, crf: int, preset: str, device: str
) -> torch.Tensor:
    """真实 H.264 编码-解码（非可微），返回与输入同形状的帧张量。"""
    np_frames = (
        (frames.detach().clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255)
        .round()
        .astype(np.uint8)
    )
    f, h, w, _ = np_frames.shape
    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "codec.mp4"
        encode = [
            FFMPEG, "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", "30",
            "-i", "-",
            "-c:v", "libx264", "-crf", str(crf), "-preset", preset,
            "-pix_fmt", "yuv420p", str(out_path),
        ]
        subprocess.run(
            encode, input=np_frames.tobytes(), check=True, capture_output=True
        )
        decode = [
            FFMPEG, "-i", str(out_path),
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ]
        raw = subprocess.run(decode, check=True, capture_output=True).stdout
    decoded = np.frombuffer(raw, dtype=np.uint8).reshape(-1, h, w, 3)
    n = min(len(decoded), len(np_frames))
    tensor = torch.from_numpy(decoded[:n].astype(np.float32) / 255.0)
    return tensor.permute(0, 3, 1, 2).to(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-card", default="videoseal")
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--eps", type=float, default=4 / 255)
    parser.add_argument("--crf", type=int, default=28)
    parser.add_argument("--preset", default="ultrafast")
    parser.add_argument(
        "--out", type=Path, default=ROOT / "research" / "output" / "eot_pgd_h264_ste"
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
    for step in range(args.steps):
        mod_centered = mod - mod.mean() + 1.0
        delta = (delta_base * mod_centered[:, None, None, None]).clamp(
            -args.eps, args.eps
        )
        x_adv = (watermarked + delta).clamp(0.0, 1.0)

        decoded = h264_roundtrip(x_adv, args.crf, args.preset, device)
        x_ste = x_adv[: len(decoded)] + (decoded - x_adv[: len(decoded)]).detach()
        logits = detector_logits(model, x_ste)
        loss = logits.pow(2).mean()
        grad_base, grad_mod = torch.autograd.grad(loss, [delta_base, mod])
        delta_base = (delta_base - alpha * grad_base.sign()).detach().clamp(
            -args.eps, args.eps
        )
        mod = (mod - 0.02 * grad_mod.sign()).detach().clamp(0.5, 1.5)
        delta_base.requires_grad_(True)
        mod.requires_grad_(True)
        if step % 5 == 0:
            print(
                f"step {step:3d} loss={float(loss):.4f} "
                f"mod_mean={float(mod.mean()):.3f}"
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

    rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        lossless = tmpdir / "attacked.mkv"
        write_video(attacked_np, lossless, codec="ffv1")
        for name, extra in [
            ("h264_crf23_512", ["-vf", "null", "-c:v", "libx264", "-crf", "23", "-preset", "medium", "-pix_fmt", "yuv420p"]),
            ("h264_crf28_512", ["-vf", "null", "-c:v", "libx264", "-crf", "28", "-preset", "medium", "-pix_fmt", "yuv420p"]),
            ("h265_crf23_512", ["-vf", "null", "-c:v", "libx265", "-crf", "23", "-preset", "medium", "-pix_fmt", "yuv420p"]),
            ("h265_crf28_512", ["-vf", "null", "-c:v", "libx265", "-crf", "28", "-preset", "medium", "-pix_fmt", "yuv420p"]),
            ("h264_crf23_360", ["-vf", "scale=360:360", "-c:v", "libx264", "-crf", "23", "-preset", "medium", "-pix_fmt", "yuv420p"]),
            ("h264_crf28_360", ["-vf", "scale=360:360", "-c:v", "libx264", "-crf", "28", "-preset", "medium", "-pix_fmt", "yuv420p"]),
            ("fps15_h264_crf23", ["-vf", "fps=15", "-c:v", "libx264", "-crf", "23", "-preset", "medium", "-pix_fmt", "yuv420p"]),
        ]:
            encoded = tmpdir / f"{name}.mp4"
            run_ffmpeg(lossless, encoded, extra)
            decoded = read_video(encoded, size=width)
            decoded_t = (
                torch.from_numpy(decoded.astype(np.float32) / 255.0)
                .permute(0, 3, 1, 2)
                .to(device)
            )
            with torch.no_grad():
                codec_logits = detector_logits(model, decoded_t)
            ba, pf = aggregate_ba(codec_logits, message)
            rows.append(
                {
                    "pipeline": name,
                    "ba": ba,
                    "per_frame_ba": pf,
                    "frames": len(decoded),
                }
            )
            print(f"{name:20s} BA={ba:.3f} per_frame={pf:.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.out.with_suffix(".csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    md_path = args.out.with_suffix(".md")
    with md_path.open("w") as handle:
        handle.write("# H.264-in-the-loop + STE 视频 PGD\n\n")
        handle.write(
            f"- 模型：`{args.model_card}`；视频：{args.frames} 帧 512x512；"
            f"ε={args.eps*255:.1f}/255；{args.steps} 步；"
            f"in-loop H.264 CRF{args.crf}（{args.preset}）\n"
        )
        handle.write(
            f"- 攻击前 BA={ba_no_codec:.3f}；PSNR {psnr:.2f} / SSIM {ssim:.4f}\n\n"
        )
        handle.write("| 管线 | BA | 逐帧 BA | 输出帧数 |\n")
        handle.write("| --- | ---: | ---: | ---: |\n")
        for row in rows:
            handle.write(
                f"| {row['pipeline']} | {row['ba']:.3f} | "
                f"{row['per_frame_ba']:.3f} | {row['frames']} |\n"
            )
    print(f"\nCSV: {csv_path}\nMarkdown: {md_path}")


if __name__ == "__main__":
    main()
