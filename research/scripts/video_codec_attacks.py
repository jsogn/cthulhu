#!/usr/bin/env python3
"""VideoSeal 视频编解码攻击矩阵（研究用途）。

在公开的 VideoSeal 256-bit 视频水印上测试真实投放链路会遇到的视频处理：
H.264/H.265 多 CRF、码率限制、分辨率缩放、帧率变换、时域裁剪与组合攻击。

边界：合成样本、公开预训练模型、本地离线实验，不接触任何平台线上系统。
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
os.chdir(VSEAL_REPO)

import videoseal  # noqa: E402

from videoseal_transfer import make_video  # noqa: E402

FFMPEG = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"


def write_video(frames: np.ndarray, path: Path, codec: str = "ffv1") -> None:
    h, w = frames.shape[1:3]
    cmd = [
        FFMPEG,
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{w}x{h}",
        "-r",
        "30",
        "-i",
        "-",
        "-c:v",
        codec,
        str(path),
    ]
    subprocess.run(cmd, input=frames.tobytes(), check=True, capture_output=True)


def read_video(path: Path, size: int = 512) -> np.ndarray:
    cmd = [
        FFMPEG,
        "-i",
        str(path),
        "-vf",
        f"scale={size}:{size}",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    out = subprocess.run(cmd, check=True, capture_output=True).stdout
    frame_bytes = size * size * 3
    n = len(out) // frame_bytes
    return np.frombuffer(out[: n * frame_bytes], dtype=np.uint8).reshape(n, size, size, 3)


def run_ffmpeg(src: Path, dst: Path, extra: list[str]) -> None:
    cmd = [FFMPEG, "-y", "-i", str(src), *extra, str(dst)]
    subprocess.run(cmd, check=True, capture_output=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument(
        "--out", type=Path, default=ROOT / "research" / "output" / "video_codec_attacks"
    )
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = videoseal.load("videoseal").to(device).eval()

    cover = make_video(args.frames).to(device)
    with torch.no_grad():
        out = model.embed(cover, is_video=True)
    message = out["msgs"][0:1].float().to(device)
    watermarked = out["imgs_w"]
    wm_np = (
        (watermarked.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255)
        .round()
        .astype(np.uint8)
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        lossless = tmpdir / "watermarked.mkv"
        write_video(wm_np, lossless, codec="ffv1")

        attacks: list[tuple[str, list[str]]] = [
            ("h264_crf18", ["-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p"]),
            ("h264_crf23", ["-c:v", "libx264", "-crf", "23", "-preset", "medium", "-pix_fmt", "yuv420p"]),
            ("h264_crf28", ["-c:v", "libx264", "-crf", "28", "-preset", "medium", "-pix_fmt", "yuv420p"]),
            ("h264_crf33", ["-c:v", "libx264", "-crf", "33", "-preset", "medium", "-pix_fmt", "yuv420p"]),
            ("h264_crf38", ["-c:v", "libx264", "-crf", "38", "-preset", "medium", "-pix_fmt", "yuv420p"]),
            ("h265_crf23", ["-c:v", "libx265", "-crf", "23", "-preset", "medium", "-pix_fmt", "yuv420p"]),
            ("h265_crf28", ["-c:v", "libx265", "-crf", "28", "-preset", "medium", "-pix_fmt", "yuv420p"]),
            ("h265_crf33", ["-c:v", "libx265", "-crf", "33", "-preset", "medium", "-pix_fmt", "yuv420p"]),
            ("h264_bitrate_500k", ["-c:v", "libx264", "-b:v", "500k", "-preset", "medium", "-pix_fmt", "yuv420p"]),
            ("h264_bitrate_200k", ["-c:v", "libx264", "-b:v", "200k", "-preset", "medium", "-pix_fmt", "yuv420p"]),
            ("scale_360", ["-vf", "scale=360:360", "-c:v", "libx264", "-crf", "23", "-preset", "medium", "-pix_fmt", "yuv420p"]),
            ("fps_15", ["-vf", "fps=15", "-c:v", "libx264", "-crf", "23", "-preset", "medium", "-pix_fmt", "yuv420p"]),
            ("temporal_crop_50", ["-ss", "0.5", "-t", "0.53", "-c:v", "libx264", "-crf", "23", "-preset", "medium", "-pix_fmt", "yuv420p"]),
            ("scale360_h264crf28", ["-vf", "scale=360:360", "-c:v", "libx264", "-crf", "28", "-preset", "medium", "-pix_fmt", "yuv420p"]),
        ]

        rows: list[dict[str, object]] = []
        for name, extra in attacks:
            attacked_path = tmpdir / f"{name}.mp4"
            run_ffmpeg(lossless, attacked_path, extra)
            attacked_np = read_video(attacked_path, size=512)
            attacked_t = (
                torch.from_numpy(attacked_np.astype(np.float32) / 255.0)
                .permute(0, 3, 1, 2)
                .to(device)
            )
            with torch.no_grad():
                preds = model.detect(attacked_t, is_video=True)["preds"]
            aggregated = (preds[:, 1:].mean(dim=0) > 0).float()
            ba = float((aggregated == message).float().mean())
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
                            wm_np[i],
                            attacked_np[i],
                            channel_axis=2,
                            data_range=255,
                        )
                        for i in range(n)
                    ]
                )
            )
            rows.append(
                {
                    "attack": name,
                    "ba": ba,
                    "frames": len(attacked_np),
                    "psnr_vs_watermarked": psnr,
                    "ssim_vs_watermarked": ssim,
                }
            )
            print(
                f"{name:24s} BA={ba:.3f} frames={len(attacked_np):3d} "
                f"PSNR={psnr:5.2f} SSIM={ssim:.4f}"
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.out.with_suffix(".csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md_path = args.out.with_suffix(".md")
    with md_path.open("w") as handle:
        handle.write("# VideoSeal 视频编解码攻击矩阵\n\n")
        handle.write(
            f"- 视频：{args.frames} 帧 512x512；无攻击时 BA=1.000\n"
        )
        handle.write("- BA=聚合比特准确率（0.5 为随机猜测）\n\n")
        handle.write("| 攻击 | BA | 输出帧数 | PSNR | SSIM |\n")
        handle.write("| --- | ---: | ---: | ---: | ---: |\n")
        for row in rows:
            handle.write(
                f"| {row['attack']} | {row['ba']:.3f} | {row['frames']} | "
                f"{row['psnr_vs_watermarked']:.2f} | "
                f"{row['ssim_vs_watermarked']:.4f} |\n"
            )

    print(f"\nCSV: {csv_path}\nMarkdown: {md_path}")


if __name__ == "__main__":
    main()
