#!/usr/bin/env python3
"""VideoSeal 共谋攻击实验：多副本平均（研究用途）。

同一段内容嵌入 N 条不同消息，求平均后检测每条原始消息的 BA，观察
共谋副本数增加时水印信号是否被稀释。

边界：合成样本、公开预训练模型、本地离线实验，不接触任何平台线上系统。
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--copies", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument(
        "--out", type=Path, default=ROOT / "research" / "output" / "collusion_videoseal"
    )
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = videoseal.load("videoseal").to(device).eval()
    cover = make_video(args.frames).to(device)
    cover_np = cover.cpu().numpy()
    rows: list[dict[str, object]] = []

    torch.manual_seed(20260909)
    for n in args.copies:
        messages = torch.randint(0, 2, (n, 256)).float().to(device)
        watermarked = []
        with torch.no_grad():
            for i in range(n):
                out = model.embed(cover, msgs=messages[i : i + 1], is_video=True)
                watermarked.append(out["imgs_w"])
        colluded = torch.stack(watermarked).mean(dim=0)

        with torch.no_grad():
            preds = model.detect(colluded, is_video=True)["preds"]
        aggregated = (preds[:, 1:].mean(dim=0) > 0).float()  # k
        bas = [
            float((aggregated == messages[i]).float().mean()) for i in range(n)
        ]
        colluded_np = colluded.cpu().numpy()
        rows.append(
            {
                "copies": n,
                "ba_mean": float(np.mean(bas)),
                "ba_min": float(np.min(bas)),
                "ba_max": float(np.max(bas)),
                "psnr_vs_cover": float(
                    np.mean(
                        [
                            psnr_metric(cover_np[i], colluded_np[i], data_range=1.0)
                            for i in range(len(cover_np))
                        ]
                    )
                ),
                "ssim_vs_cover": float(
                    np.mean(
                        [
                            ssim_metric(
                                cover_np[i].transpose(1, 2, 0),
                                colluded_np[i].transpose(1, 2, 0),
                                channel_axis=2,
                                data_range=1.0,
                            )
                            for i in range(len(cover_np))
                        ]
                    )
                ),
            }
        )
        print(
            f"copies={n} BA mean={rows[-1]['ba_mean']:.3f} "
            f"min={rows[-1]['ba_min']:.3f} max={rows[-1]['ba_max']:.3f} "
            f"PSNR={rows[-1]['psnr_vs_cover']:.2f} SSIM={rows[-1]['ssim_vs_cover']:.4f}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.out.with_suffix(".csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md_path = args.out.with_suffix(".md")
    with md_path.open("w") as handle:
        handle.write("# VideoSeal 共谋攻击：多副本平均\n\n")
        handle.write(f"- 视频：{args.frames} 帧 512x512；每副本一条独立 256-bit 消息\n\n")
        handle.write("| 副本数 | BA 均值 | BA 最小 | BA 最大 | PSNR | SSIM |\n")
        handle.write("| ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for row in rows:
            handle.write(
                f"| {row['copies']} | {row['ba_mean']:.3f} | {row['ba_min']:.3f} | "
                f"{row['ba_max']:.3f} | {row['psnr_vs_cover']:.2f} | "
                f"{row['ssim_vs_cover']:.4f} |\n"
            )

    print(f"\nCSV: {csv_path}\nMarkdown: {md_path}")


if __name__ == "__main__":
    main()
