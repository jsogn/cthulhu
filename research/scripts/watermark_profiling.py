#!/usr/bin/env python3
"""水印嵌入域画像（研究用途）。

对公开方案直接做“干净 vs 带水印”差分，分析：
- 残差的通道分布（Y/Cb/Cr）
- 8x8 DCT 子带能量分布（水印集中在哪些频带）
- 视频帧间残差相关性（水印是否跨帧一致）
- WAM 的局部水印在 mask 内/外的残差分布

这是文本 Phase 1 侦察的公开方案版本：不需要生产系统，直接对公开模型做。
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.fftpack import dctn

ROOT = Path(__file__).resolve().parents[2]
VSEAL_REPO = ROOT / "research" / "vendor" / "videoseal"
WAM_REPO = ROOT / "research" / "vendor" / "watermark-anything"
sys.path.insert(0, str(ROOT / "research" / "scripts"))
sys.path.insert(0, str(VSEAL_REPO))
sys.path.insert(0, str(WAM_REPO))
sys.path.insert(0, str(ROOT / "backend" / "src"))
os.chdir(VSEAL_REPO)


def rgb_to_ycbcr(arr: np.ndarray) -> np.ndarray:
    """arr: (...,3) in [0,1] -> (...,3) YCbCr。"""
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5
    return np.stack([y, cb, cr], axis=-1)


def dct_band_energy(y_residual: np.ndarray) -> np.ndarray:
    """对 (H,W) 残差做 8x8 DCT，返回 64 个频带的归一化能量。"""
    h, w = y_residual.shape
    h8, w8 = h // 8, w // 8
    blocks = y_residual[: h8 * 8, : w8 * 8].reshape(h8, 8, w8, 8)
    coeffs = dctn(blocks, axes=(1, 3), norm="ortho")
    energy = (coeffs**2).mean(axis=(0, 2))
    energy = energy.reshape(64)
    total = energy.sum() + 1e-12
    return energy / total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheme", choices=["videoseal", "wam"], default="videoseal")
    parser.add_argument("--input-video", type=Path, default=None)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument(
        "--out", type=Path, default=ROOT / "research" / "output" / "watermark_profiling"
    )
    args = parser.parse_args()
    if args.input_video is not None and not args.input_video.is_absolute():
        args.input_video = ROOT / args.input_video

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    rows: list[dict[str, object]] = []
    if args.scheme == "videoseal":
        import videoseal
        from diffusion_natural_video import make_kenburns_video
        from video_codec_attacks import read_video

        model = videoseal.load("videoseal").to(device).eval()
        if args.input_video is not None:
            frames_np = read_video(args.input_video, size=512)
            cover = (
                torch.from_numpy(frames_np.astype(np.float32) / 255.0)
                .permute(0, 3, 1, 2)
                .to(device)
            )[: args.frames]
        else:
            cover = make_kenburns_video(
                VSEAL_REPO / "assets" / "imgs" / "1.jpg", args.frames
            ).to(device)
        with torch.no_grad():
            watermarked = model.embed(cover, is_video=True)["imgs_w"]
        clean_np = cover.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
        wm_np = watermarked.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
        residual = wm_np - clean_np  # (F,H,W,3)
        label = "videoseal"
    else:
        os.chdir(WAM_REPO)
        from notebooks.inference_utils import (
            create_random_mask,
            default_transform,
            load_model_from_checkpoint,
            unnormalize_img,
        )

        wam = (
            load_model_from_checkpoint(
                str(WAM_REPO / "checkpoints" / "params.json"),
                str(WAM_REPO / "checkpoints" / "wam_mit.pth"),
            )
            .to(device)
            .eval()
        )
        from PIL import Image

        source = Image.open(VSEAL_REPO / "assets" / "imgs" / "1.jpg").convert("RGB")
        source = source.resize((512, 512), Image.BICUBIC)
        img_pt = default_transform(source).unsqueeze(0).to(device)
        torch.manual_seed(20260909)
        message = torch.randint(0, 2, (1, 32)).float().to(device)
        with torch.no_grad():
            embedded = wam.embed(img_pt, message)["imgs_w"]
        mask = create_random_mask(img_pt, num_masks=1, mask_percentage=0.5).to(device)
        watermarked = embedded * mask + img_pt * (1 - mask)
        clean_np = (
            unnormalize_img(img_pt).clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
        )
        wm_np = (
            unnormalize_img(watermarked)
            .clamp(0, 1)
            .squeeze(0)
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )
        residual = (wm_np - clean_np)[None, ...]
        mask_np = mask.squeeze().cpu().numpy() > 0.5
        label = "wam"

    ycbcr_clean = rgb_to_ycbcr(clean_np)
    ycbcr_wm = rgb_to_ycbcr(wm_np)
    y_residual = ycbcr_wm[..., 0] - ycbcr_clean[..., 0]
    cb_residual = ycbcr_wm[..., 1] - ycbcr_clean[..., 1]
    cr_residual = ycbcr_wm[..., 2] - ycbcr_clean[..., 2]
    if y_residual.ndim == 2:
        y_residual = y_residual[None, ...]
    channel_energy = {
        "Y": float(np.mean(np.abs(y_residual))),
        "Cb": float(np.mean(np.abs(cb_residual))),
        "Cr": float(np.mean(np.abs(cr_residual))),
    }

    band_energies = []
    for frame in y_residual:
        band_energies.append(dct_band_energy(frame))
    band_mean = np.mean(band_energies, axis=0)
    top_bands = np.argsort(band_mean)[::-1][:10]
    low = band_mean[:16].sum()
    mid = band_mean[16:48].sum()
    high = band_mean[48:].sum()

    temporal_corr = float("nan")
    if len(y_residual) > 1:
        flat = y_residual.reshape(len(y_residual), -1)
        corrs = []
        for i in range(len(flat) - 1):
            a, b = flat[i], flat[i + 1]
            denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-12
            corrs.append(float(np.dot(a, b) / denom))
        temporal_corr = float(np.mean(corrs))

    rows.append(
        {
            "scheme": label,
            "frames": len(y_residual),
            "channel_Y": channel_energy["Y"],
            "channel_Cb": channel_energy["Cb"],
            "channel_Cr": channel_energy["Cr"],
            "dct_low_0_15": float(low),
            "dct_mid_16_47": float(mid),
            "dct_high_48_63": float(high),
            "temporal_corr": temporal_corr,
            "top_bands": " ".join(str(int(b)) for b in top_bands),
        }
    )

    if label == "wam":
        chroma_mag = np.sqrt(cb_residual**2 + cr_residual**2)
        if chroma_mag.ndim == 2:
            chroma_mag = chroma_mag[None, ...]
        inside = float(np.mean(chroma_mag[0][mask_np]))
        outside = float(np.mean(chroma_mag[0][~mask_np]))
        rows[0]["mask_inside_chroma"] = inside
        rows[0]["mask_outside_chroma"] = outside

    args.out.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.out.with_suffix(".csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({k for r in rows for k in r}))
        writer.writeheader()
        writer.writerows(rows)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(band_mean.reshape(8, 8), cmap="viridis")
        ax.set_title(f"{label}: 8x8 DCT band energy")
        ax.set_xlabel("u")
        ax.set_ylabel("v")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        png_path = args.out.with_suffix(".png")
        fig.savefig(png_path, dpi=160)
        plt.close(fig)
        print(f"heatmap: {png_path}")
    except Exception as exc:
        print(f"heatmap skipped: {exc}")

    md_path = args.out.with_suffix(".md")
    with md_path.open("w") as handle:
        handle.write(f"# 水印嵌入域画像：{label}\n\n")
        handle.write(f"- 帧数：{len(y_residual)}\n")
        handle.write(
            f"- 通道残差（平均绝对值）：Y {channel_energy['Y']:.6f}，"
            f"Cb {channel_energy['Cb']:.6f}，Cr {channel_energy['Cr']:.6f}\n"
        )
        handle.write(
            f"- DCT 能量占比：低 0-15 {low:.3f}，中 16-47 {mid:.3f}，"
            f"高 48-63 {high:.3f}\n"
        )
        handle.write(f"- 帧间残差相关性：{temporal_corr:.4f}\n")
        handle.write(
            "- Top-10 频带（Zigzag 无关的 8x8 行优先索引）："
            f"{' '.join(str(int(b)) for b in top_bands)}\n"
        )
        if label == "wam":
            handle.write(
                f"- mask 内色度残差 {rows[0]['mask_inside_chroma']:.6f}，"
                f"mask 外色度残差 {rows[0]['mask_outside_chroma']:.6f}\n"
            )
    print(f"CSV: {csv_path}\nMarkdown: {md_path}")
    print(rows[0])


if __name__ == "__main__":
    main()
