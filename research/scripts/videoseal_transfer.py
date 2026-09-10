#!/usr/bin/env python3
"""扩散净化与视频时序攻击对 VideoSeal 的迁移验证（研究用途）。

VideoSeal（Meta，256-bit，公开权重）是视频水印方案。本脚本测试：
- 图像：扩散净化 strength 扫描
- 视频：逐帧扩散净化、时序聚合、帧序扰动（shuffle/drop/reverse）、双副本共谋

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
from diffusers import AutoPipelineForImage2Image
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[2]
VSEAL_REPO = ROOT / "research" / "vendor" / "videoseal"
sys.path.insert(0, str(ROOT / "research" / "scripts"))
sys.path.insert(0, str(VSEAL_REPO))
os.chdir(VSEAL_REPO)

import videoseal  # noqa: E402

from baseline_trustmark import synthetic_image  # noqa: E402


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    return transforms.ToPILImage()(tensor.clamp(0.0, 1.0).cpu())


def pil_to_tensor(img: Image.Image, device: str) -> torch.Tensor:
    return transforms.ToTensor()(img).to(device)


def image_bits(model, img_tensor: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        preds = model.detect(img_tensor, is_video=False)["preds"]
    return (preds[:, 1:] > 0).float()


def video_bits(model, video: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        preds = model.detect(video, is_video=True)["preds"]
    per_frame = (preds[:, 1:] > 0).float()  # f k
    aggregated = (preds[:, 1:].mean(dim=0) > 0).float().unsqueeze(0)  # 1 k
    return per_frame, aggregated


def make_video(frames: int = 16, size: int = 512) -> torch.Tensor:
    out = []
    yy, xx = np.mgrid[0:size, 0:size]
    for t in range(frames):
        arr = np.zeros((size, size, 3), dtype=np.uint8)
        arr[..., 0] = (60 + 120 * xx / size).astype(np.uint8)
        arr[..., 1] = (50 + 100 * yy / size).astype(np.uint8)
        arr[..., 2] = 140
        x0 = 30 + t * (size - 180) // max(frames - 1, 1)
        arr[size // 3 : size // 3 + 150, x0 : x0 + 120] = [230, 90, 60]
        out.append(arr)
    video = torch.from_numpy(np.stack(out)).permute(0, 3, 1, 2).float() / 255.0
    return video


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=int, default=4)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument(
        "--out", type=Path, default=ROOT / "research" / "output" / "videoseal_transfer"
    )
    parser.add_argument("--model", default="stabilityai/sd-turbo")
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = videoseal.load("videoseal").to(device).eval()
    dtype = torch.float16 if device == "mps" else torch.float32
    pipe = AutoPipelineForImage2Image.from_pretrained(
        args.model, torch_dtype=dtype, variant="fp16" if dtype == torch.float16 else None
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    def diffuse(img: Image.Image, strength: float, seed: int) -> Image.Image:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        return pipe(
            prompt="",
            image=img,
            strength=strength,
            guidance_scale=0.0,
            num_inference_steps=args.steps,
            generator=generator,
        ).images[0]

    rows: list[dict[str, object]] = []

    # ---------- 图像 ----------
    for image_id in range(args.images):
        cover = synthetic_image(image_id)
        cover_t = transforms.ToTensor()(cover).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model.embed(cover_t, is_video=False)
        message = out["msgs"].to(device)
        watermarked = tensor_to_pil(out["imgs_w"][0])
        clean_ba = float((image_bits(model, out["imgs_w"]) == message).float().mean())
        print(f"[image {image_id}] clean BA={clean_ba:.3f}")
        for strength in (0.15, 0.25, 0.35, 0.45, 0.55, 0.65):
            attacked = diffuse(watermarked, strength, 20260909 + image_id)
            attacked_t = pil_to_tensor(attacked, device).unsqueeze(0)
            ba = float((image_bits(model, attacked_t) == message).float().mean())
            rows.append(
                {
                    "kind": "image",
                    "sample": image_id,
                    "attack": f"sd_turbo_s{strength:.2f}",
                    "ba": ba,
                    "psnr": psnr_metric(
                        np.asarray(cover).astype(float),
                        np.asarray(attacked).astype(float),
                        data_range=255,
                    ),
                    "ssim": ssim_metric(
                        np.asarray(cover).astype(float),
                        np.asarray(attacked).astype(float),
                        channel_axis=2,
                        data_range=255,
                    ),
                }
            )

    # ---------- 视频 ----------
    cover_video = make_video(args.frames).to(device)
    with torch.no_grad():
        vout = model.embed(cover_video, is_video=True)
    message = vout["msgs"][0:1].to(device)
    watermarked_video = vout["imgs_w"]
    clean_per_frame, clean_agg = video_bits(model, watermarked_video)
    print(
        f"[video] clean per-frame BA="
        f"{float((clean_per_frame == message).float().mean()):.3f} "
        f"aggregated BA={float((clean_agg == message).float().mean()):.3f}"
    )

    def record_video(name: str, attacked: torch.Tensor) -> None:
        per_frame, aggregated = video_bits(model, attacked)
        per_frame_ba = float(
            (per_frame == message).float().mean()
        )
        agg_ba = float((aggregated == message).float().mean())
        cover_np = cover_video.cpu().numpy()
        att_np = attacked.cpu().numpy()
        n = min(len(cover_np), len(att_np))
        rows.append(
            {
                "kind": "video",
                "sample": 0,
                "attack": name,
                "ba": agg_ba,
                "per_frame_ba": per_frame_ba,
                "psnr": float(
                    np.mean(
                        [
                            psnr_metric(cover_np[i], att_np[i], data_range=1.0)
                            for i in range(n)
                        ]
                    )
                ),
                "ssim": float(
                    np.mean(
                        [
                            ssim_metric(
                                cover_np[i].transpose(1, 2, 0),
                                att_np[i].transpose(1, 2, 0),
                                channel_axis=2,
                                data_range=1.0,
                            )
                            for i in range(n)
                        ]
                    )
                ),
            }
        )

    record_video("identity", watermarked_video)

    for strength in (0.15, 0.25, 0.35):
        frames = [
            diffuse(
                tensor_to_pil(watermarked_video[i]),
                strength,
                20260909 + i,
            )
            for i in range(len(watermarked_video))
        ]
        attacked = torch.stack([pil_to_tensor(f, device) for f in frames])
        record_video(f"per_frame_sd_s{strength:.2f}", attacked)

    # 相关噪声：所有帧用同一 seed
    frames = [
        diffuse(tensor_to_pil(watermarked_video[i]), 0.25, 20260909)
        for i in range(len(watermarked_video))
    ]
    record_video(
        "correlated_sd_s0.25",
        torch.stack([pil_to_tensor(f, device) for f in frames]),
    )

    # 帧序扰动
    order = np.random.default_rng(20260909).permutation(len(watermarked_video))
    record_video(
        "frame_shuffle",
        watermarked_video[torch.from_numpy(order).to(device)],
    )
    record_video("frame_drop_50", watermarked_video[::2])
    record_video(
        "frame_reverse",
        watermarked_video[torch.arange(len(watermarked_video) - 1, -1, -1).to(device)],
    )

    # 双副本共谋：同一内容嵌入两条不同消息后平均
    torch.manual_seed(20260909)
    msg_a = torch.randint(0, 2, (1, 256)).float().to(device)
    msg_b = torch.randint(0, 2, (1, 256)).float().to(device)
    with torch.no_grad():
        wm_a = model.embed(cover_video, msgs=msg_a, is_video=True)["imgs_w"]
        wm_b = model.embed(cover_video, msgs=msg_b, is_video=True)["imgs_w"]
    colluded = (wm_a + wm_b) / 2.0
    per_frame, aggregated = video_bits(model, colluded)
    rows.append(
        {
            "kind": "video",
            "sample": 0,
            "attack": "collusion_2_avg",
            "ba": float((aggregated == msg_a).float().mean()),
            "per_frame_ba": float((per_frame == msg_a).float().mean()),
            "ba_msg_b": float((aggregated == msg_b).float().mean()),
            "psnr": float("nan"),
            "ssim": float("nan"),
        }
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.out.with_suffix(".csv")
    with csv_path.open("w", newline="") as handle:
        fieldnames = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    md_path = args.out.with_suffix(".md")
    with md_path.open("w") as handle:
        handle.write("# VideoSeal 迁移验证（研究基线）\n\n")
        handle.write(
            f"- 图像样本：{args.images} 张；视频：{args.frames} 帧 512x512\n"
        )
        handle.write(f"- 攻击：`{args.model}` img2img，{args.steps} steps\n")
        handle.write("- BA=比特准确率（0.5 为随机猜测）\n\n")
        handle.write("| 类型 | 攻击 | BA | 逐帧 BA | PSNR | SSIM |\n")
        handle.write("| --- | --- | ---: | ---: | ---: | ---: |\n")
        for row in rows:
            handle.write(
                f"| {row['kind']} | {row['attack']} | {row['ba']:.3f} | "
                f"{row.get('per_frame_ba', float('nan')):.3f} | "
                f"{row['psnr']:.2f} | {row['ssim']:.4f} |\n"
            )

    print(f"\nCSV: {csv_path}\nMarkdown: {md_path}")
    for row in rows:
        print(
            f"{row['kind']:5s} {row['attack']:24s} BA={row['ba']:.3f} "
            f"per_frame={row.get('per_frame_ba', float('nan')):.3f}"
        )


if __name__ == "__main__":
    main()
