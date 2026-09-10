#!/usr/bin/env python3
"""扩散净化 + 转码组合攻击（自然内容视频复核，研究用途）。

使用 VideoSeal 仓库自带的自然图像（MIT 许可）生成 Ken Burns 运镜视频，
测试：单独转码、逐帧扩散净化、以及净化后接 H.264 转码的组合攻击。

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

from video_codec_attacks import read_video, run_ffmpeg, write_video  # noqa: E402


def make_kenburns_video(path: Path, frames: int = 16, size: int = 512) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    w, h = image.size
    out = []
    for t in range(frames):
        frac = t / max(frames - 1, 1)
        crop_w = int(w * (0.55 - 0.15 * frac))
        crop_h = int(h * (0.55 - 0.15 * frac))
        x0 = int((w - crop_w) * frac)
        y0 = int((h - crop_h) * (0.5 - 0.5 * frac))
        crop = image.crop((x0, y0, x0 + crop_w, y0 + crop_h)).resize(
            (size, size), Image.BICUBIC
        )
        out.append(np.asarray(crop))
    video = torch.from_numpy(np.stack(out)).permute(0, 3, 1, 2).float() / 255.0
    return video


def to_pil(tensor: torch.Tensor) -> Image.Image:
    return transforms.ToPILImage()(tensor.clamp(0.0, 1.0).cpu())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "research" / "output" / "diffusion_natural_video",
    )
    parser.add_argument("--model", default="stabilityai/sd-turbo")
    parser.add_argument("--model-card", default="videoseal")
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = videoseal.load(args.model_card).to(device).eval()
    dtype = torch.float16 if device == "mps" else torch.float32
    pipe = AutoPipelineForImage2Image.from_pretrained(
        args.model, torch_dtype=dtype, variant="fp16" if dtype == torch.float16 else None
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    cover = make_kenburns_video(
        VSEAL_REPO / "assets" / "imgs" / "1.jpg", args.frames
    ).to(device)
    with torch.no_grad():
        out = model.embed(cover, is_video=True)
    message = out["msgs"][0:1].float().to(device)
    watermarked = out["imgs_w"]
    wm_np = (
        (watermarked.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255)
        .round()
        .astype(np.uint8)
    )

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

    def frames_to_np(frames: list[Image.Image]) -> np.ndarray:
        return (
            torch.stack([transforms.ToTensor()(f) for f in frames])
            .clamp(0, 1)
            .permute(0, 2, 3, 1)
            .numpy()
            * 255
        ).round().astype(np.uint8)

    rows: list[dict[str, object]] = []

    def evaluate(name: str, attacked_np: np.ndarray) -> None:
        attacked_t = (
            torch.from_numpy(attacked_np.astype(np.float32) / 255.0)
            .permute(0, 3, 1, 2)
            .to(device)
        )
        with torch.no_grad():
            preds = model.detect(attacked_t, is_video=True)["preds"]
        aggregated = (preds[:, 1:].mean(dim=0) > 0).float()
        per_frame = (preds[:, 1:] > 0).float()
        ba = float((aggregated == message).float().mean())
        per_frame_ba = float((per_frame == message).float().mean())
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
        rows.append(
            {
                "attack": name,
                "ba": ba,
                "per_frame_ba": per_frame_ba,
                "psnr_vs_watermarked": psnr,
                "ssim_vs_watermarked": ssim,
            }
        )
        print(
            f"{name:28s} BA={ba:.3f} per_frame={per_frame_ba:.3f} "
            f"PSNR={psnr:5.2f} SSIM={ssim:.4f}"
        )

    evaluate("identity", wm_np.copy())

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        lossless = tmpdir / "watermarked.mkv"
        write_video(wm_np, lossless, codec="ffv1")
        h264 = tmpdir / "h264_crf23.mp4"
        run_ffmpeg(
            lossless,
            h264,
            ["-c:v", "libx264", "-crf", "23", "-preset", "medium", "-pix_fmt", "yuv420p"],
        )
        evaluate("h264_crf23_only", read_video(h264, size=512))

        for strength in (0.15, 0.25):
            frames = [
                diffuse(to_pil(watermarked[i]), strength, 20260909 + i)
                for i in range(len(watermarked))
            ]
            purified_np = frames_to_np(frames)
            evaluate(f"per_frame_s{strength:.2f}", purified_np)

            if strength == 0.15:
                combined = tmpdir / "purified_h264.mp4"
                write_video(purified_np, combined.with_suffix(".mkv"), codec="ffv1")
                run_ffmpeg(
                    combined.with_suffix(".mkv"),
                    combined,
                    [
                        "-c:v",
                        "libx264",
                        "-crf",
                        "23",
                        "-preset",
                        "medium",
                        "-pix_fmt",
                        "yuv420p",
                    ],
                )
                evaluate("purified_s0.15_h264crf23", read_video(combined, size=512))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.out.with_suffix(".csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    md_path = args.out.with_suffix(".md")
    with md_path.open("w") as handle:
        handle.write(
            f"# 扩散净化 + 转码组合攻击（自然内容视频，{args.model_card}）\n\n"
        )
        handle.write(
            f"- 视频：{args.frames} 帧 512x512，VideoSeal 自带自然图像 Ken Burns 运镜\n"
        )
        handle.write(f"- 攻击：`{args.model}` img2img，{args.steps} steps\n\n")
        handle.write("| 攻击 | 聚合 BA | 逐帧 BA | PSNR | SSIM |\n")
        handle.write("| --- | ---: | ---: | ---: | ---: |\n")
        for row in rows:
            handle.write(
                f"| {row['attack']} | {row['ba']:.3f} | {row['per_frame_ba']:.3f} | "
                f"{row['psnr_vs_watermarked']:.2f} | "
                f"{row['ssim_vs_watermarked']:.4f} |\n"
            )
    print(f"\nCSV: {csv_path}\nMarkdown: {md_path}")


if __name__ == "__main__":
    main()
