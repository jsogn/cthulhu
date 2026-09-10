#!/usr/bin/env python3
"""视频帧插值攻击（研究用途）。

用 ffmpeg minterpolate 做运动补偿帧插值，再抽帧回到原帧率或用插值帧替换
原帧，测试时间域水印是否被插值平均掉。对比普通丢帧/重复帧（已测无效）。

边界：自有/授权素材或生成素材、公开预训练模型、本地离线实验，不接触
任何平台线上系统。
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
    parser.add_argument("--input-video", type=Path, default=None)
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "research" / "output" / "frame_interpolation_attack",
    )
    args = parser.parse_args()
    if args.input_video is not None and not args.input_video.is_absolute():
        args.input_video = ROOT / args.input_video

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = videoseal.load(args.model_card).to(device).eval()
    if args.input_video is not None:
        frames_np = read_video(args.input_video, size=512)
        cover = (
            torch.from_numpy(frames_np.astype(np.float32) / 255.0)
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
    width = watermarked.shape[-1]
    wm_np = (
        (watermarked.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255)
        .round()
        .astype(np.uint8)
    )

    rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        lossless = tmpdir / "watermarked.mkv"
        write_video(wm_np, lossless, codec="ffv1")
        reference = tmpdir / "reference.mkv"
        write_video(wm_np, reference, codec="ffv1")

        def evaluate(name: str, extra: list[str], align_step: int = 1) -> None:
            output = tmpdir / f"{name}.mp4"
            run_ffmpeg(lossless, output, extra)
            decoded = read_video(output, size=width)
            decoded_t = (
                torch.from_numpy(decoded.astype(np.float32) / 255.0)
                .permute(0, 3, 1, 2)
                .to(device)
            )
            with torch.no_grad():
                logits = detector_logits(model, decoded_t)
            ba, per_frame = aggregate_ba(logits, message)
            ref = wm_np[::align_step]
            n = min(len(ref), len(decoded))
            row: dict[str, object] = {
                "attack": name,
                "ba": ba,
                "per_frame_ba": per_frame,
                "frames": len(decoded),
                "psnr": float(
                    np.mean(
                        [
                            psnr_metric(ref[i], decoded[i], data_range=255)
                            for i in range(n)
                        ]
                    )
                ),
                "ssim": float(
                    np.mean(
                        [
                            ssim_metric(
                                ref[i], decoded[i], channel_axis=2, data_range=255
                            )
                            for i in range(n)
                        ]
                    )
                ),
            }
            if align_step == 1:
                row["vmaf"] = vmaf_score(str(output), str(reference), subsample=2)
            rows.append(row)
            vmaf_value = row.get("vmaf")
            vmaf_text = f"{vmaf_value:.2f}" if vmaf_value is not None else "nan"
            print(
                f"{name:28s} BA={ba:.3f} per_frame={per_frame:.3f} "
                f"frames={len(decoded):3d} PSNR={row['psnr']:5.2f} "
                f"SSIM={row['ssim']:.4f} VMAF={vmaf_text}"
            )

        h264 = ["-c:v", "libx264", "-preset", "medium", "-pix_fmt", "yuv420p"]
        interp = (
            "minterpolate=fps={fps}:mi_mode=mci:mc_mode=aobmc:"
            "me_mode=bidir:vsbmc=1"
        )

        evaluate("identity", ["-crf", "18", *h264])
        evaluate(
            "interp60_decimate30",
            ["-vf", f"{interp.format(fps=60)},fps=30", "-crf", "18", *h264],
            align_step=1,
        )
        evaluate(
            "interp120_decimate30",
            ["-vf", f"{interp.format(fps=120)},fps=30", "-crf", "18", *h264],
            align_step=1,
        )
        evaluate(
            "interp60_odd_frames",
            [
                "-vf",
                f"{interp.format(fps=60)},select='eq(mod(n,2),1)',setpts=N/30/TB",
                "-crf",
                "18",
                *h264,
            ],
            align_step=1,
        )
        evaluate(
            "interp60_decimate30_h264crf23",
            [
                "-vf",
                f"{interp.format(fps=60)},fps=30",
                "-crf",
                "23",
                *h264,
            ],
            align_step=1,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.out.with_suffix(".csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({k for r in rows for k in r}))
        writer.writeheader()
        writer.writerows(rows)
    md_path = args.out.with_suffix(".md")
    with md_path.open("w") as handle:
        handle.write("# 视频帧插值攻击\n\n")
        handle.write(
            f"- 模型：`{args.model_card}`；视频：{args.frames} 帧 512x512；"
            f"载体：{'真实片段' if args.input_video else 'Ken Burns'}\n"
        )
        handle.write("- 攻击：ffmpeg minterpolate 运动补偿插值 + 抽帧/替换\n\n")
        handle.write("| 攻击 | BA | 逐帧 BA | 输出帧数 | PSNR | SSIM | VMAF |\n")
        handle.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for row in rows:
            vmaf_value = row.get("vmaf")
            vmaf_text = f"{vmaf_value:.2f}" if vmaf_value is not None else "nan"
            handle.write(
                f"| {row['attack']} | {row['ba']:.3f} | {row['per_frame_ba']:.3f} | "
                f"{row['frames']} | {row['psnr']:.2f} | {row['ssim']:.4f} | "
                f"{vmaf_text} |\n"
            )
    print(f"\nCSV: {csv_path}\nMarkdown: {md_path}")


if __name__ == "__main__":
    main()
