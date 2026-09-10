#!/usr/bin/env python3
"""强档画质对比：原帧 / 现行深度档（扩散512）/ 现行强力档（扩散720）/
潜空间@128 / 潜空间@256+细节回注 1.0。

用法：
    backend/.venv/bin/python research/scripts/compare_strong_visual.py --content real

输出上下两行拼图：第一行取中部纹理带（建筑、纹理），第二行取字幕带，
五列依次为 原帧、扩散深度档、扩散强力档、潜空间@128、潜空间@256（细节 1.0）；
同时打印各自相对原帧的 PSNR/SSIM（候选与原帧的差异即攻击痕迹，越高越保真）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
for extra in (ROOT / "research" / "scripts", ROOT / "backend" / "src"):
    sys.path.insert(0, str(extra))

from skimage.metrics import peak_signal_noise_ratio as psnr_metric  # noqa: E402
from skimage.metrics import structural_similarity as ssim_metric  # noqa: E402

from cthulhu_backend.transform import purify  # noqa: E402


def _load_frames(content: str, frames: int) -> np.ndarray:
    if content == "real":
        from purify_speed_tiers import read_frames

        return read_frames(ROOT / "research" / "data" / "real_clip_512.mp4", frames)
    from purify_speed_tiers import vertical_kenburns

    return vertical_kenburns(frames)


def _fidelity(label: str, candidate: np.ndarray, source: np.ndarray) -> None:
    length = min(len(source), len(candidate))
    ref = source[:length].astype(np.float64)
    atk = candidate[:length].astype(np.float64)
    psnr = psnr_metric(ref, atk, data_range=255)
    ssim = float(
        np.mean(
            [
                ssim_metric(ref[i], atk[i], channel_axis=2, data_range=255)
                for i in range(length)
            ]
        )
    )
    print(f"  {label:<12} 对原帧 PSNR={psnr:5.2f}dB  SSIM={ssim:.4f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--content", default="real", choices=["real", "kenburns"])
    parser.add_argument("--frames", type=int, default=6)
    parser.add_argument("--pick", type=int, default=0, help="用于拼图的帧号")
    parser.add_argument("--latent-edge", type=int, default=256)
    parser.add_argument("--latent-sigma", type=float, default=1.5)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "research" / "output" / "strong_visual_compare.png",
    )
    args = parser.parse_args()

    frames = _load_frames(args.content, args.frames)
    height, width = frames.shape[1], frames.shape[2]
    print(f"内容={args.content} {width}x{height} 帧数={len(frames)}", flush=True)

    latent128 = purify.purify_frames(
        frames, strength=0.10, engine="latent", max_edge=128, batch=8, detail=0.5
    )
    latent256 = purify.purify_frames(
        frames,
        strength=0.10,
        engine="latent",
        max_edge=args.latent_edge,
        batch=8,
        detail=1.0,
        detail_sigma=args.latent_sigma,
    )
    # 现行「深度净化」预设：512 / strength0.15 / 20 步 / 细节 0.5。
    deep = purify.purify_frames(
        frames,
        strength=0.15,
        steps=20,
        detail=0.5,
        max_edge=512,
        batch=4,
    )
    # 现行「强力净化」预设：720 / strength0.35 / 30 步 / 细节 0 / 时序 0.5。
    strong = purify.purify_frames(
        frames,
        strength=0.35,
        steps=30,
        detail=0.0,
        temporal_strength=0.5,
        max_edge=720,
        batch=2,
    )

    for label, candidate in (
        ("扩散深度档@512", deep),
        ("扩散强力档@720", strong),
        ("潜空间@128", latent128),
        (f"潜空间@{args.latent_edge}+细节1.0/σ{args.latent_sigma}", latent256),
    ):
        _fidelity(label, candidate, frames)

    band_top = slice(int(height * 0.05), int(height * 0.45))
    band_bottom = slice(int(height * 0.78), height)

    def row(band: slice) -> np.ndarray:
        tiles = [
            frames[args.pick][band],
            deep[args.pick][band],
            strong[args.pick][band],
            latent128[args.pick][band],
            latent256[args.pick][band],
        ]
        return np.concatenate(tiles, axis=1)

    strip = np.concatenate([row(band_top), row(band_bottom)], axis=0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(strip).save(args.out)
    print(
        f"对比图（原帧 | 扩散深度 | 扩散强力 | 潜空间@128 | 潜空间@256）：{args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
