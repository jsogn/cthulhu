#!/usr/bin/env python3
"""净化性能拆解：把 s/帧 拆成 VAE 编解码、UNet 步、前后处理与编码。

回答「物理天花板在哪」：如果 VAE 编解码占大头，就换小 VAE / 少走 VAE；
如果 UNet 占大头，就只能在分辨率/步数/模型结构上想办法。

用法：
    backend/.venv/bin/python research/scripts/profile_purify_breakdown.py \
        --edges 256 192 160 128 --batch 8 --steps 2
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend" / "src"))

from cthulhu_backend.transform import purify  # noqa: E402


def _sync() -> None:
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


def timed(fn, repeats: int = 3) -> float:
    """返回中位耗时（秒）；MPS 异步执行，必须在读表前后各同步一次。"""
    samples = []
    for _ in range(repeats):
        _sync()
        start = time.perf_counter()
        fn()
        _sync()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edges", nargs="+", type=int, default=[256, 192, 128])
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    pipe = purify._load_pipe(True)
    device = pipe._execution_device
    dtype = pipe.unet.dtype
    vae = pipe.vae
    unet = pipe.unet
    print(f"device={device} dtype={dtype} batch={args.batch} unet_steps={args.steps}")

    text = pipe.encode_prompt("", device, 1, do_classifier_free_guidance=False)
    prompt_embeds = text[0] if isinstance(text, tuple) else text
    prompt_embeds = prompt_embeds.repeat(args.batch, 1, 1).to(dtype)
    latent_channels = unet.config.in_channels

    # 预热：MPS 首次 kernel 编译很慢，不计入。
    warm = torch.zeros(1, 3, 256, 256, dtype=dtype, device=device)
    with torch.inference_mode():
        vae.encode(warm).latent_dist.sample()

    rows = []
    for edge in args.edges:
        size = edge - edge % 8
        image = torch.randn(args.batch, 3, size, size, dtype=dtype, device=device)
        latent = size // 8
        noise = torch.randn(args.batch, latent_channels, latent, latent, dtype=dtype, device=device)
        timestep = torch.full((args.batch,), 500.0, dtype=dtype, device=device)
        vae_scale = float(getattr(vae.config, "scaling_factor", 0.18215))

        # 每个尺寸先跑一遍：MPS 首次遇到新形状要编译 kernel，不能计入。
        with torch.inference_mode():
            vae.encode(image).latent_dist.sample()
            vae.decode(torch.randn(args.batch, latent_channels, latent, latent, dtype=dtype, device=device)).sample
            unet(noise, timestep, encoder_hidden_states=prompt_embeds, return_dict=False)
        _sync()

        encode_s = timed(lambda: vae.encode(image).latent_dist.sample(), args.repeats)
        with torch.inference_mode():
            latents = vae.encode(image).latent_dist.sample() * vae_scale
        unet_s = timed(
            lambda: unet(noise, timestep, encoder_hidden_states=prompt_embeds, return_dict=False),
            args.repeats,
        )
        decode_s = timed(lambda: vae.decode(latents / vae_scale).sample, args.repeats)

        per_image = (encode_s + decode_s + args.steps * unet_s) / args.batch
        rows.append(
            {
                "edge": size,
                "vae_encode": encode_s,
                "unet_step": unet_s,
                "vae_decode": decode_s,
                "core_per_frame": per_image,
            }
        )
        print(
            f"{size:4d}: VAE编码 {encode_s * 1000:7.1f}ms  单步UNet {unet_s * 1000:7.1f}ms  "
            f"VAE解码 {decode_s * 1000:7.1f}ms  → 每帧核心 {per_image * 1000:6.1f}ms "
            f"(UNet {args.steps} 步占 {args.steps * unet_s / (encode_s + decode_s + args.steps * unet_s) * 100:.0f}%)",
            flush=True,
        )

    # 前后处理成本：PIL 缩放 + 细节回注（按 720x1280 原图核算）。
    frames = np.random.default_rng(0).integers(
        0, 255, size=(args.batch, 1280, 720, 3), dtype=np.uint8
    )
    prep_s = timed(
        lambda: [
            purify._fit_max_edge(
                __import__("PIL.Image", fromlist=["Image"]).fromarray(frame), 256
            )
            for frame in frames
        ],
        args.repeats,
    )
    detail_s = timed(
        lambda: [
            purify._reinject_detail(
                frame.astype(np.float32) / 255.0,
                frame,
                strength=0.5,
                sigma=0.6,
                is_u8=True,
                gray=False,
            )
            for frame in frames
        ],
        args.repeats,
    )
    print(
        f"前后处理（720x1280，逐帧）：PIL 缩放 {prep_s / args.batch * 1000:6.1f}ms  "
        f"细节回注 {detail_s / args.batch * 1000:6.1f}ms"
    )


if __name__ == "__main__":
    main()
