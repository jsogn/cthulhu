#!/usr/bin/env python3
"""黑盒净化提速候选：用微型自编码器替换 sd-turbo 的重 VAE。

动机（profile_purify_breakdown.py 实测，256/batch8/2 步，720p 输入）：

| 环节 | 每帧成本 |
| --- | ---: |
| VAE 编码 | 35ms |
| UNet ×2 步 | 70ms |
| VAE 解码 | 84ms |
| 前后处理 | 17ms |

VAE 编解码占核心的 63%，而"低分辨率重建"本身才是打水印的机制。于是候选：

- `vae_rt*`：全 VAE 往返（无 UNet）——检验 VAE 瓶颈本身够不够；
- `taesd_rt*`：TAESD（~10MB 微型 AE）往返——每帧约 10ms；
- `taesd_unet*`：TAESD 编码 → 1 步 UNet → TAESD 解码；
- `turbo256_s010`：现有产品极速档，作为对照。

指标：BA（0.5 随机）、BA@h264（再过 H.264 CRF23）、PSNR/SSIM、秒/帧。
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "research" / "scripts"
for extra in (SCRIPTS, ROOT / "backend" / "src"):
    sys.path.insert(0, str(extra))

from cthulhu_backend.transform import purify  # noqa: E402
from profile_purify_breakdown import _sync  # noqa: E402


def _scheme_helpers():
    """公开方案/EOT 相关 helper 只在跑矩阵时需要（依赖 skimage）。"""
    from purify_speed_tiers import (
        build_seal,
        build_wam_frames,
        h264_crf23,
        natural_kenburns,
        quality,
        read_frames,
        vertical_kenburns,
    )

    return {
        "build_seal": build_seal,
        "build_wam_frames": build_wam_frames,
        "h264_crf23": h264_crf23,
        "natural_kenburns": natural_kenburns,
        "quality": quality,
        "read_frames": read_frames,
        "vertical_kenburns": vertical_kenburns,
    }

def _resize_to_edge(frames: np.ndarray, edge: int) -> tuple[np.ndarray, tuple[int, int]]:
    """按长边等比缩放到 8 的倍数，返回缩放后的帧与原始 (w, h)。"""
    height, width = frames.shape[1], frames.shape[2]
    longest = max(width, height)
    if edge <= 0 or longest <= edge:
        return frames, (width, height)
    scale = edge / longest
    target_w = min(width, max(8, round(width * scale / 8) * 8))
    target_h = min(height, max(8, round(height * scale / 8) * 8))
    out = np.stack(
        [
            np.asarray(Image.fromarray(frame).resize((target_w, target_h), Image.LANCZOS))
            for frame in frames
        ]
    )
    return out, (width, height)


def _restore_size(frames: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    if frames.shape[2] == width and frames.shape[1] == height:
        return frames
    return np.stack(
        [
            np.asarray(Image.fromarray(frame).resize((width, height), Image.BICUBIC))
            for frame in frames
        ]
    )


def _cv2_resize(frames: np.ndarray, size: tuple[int, int], *, up: bool) -> np.ndarray:
    """cv2 批量缩放：比逐帧 PIL 快一个量级（研究用快路径）。"""
    import cv2

    width, height = size
    interp = cv2.INTER_CUBIC if up else cv2.INTER_AREA
    return np.stack([cv2.resize(frame, (width, height), interpolation=interp) for frame in frames])


def _cv2_highpass(frame: np.ndarray, sigma: float) -> np.ndarray:
    import cv2

    source = frame.astype(np.float32) / 255.0
    blurred = cv2.GaussianBlur(source, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return source - blurred


def _reinject(frames: np.ndarray, original: np.ndarray, strength: float) -> np.ndarray:
    if strength <= 0:
        return frames
    return np.stack(
        [
            (
                np.clip(
                    purify._reinject_detail(
                        frame.astype(np.float32) / 255.0,
                        source,
                        strength=strength,
                        sigma=0.6,
                        is_u8=True,
                        gray=False,
                    ),
                    0.0,
                    1.0,
                )
                * 255.0
            )
            .round()
            .astype(np.uint8)
            for frame, source in zip(frames, original, strict=True)
        ]
    )


def make_attacks(device: str, batch: int) -> dict[str, callable]:
    """返回 {候选名: (帧数组 -> 帧数组)}；模型懒加载，首次调用计费不计入计时。"""
    from diffusers import AutoencoderTiny

    cache: dict[str, object] = {}

    def taesd():
        if "taesd" not in cache:
            model = AutoencoderTiny.from_pretrained("madebyollin/taesd")
            cache["taesd"] = model.to(device).to(torch.float16).eval()
        return cache["taesd"]

    def to_tensor(frames: np.ndarray) -> torch.Tensor:
        return (
            torch.from_numpy(frames.astype(np.float32) / 255.0)
            .permute(0, 3, 1, 2)
            .to(device=device, dtype=torch.float16)
        )

    def from_tensor(tensor: torch.Tensor) -> np.ndarray:
        arr = (tensor.clamp(0, 1).permute(0, 2, 3, 1).float().cpu().numpy() * 255.0)
        return arr.round().astype(np.uint8)

    def autoencoder_roundtrip(
        frames: np.ndarray,
        *,
        edge: int,
        detail: float,
    ) -> np.ndarray:
        """TAESD 往返（PIL 缩放版，用于与研究快路径对照）。"""
        small, original_size = _resize_to_edge(frames, edge)
        pieces: list[np.ndarray] = []
        for start in range(0, len(small), batch):
            chunk = small[start : start + batch]
            tensor = to_tensor(chunk)
            with torch.inference_mode():
                tiny = taesd()
                # TAESD 直接输出 SD 潜空间尺度（scaling_factor=1.0）。
                decoded = tiny.decode(tiny.encode(tensor).latents).sample
            pieces.append(from_tensor(decoded))
            _sync()
        merged = np.concatenate(pieces, axis=0)
        restored = _restore_size(merged, original_size)
        return _reinject(restored, frames, detail)

    def resize_roundtrip(frames: np.ndarray, edge: int, detail: float = 0.5) -> np.ndarray:
        """纯几何瓶颈：缩到 edge 再放大回来（不经过任何模型）。"""
        small, original_size = _resize_to_edge(frames, edge)
        return _reinject(_restore_size(small, original_size), frames, detail)

    def taesd_fast(frames: np.ndarray, edge: int, detail: float = 0.5) -> np.ndarray:
        """生产化走向的实现：cv2 缩放 + torch 批张量 + TAESD，无逐帧 PIL。"""
        height, width = frames.shape[1], frames.shape[2]
        longest = max(width, height)
        scale = min(1.0, edge / longest) if edge > 0 else 1.0
        target_w = min(width, max(8, round(width * scale / 8) * 8))
        target_h = min(height, max(8, round(height * scale / 8) * 8))
        small = (
            frames
            if (target_w, target_h) == (width, height)
            else _cv2_resize(frames, (target_w, target_h), up=False)
        )
        tiny = taesd()
        outs: list[np.ndarray] = []
        for start in range(0, len(small), batch):
            chunk = small[start : start + batch]
            tensor = to_tensor(chunk)
            with torch.inference_mode():
                latents = tiny.encode(tensor).latents
                decoded = tiny.decode(latents).sample
            arr = (
                (decoded.clamp(0, 1).permute(0, 2, 3, 1).float().cpu().numpy() * 255.0)
                .round()
                .astype(np.uint8)
            )
            outs.append(arr)
            _sync()
        merged = np.concatenate(outs, axis=0)
        restored = (
            merged
            if (target_w, target_h) == (width, height)
            else _cv2_resize(merged, (width, height), up=True)
        )
        if detail <= 0:
            return restored
        out = np.empty_like(restored)
        for index, frame in enumerate(restored):
            high = _cv2_highpass(frames[index], 0.6)
            energy = float(np.mean(np.abs(high)))
            scale_detail = float(np.clip(0.35 + 0.65 * (energy / (energy + 0.02)), 0.35, 1.0))
            canvas = frame.astype(np.float32) / 255.0 + detail * scale_detail * high
            out[index] = np.clip(canvas * 255.0, 0, 255).round().astype(np.uint8)
        return out

    def latent_product(
        frames: np.ndarray,
        edge: int,
        *,
        passes: int = 1,
        detail: float = 0.5,
        sigma: float = 0.6,
        temporal: float = 0.0,
    ) -> np.ndarray:
        """产品 latent 引擎路径；passes>1 时连续多次重建（研究用强档）。"""
        out = frames
        for _ in range(max(1, passes)):
            out = purify.purify_frames(
                out,
                strength=0.10,
                max_edge=edge,
                batch=8,
                detail=detail,
                detail_sigma=sigma,
                temporal_strength=temporal,
            )
        return out

    return {
        # 产品代码路径（backend purify，latent 引擎）：与研究原型对照用。
        "product_latent128": lambda frames: latent_product(frames, 128),
        "product_latent96": lambda frames: latent_product(frames, 96),
        "product_latent128_nodetail": lambda frames: latent_product(frames, 128, detail=0.0),
        "product_latent128_t050": lambda frames: latent_product(
            frames, 128, detail=0.0, temporal=0.5
        ),
        # 保真档候选：细节回注拉满，检验能否在不掉清除率的前提下补回清晰度。
        "product_latent128_d10": lambda frames: latent_product(frames, 128, detail=1.0),
        "product_latent192_d10": lambda frames: latent_product(frames, 192, detail=1.0),
        "product_latent256_d10": lambda frames: latent_product(frames, 256, detail=1.0),
        # 频带宽度扫描：sigma 越大，回注的中频越多（字幕/纹理），水印在低频不受影响。
        "product_latent128_d10_s15": lambda frames: latent_product(
            frames, 128, detail=1.0, sigma=1.5
        ),
        "product_latent192_d10_s15": lambda frames: latent_product(
            frames, 192, detail=1.0, sigma=1.5
        ),
        "product_latent256_d10_s15": lambda frames: latent_product(
            frames, 256, detail=1.0, sigma=1.5
        ),
        "product_latent256_d10_s25": lambda frames: latent_product(
            frames, 256, detail=1.0, sigma=2.5
        ),
        "product_latent256_d10_s20": lambda frames: latent_product(
            frames, 256, detail=1.0, sigma=2.0
        ),
        # 上线预设逐一对齐（research §19.4）：极速/画质优先/强力。
        "preset_extreme_192s15": lambda frames: latent_product(
            frames, 192, detail=1.0, sigma=1.5
        ),
        "preset_quality_256s20": lambda frames: latent_product(
            frames, 256, detail=1.0, sigma=2.0
        ),
        "preset_quality_256s15": lambda frames: latent_product(
            frames, 256, detail=1.0, sigma=1.5
        ),
        "preset_strong_256s15t050": lambda frames: latent_product(
            frames, 256, detail=1.0, sigma=1.5, temporal=0.5
        ),
        "preset_strong_128s15t050": lambda frames: latent_product(
            frames, 128, detail=1.0, sigma=1.5, temporal=0.5
        ),
        "product_latent256_d08_s25": lambda frames: latent_product(
            frames, 256, detail=0.8, sigma=2.5
        ),
        # 更高边缘 + 宽带回注：能否在保住清除率的前提下进一步提升保真度。
        "product_latent320_d10_s15": lambda frames: latent_product(
            frames, 320, detail=1.0, sigma=1.5
        ),
        "product_latent384_d10_s15": lambda frames: latent_product(
            frames, 384, detail=1.0, sigma=1.5
        ),
        # 强档候选：提高潜空间重建分辨率 / 多次重建，检验能否顶掉扩散档。
        "product_latent192": lambda frames: latent_product(frames, 192),
        "product_latent256": lambda frames: latent_product(frames, 256),
        "product_latent512": lambda frames: latent_product(frames, 512),
        "product_latent128_x2": lambda frames: latent_product(frames, 128, passes=2),
        "product_latent256_x2": lambda frames: latent_product(frames, 256, passes=2),
        # 消融：只做几何瓶颈（缩放往返），不加任何模型。
        "resize256": lambda frames: resize_roundtrip(frames, 256),
        "resize192": lambda frames: resize_roundtrip(frames, 192),
        "resize128": lambda frames: resize_roundtrip(frames, 128),
        "taesd_rt256": lambda frames: autoencoder_roundtrip(
            frames, edge=256, detail=0.5
        ),
        "taesd_fast256": lambda frames: taesd_fast(frames, 256),
        "taesd_fast256_nodetail": lambda frames: taesd_fast(frames, 256, detail=0.0),
        "taesd_fast192": lambda frames: taesd_fast(frames, 192),
        "taesd_fast128": lambda frames: taesd_fast(frames, 128),
        "taesd_fast128_x2": lambda frames: taesd_fast(taesd_fast(frames, 128), 128),
        "taesd_chain192_128": lambda frames: taesd_fast(taesd_fast(frames, 192), 128),
        "taesd_fast96": lambda frames: taesd_fast(frames, 96),
        "taesd_fast128_nodetail": lambda frames: taesd_fast(frames, 128, detail=0.0),
        "taesd_rt128": lambda frames: autoencoder_roundtrip(
            frames, edge=128, detail=0.5
        ),
        "taesd_rt192": lambda frames: autoencoder_roundtrip(
            frames, edge=192, detail=0.5
        ),
        "taesd_rt256_nodetail": lambda frames: autoencoder_roundtrip(
            frames, edge=256, detail=0.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schemes", nargs="+", default=["videoseal", "pixelseal", "wam"])
    parser.add_argument("--contents", default="natural512,vertical720")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument(
        "--candidates",
        default="turbo256_s010,vae_rt256,taesd_rt256,taesd_rt128,taesd_rt256_nodetail",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "research" / "output" / "purify_fast_candidates",
    )
    args = parser.parse_args()
    if not args.out.is_absolute():
        args.out = ROOT / args.out

    helpers = _scheme_helpers()
    build_seal = helpers["build_seal"]
    build_wam_frames = helpers["build_wam_frames"]
    h264_crf23 = helpers["h264_crf23"]
    natural_kenburns = helpers["natural_kenburns"]
    quality = helpers["quality"]
    read_frames = helpers["read_frames"]
    vertical_kenburns = helpers["vertical_kenburns"]

    os.environ.setdefault("HF_HOME", str(ROOT / "research" / "models" / "hf"))
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device={device} torch={torch.__version__}", flush=True)
    warm = purify.purify_frames(
        np.zeros((2, 64, 64, 3), dtype=np.uint8),
        strength=0.1,
        max_edge=256,
        batch=2,
    )
    del warm
    attacks = make_attacks(device, args.batch)
    wanted = [name for name in args.candidates.split(",") if name]
    # 先各自跑一遍小样本预热（TAESD/UNet 首次调用含加载与 kernel 编译）。
    probe = np.zeros((args.batch, 64, 64, 3), dtype=np.uint8)
    for name in wanted:
        attacks[name](probe)
        print(f"预热完成：{name}", flush=True)

    contents: dict[str, np.ndarray] = {}
    for item in args.contents.split(","):
        item = item.strip()
        if item == "natural512":
            contents[item] = natural_kenburns(args.frames)
        elif item == "vertical720":
            contents[item] = vertical_kenburns(args.frames)
        elif item == "real512":
            contents[item] = read_frames(
                ROOT / "research" / "data" / "real_clip_512.mp4", args.frames
            )
        elif item:
            print(f"跳过未知内容：{item}")

    builders = {
        "videoseal": lambda: build_seal(device, "videoseal"),
        "pixelseal": lambda: build_seal(device, "pixelseal"),
        "wam": lambda: build_wam_frames(device),
    }

    rows: list[dict[str, object]] = []
    for scheme in args.schemes:
        builder = builders.get(scheme)
        if builder is None:
            continue
        try:
            _, embed_video, ba_video = builder()
        except Exception as exc:  # noqa: BLE001
            print(f"[{scheme}] 加载失败，跳过：{exc}", flush=True)
            continue
        print(f"[{scheme}] 已加载", flush=True)
        for content_name, frames in contents.items():
            cover = frames[: args.frames]
            watermarked, message = embed_video(cover)
            clean_ba, _ = ba_video(watermarked, message)
            for name in wanted:
                _sync()
                start = time.perf_counter()
                attacked = attacks[name](watermarked)
                _sync()
                elapsed = time.perf_counter() - start
                ba, _ = ba_video(attacked, message)
                ba_h264, _ = ba_video(h264_crf23(attacked), message)
                psnr, ssim = quality(watermarked, attacked)
                # 保真度：与水印前的干净原帧比（psnr/ssim 只反映"改了多少"，
                # 这一列才反映"改完之后还像不像原片"）。
                psnr_src, ssim_src = quality(cover, attacked)
                row = {
                    "scheme": scheme,
                    "content": content_name,
                    "candidate": name,
                    "clean_ba": round(clean_ba, 4),
                    "ba": round(ba, 4),
                    "ba_h264": round(ba_h264, 4),
                    "psnr": round(psnr, 2),
                    "ssim": round(ssim, 4),
                    "psnr_src": round(psnr_src, 2),
                    "ssim_src": round(ssim_src, 4),
                    "sec_per_frame": round(elapsed / len(cover), 4),
                }
                rows.append(row)
                print(
                    f"  {name:<22} BA={ba:.4f} BA@h264={ba_h264:.4f} "
                    f"PSNR={psnr:5.2f} 保真={psnr_src:5.2f}/SSIM={ssim_src:.3f} "
                    f"{row['sec_per_frame']:.3f}s/帧",
                    flush=True,
                )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["scheme"])
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# 黑盒净化提速候选（研究）",
        "",
        f"- 帧数 {args.frames}；batch {args.batch}；BA@h264 = 净化后再过 H.264 CRF23",
        "",
        "| 方案 | 内容 | 候选 | clean | BA | BA@h264 | PSNR | SSIM | 对原帧 PSNR | 对原帧 SSIM | 秒/帧 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['scheme']} | {row['content']} | {row['candidate']} | {row['clean_ba']:.4f} | "
            f"{row['ba']:.4f} | {row['ba_h264']:.4f} | {row['psnr']:.2f} | {row['ssim']:.4f} | "
            f"{row['psnr_src']:.2f} | {row['ssim_src']:.4f} | {row['sec_per_frame']:.3f} |"
        )
    args.out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(f"已写出 {args.out.with_suffix('.md')}", flush=True)


if __name__ == "__main__":
    main()
