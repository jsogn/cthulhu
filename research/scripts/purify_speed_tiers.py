#!/usr/bin/env python3
"""净化速度档位 × 公开深度方案验收（研究用途）。

回答的问题是：把扩散净化从「全分辨率逐帧」降到「低分辨率 + 批处理」之后，
VideoSeal / PixelSeal / WAM 三个公开深度方案的水印是否还能被抹掉。

档位与 UI 的「性能模式」一一对应：

- original       max_edge=0   batch=1  strength=0.15（最慢、最保守）
- fast512        max_edge=512 batch=4  strength=0.15
- turbo256_s010  max_edge=256 batch=8  strength=0.10 detail=0.5（极速清除预设）
- turbo256_s015  max_edge=256 batch=8  strength=0.15 detail=0.5

指标：BA（0.5 为随机；视频方案取跨帧聚合）、H.264 CRF23 后的 BA、PSNR/SSIM
（相对带水印帧）、单帧耗时。只使用公开预训练权重与本地素材。
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
VSEAL_REPO = ROOT / "research" / "vendor" / "videoseal"
WAM_REPO = ROOT / "research" / "vendor" / "watermark-anything"
for extra in (SCRIPTS, VSEAL_REPO, WAM_REPO, ROOT / "backend" / "src"):
    sys.path.insert(0, str(extra))

from _harness import quality as harness_quality  # noqa: E402
from _harness import read_video, transcode  # noqa: E402
from cthulhu_backend.transform import purify

TIERS: list[tuple[str, int, int, float, float]] = [
    ("original", 0, 1, 0.15, 0.0),
    ("fast512", 512, 4, 0.15, 0.0),
    ("turbo256_s010", 256, 8, 0.10, 0.5),
    ("turbo256_s015", 256, 8, 0.15, 0.5),
    # 追加候选：1 步扩散、更大批、更小长边（仅 --tiers 显式指定时跑）
    ("turbo256_s005", 256, 8, 0.05, 0.5),
    ("turbo192_s010", 192, 8, 0.10, 0.5),
    ("turbo256_b16_s010", 256, 16, 0.10, 0.5),
    ("turbo192_s005", 192, 8, 0.05, 0.5),
]


def device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def quality(reference: np.ndarray, attacked: np.ndarray) -> tuple[float, float]:
    """画质口径统一走 _harness（与生产清洗报告同源）。"""
    return harness_quality(reference, attacked)


def h264_crf23(frames_np: np.ndarray) -> np.ndarray:
    """真实 H.264 CRF23 链路：净化扰动能否穿过平台常规重编码。"""
    return transcode(frames_np, crf=23, fps=30.0)


def build_seal(dev: str, scheme: str):
    """VideoSeal / PixelSeal：视频级嵌入，跨帧聚合 BA。"""
    os.chdir(VSEAL_REPO)
    sys.path.insert(0, str(VSEAL_REPO))
    import videoseal

    model = videoseal.load(scheme).to(dev).eval()

    def embed_video(frames_np: np.ndarray) -> tuple[np.ndarray, torch.Tensor]:
        tensor = (
            torch.from_numpy(frames_np.astype(np.float32) / 255.0)
            .permute(0, 3, 1, 2)
            .to(dev)
        )
        with torch.no_grad():
            out = model.embed(tensor, is_video=True)
        watermarked = (
            (out["imgs_w"].clamp(0.0, 1.0).permute(0, 2, 3, 1).cpu().numpy() * 255.0)
            .round()
            .astype(np.uint8)
        )
        return watermarked, out["msgs"][0:1].float().to(dev)

    def ba_video(frames_np: np.ndarray, message: torch.Tensor) -> tuple[float, float]:
        tensor = (
            torch.from_numpy(frames_np.astype(np.float32) / 255.0)
            .permute(0, 3, 1, 2)
            .to(dev)
        )
        with torch.no_grad():
            preds = model.detect(tensor, is_video=True)["preds"]
        # 与既有研究口径一致（eot_pgd_video.aggregate_ba）：先平均 logits 再判决。
        # 逐帧 key-frame 复制模式下，二值多数投票会被未嵌入帧稀释。
        logits = preds[:, 1:]
        bits = (logits > 0).float()
        per_frame = float((bits == message).float().mean().item())
        aggregated = (logits.mean(dim=0, keepdim=True) > 0).float()
        agg = float((aggregated == message).float().mean().item())
        return agg, per_frame

    return "video", embed_video, ba_video


def build_wam_frames(dev: str):
    """WAM：256 原生图像方案，逐帧嵌入同一 payload 后取平均 BA。"""
    os.chdir(WAM_REPO)
    sys.path.insert(0, str(WAM_REPO))
    import argparse as arg_parser
    import json as json_module

    import omegaconf
    from watermark_anything.augmentation.augmenter import Augmenter
    from watermark_anything.data.metrics import msg_predict_inference
    from watermark_anything.data.transforms import (
        default_transform,
        normalize_img,
        unnormalize_img,
    )
    from watermark_anything.models import Wam, build_embedder, build_extractor
    from watermark_anything.modules.jnd import JND

    # 内联 notebooks/inference_utils 的加载逻辑，避免为一个函数引入 matplotlib。
    params = json_module.loads((WAM_REPO / "checkpoints" / "params.json").read_text())
    args = arg_parser.Namespace(**params)
    embedder_params = omegaconf.OmegaConf.load(args.embedder_config)[args.embedder_model]
    extractor_cfg = omegaconf.OmegaConf.load(args.extractor_config)
    extractor_params = extractor_cfg[args.extractor_model]
    augmenter_cfg = omegaconf.OmegaConf.load(args.augmentation_config)
    attenuation_cfg = omegaconf.OmegaConf.load(args.attenuation_config)
    embedder = build_embedder(args.embedder_model, embedder_params, args.nbits)
    extractor = build_extractor(
        extractor_cfg.model, extractor_params, args.img_size, args.nbits
    )
    augmenter = Augmenter(**augmenter_cfg)
    try:
        attenuation = JND(
            **attenuation_cfg[args.attenuation],
            preprocess=unnormalize_img,
            postprocess=normalize_img,
        )
    except Exception:  # noqa: BLE001 - 无 JND 配置时退化为无衰减
        attenuation = None

    wam = Wam(embedder, extractor, augmenter, attenuation, args.scaling_w, args.scaling_i)
    wam.load_state_dict(
        torch.load(WAM_REPO / "checkpoints" / "wam_mit.pth", map_location="cpu")
    )
    wam = wam.to(dev).eval()
    torch.manual_seed(20260910)
    message = torch.randint(0, 2, (1, 32)).float().to(dev)

    def to_pil(tensor: torch.Tensor) -> Image.Image:
        arr = unnormalize_img(tensor).clamp(0.0, 1.0).squeeze(0).permute(1, 2, 0).cpu().numpy()
        return Image.fromarray((arr * 255.0).round().astype(np.uint8))

    def embed_video(frames_np: np.ndarray) -> tuple[np.ndarray, torch.Tensor]:
        out = []
        with torch.no_grad():
            for frame in frames_np:
                cover = default_transform(Image.fromarray(frame)).unsqueeze(0).to(dev)
                watermarked = wam.embed(cover, message)["imgs_w"]
                out.append(np.asarray(to_pil(watermarked)))
        return np.stack(out), message

    def ba_video(frames_np: np.ndarray, msg: torch.Tensor) -> tuple[float, float]:
        scores = []
        with torch.no_grad():
            for frame in frames_np:
                attacked = default_transform(Image.fromarray(frame)).unsqueeze(0).to(dev)
                preds = wam.detect(attacked)["preds"].cpu()
                mask = torch.sigmoid(preds[:, 0, :, :])
                bits = preds[:, 1:, :, :]
                pred = msg_predict_inference(bits, mask)
                scores.append(float((pred == msg.cpu()).float().mean().item()))
        mean = float(np.mean(scores))
        return mean, mean

    return "video", embed_video, ba_video


def read_frames(path: Path, count: int) -> np.ndarray:
    return read_video(path, count)


def vertical_kenburns(frames: int, width: int = 720, height: int = 1280) -> np.ndarray:
    image = Image.open(VSEAL_REPO / "assets" / "imgs" / "1.jpg").convert("RGB")
    img_w, img_h = image.size
    crop_w = int(img_h * width / height)
    out = []
    for index in range(frames):
        frac = index / max(frames - 1, 1)
        x0 = int((img_w - crop_w) * frac)
        crop = image.crop((x0, 0, x0 + crop_w, img_h)).resize((width, height), Image.BICUBIC)
        out.append(np.asarray(crop))
    return np.stack(out)


def natural_kenburns(frames: int, size: int = 512) -> np.ndarray:
    """与既有验收矩阵同源的 512 方内容（Ken Burns 运镜），保证方案 roundtrip 有效。"""
    image = Image.open(VSEAL_REPO / "assets" / "imgs" / "1.jpg").convert("RGB")
    img_w, img_h = image.size
    out = []
    for index in range(frames):
        frac = index / max(frames - 1, 1)
        crop_w = int(img_w * (0.55 - 0.15 * frac))
        crop_h = int(img_h * (0.55 - 0.15 * frac))
        x0 = int((img_w - crop_w) * frac)
        y0 = int((img_h - crop_h) * (0.5 - 0.5 * frac))
        crop = image.crop((x0, y0, x0 + crop_w, y0 + crop_h)).resize(
            (size, size), Image.BICUBIC
        )
        out.append(np.asarray(crop))
    return np.stack(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schemes", nargs="+", default=["videoseal", "pixelseal", "wam"])
    parser.add_argument("--contents", default="natural512,vertical720")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--scale", type=float, default=1.0, help="content 缩放系数（调试用）")
    parser.add_argument("--tiers", default="")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "research" / "output" / "purify_speed_tiers",
    )
    args = parser.parse_args()
    if not args.out.is_absolute():
        args.out = ROOT / args.out

    tiers = TIERS
    if args.tiers:
        wanted = {item for item in args.tiers.split(",") if item}
        tiers = [tier for tier in TIERS if tier[0] in wanted]

    dev = device()
    print(f"device={dev} torch={torch.__version__}", flush=True)
    warm_start = time.perf_counter()
    purify.purify_frames(
        np.zeros((2, 64, 64, 3), dtype=np.uint8),
        strength=0.1,
        steps=4,
        max_edge=256,
        batch=2,
    )
    print(f"净化模型预热完成：{time.perf_counter() - warm_start:.1f}s", flush=True)

    contents: dict[str, np.ndarray] = {}
    for name in args.contents.split(","):
        name = name.strip()
        if not name:
            continue
        if name == "vertical720":
            contents[name] = vertical_kenburns(args.frames)
        elif name == "natural512":
            contents[name] = natural_kenburns(args.frames)
        elif name == "real512":
            contents[name] = read_frames(ROOT / "research" / "data" / "real_clip_512.mp4", args.frames)
        else:
            print(f"跳过未知内容：{name}")

    builders = {
        "videoseal": lambda: build_seal(dev, "videoseal"),
        "pixelseal": lambda: build_seal(dev, "pixelseal"),
        "wam": lambda: build_wam_frames(dev),
    }

    rows: list[dict[str, object]] = []
    for scheme in args.schemes:
        builder = builders.get(scheme)
        if builder is None:
            print(f"跳过未知方案：{scheme}")
            continue
        try:
            kind, embed_video, ba_video = builder()
        except Exception as exc:  # noqa: BLE001 - 缺权重时跳过
            print(f"[{scheme}] 加载失败，跳过：{exc}", flush=True)
            continue
        print(f"[{scheme}] 已加载（{kind}）", flush=True)

        for content_name, frames in contents.items():
            cover = frames[: args.frames]
            watermarked, message = embed_video(cover)
            clean_ba, _ = ba_video(watermarked, message)
            valid = clean_ba >= 0.9
            print(
                f"[{scheme}/{content_name}] clean BA={clean_ba:.4f} "
                f"size={cover.shape[2]}x{cover.shape[1]} "
                f"{'（有效）' if valid else '（嵌入本身无效，仅作测速）'}",
                flush=True,
            )
            for tier_name, max_edge, batch, strength, detail in tiers:
                start = time.perf_counter()
                attacked = purify.purify_frames(
                    watermarked,
                    strength=strength,
                    steps=args.steps,
                    seed=args.seed,
                    max_edge=max_edge,
                    batch=batch,
                    detail=detail,
                    detail_sigma=0.6,
                )
                elapsed = time.perf_counter() - start
                ba, per_frame = ba_video(attacked, message)
                psnr, ssim = quality(watermarked, attacked)
                if valid:
                    h264 = h264_crf23(attacked)
                    ba_h264, _ = ba_video(h264, message)
                else:
                    ba_h264 = float("nan")
                row = {
                    "scheme": scheme,
                    "content": content_name,
                    "clean_ba": round(clean_ba, 4),
                    "valid": int(valid),
                    "tier": tier_name,
                    "max_edge": max_edge,
                    "batch": batch,
                    "strength": strength,
                    "detail": detail,
                    "ba": round(ba, 4),
                    "ba_h264": round(ba_h264, 4),
                    "per_frame_ba": round(per_frame, 4),
                    "psnr": round(psnr, 2),
                    "ssim": round(ssim, 4),
                    "seconds": round(elapsed, 2),
                    "sec_per_frame": round(elapsed / max(len(cover), 1), 3),
                }
                rows.append(row)
                print(
                    f"  {tier_name:<14} BA={ba:.4f} BA@h264={ba_h264:.4f} "
                    f"PSNR={psnr:.2f} {row['sec_per_frame']}s/帧",
                    flush=True,
                )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.out.with_suffix(".csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["scheme"])
        writer.writeheader()
        writer.writerows(rows)

    md_path = args.out.with_suffix(".md")
    lines = [
        "# 净化速度档位 × 公开深度方案验收（研究）",
        "",
        f"- 帧数：{args.frames}；steps={args.steps}；detail_sigma=0.6；device={dev}",
        "- original：0/batch1/s0.15 ｜ fast512：512/batch4/s0.15 ｜ "
        "turbo256_s010：256/batch8/s0.10/detail0.5 ｜ turbo256_s015：256/batch8/s0.15/detail0.5",
        "- BA 为随机水平 0.5；BA@h264 为净化后再过 H.264 CRF23 的结果",
        "- clean=嵌入后未净化的 BA：< 0.9 说明该内容下方案自身 roundtrip 就无效，只作测速用（BA@h264 记为 nan）",
        "",
        "| 方案 | 内容 | 档位 | clean | BA | BA@h264 | PSNR | SSIM | 秒/帧 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        h264_text = "nan" if not row["valid"] else f"{row['ba_h264']:.4f}"
        lines.append(
            f"| {row['scheme']} | {row['content']} | {row['tier']} | {row['clean_ba']:.4f} | "
            f"{row['ba']:.4f} | {h264_text} | {row['psnr']:.2f} | {row['ssim']:.4f} | "
            f"{row['sec_per_frame']:.3f} |"
        )
    md_path.write_text("\n".join(lines) + "\n")
    print(f"已写出 {csv_path} 与 {md_path}", flush=True)


if __name__ == "__main__":
    main()
