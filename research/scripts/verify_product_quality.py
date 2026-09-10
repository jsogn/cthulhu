#!/usr/bin/env python3
"""验证产品默认净化路径的画质：走真实 services.run_desensitize，出对比图。

用法：
    backend/.venv/bin/python research/scripts/verify_product_quality.py 
        --input research/data/real_clip_512.mp4 --edge 256 --sigma 1.5

输出：对原帧的 PSNR/SSIM（逐帧均值）、净化后的视频、以及「原帧 | 净化后」
两行拼图（第一行中部纹理带，第二行字幕带），便于肉眼确认字幕是否可读。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
for extra in (ROOT / "research" / "scripts", ROOT / "backend" / "src"):
    sys.path.insert(0, str(extra))

from cthulhu_backend import services  # noqa: E402
from cthulhu_backend.media import ffmpeg  # noqa: E402


def _psnr(ref: np.ndarray, out: np.ndarray) -> float:
    mse = float(np.mean((ref - out) ** 2))
    return float("inf") if mse <= 0 else 10 * np.log10(255.0**2 / mse)


def _to_u8_range(frames: np.ndarray) -> np.ndarray:
    """decode_video 返回 [0,1] 浮点（可能是灰度）；统一成 0~255 便于读数与出图。"""
    work = frames.astype(np.float64)
    if work.size and float(work.max()) <= 1.5:
        work = work * 255.0
    return work


def _ssim(ref: np.ndarray, out: np.ndarray) -> float:
    """全局 SSIM（无 skimage 依赖）：按通道滑动窗口统计。"""
    if ref.ndim == 2:
        ref = ref[..., None]
        out = out[..., None]
    window = 8
    values: list[float] = []
    for channel in range(ref.shape[2]):
        a = ref[..., channel]
        b = out[..., channel]
        height, width = a.shape
        for y in range(0, height - window + 1, window):
            for x in range(0, width - window + 1, window):
                pa = a[y : y + window, x : x + window]
                pb = b[y : y + window, x : x + window]
                mu_a, mu_b = float(pa.mean()), float(pb.mean())
                va, vb = float(pa.var()), float(pb.var())
                cov = float(((pa - mu_a) * (pb - mu_b)).mean())
                c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
                values.append(
                    ((2 * mu_a * mu_b + c1) * (2 * cov + c2))
                    / ((mu_a**2 + mu_b**2 + c1) * (va + vb + c2))
                )
    return float(np.mean(values)) if values else 1.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(ROOT / "research" / "data" / "real_clip_512.mp4"))
    parser.add_argument("--output", default="/tmp/cthulhu_quality/out.mp4")
    parser.add_argument("--edge", type=int, default=256)
    parser.add_argument("--strength", type=float, default=0.15, help="0 = 关闭净化（对照）")
    parser.add_argument("--sigma", type=float, default=0.0, help="0 = 自动（按分辨率换算）")
    parser.add_argument("--detail", type=float, default=1.0)
    parser.add_argument("--temporal", type=float, default=0.0)
    parser.add_argument(
        "--metric-frames",
        type=int,
        default=120,
        help="只对前 N 帧算 PSNR/SSIM，长片全量计算会吃光内存",
    )
    parser.add_argument(
        "--out-image",
        type=Path,
        default=ROOT / "research" / "output" / "product_quality_compare.png",
    )
    args = parser.parse_args()

    source = Path(args.input)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    report = services.run_desensitize(
        str(source),
        str(output),
        purify_strength=args.strength,
        purify_max_edge=args.edge,
        purify_batch=8,
        purify_detail=args.detail,
        purify_detail_sigma=args.sigma,
        purify_temporal=args.temporal,
        auto_profile=True,
        audio_remix=False,
        compute_metrics=False,
    )
    elapsed = time.perf_counter() - started
    print(
        f"purify_note={report['purify_note']} out_frames={report['out_frames']} "
        f"耗时={elapsed:.1f}s（{elapsed / max(1, report['out_frames']) * 1000:.1f} ms/帧）",
        flush=True,
    )

    original, _ = ffmpeg.decode_video(str(source))
    cleaned, _ = ffmpeg.decode_video(str(output))
    length = min(len(original), len(cleaned))
    sample = min(length, args.metric_frames)
    ref = _to_u8_range(original[:sample])
    out = _to_u8_range(cleaned[:sample])
    psnr = _psnr(ref, out)
    ssim = float(np.mean([_ssim(ref[i], out[i]) for i in range(sample)]))
    print(
        f"对原帧：PSNR={psnr:.2f}dB SSIM={ssim:.4f}（抽样 {sample}/{length} 帧）",
        flush=True,
    )

    height = ref.shape[1]
    bands = [slice(int(height * 0.05), int(height * 0.45)), slice(int(height * 0.78), height)]
    rows = [np.concatenate([ref[0][band], out[0][band]], axis=1) for band in bands]
    strip = np.concatenate(rows, axis=0)
    args.out_image.parent.mkdir(parents=True, exist_ok=True)
    canvas = np.clip(strip, 0, 255).astype(np.uint8)
    Image.fromarray(canvas).save(args.out_image)
    print(f"对比图（左原帧 | 右净化后）：{args.out_image}", flush=True)


if __name__ == "__main__":
    main()
