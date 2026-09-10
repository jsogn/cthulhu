#!/usr/bin/env python3
"""黑盒净化增强栈验收矩阵（研究用途）。

对比四类组合在公开方案上的 BA / 画质：
- baseline：现有扩散净化；
- detail：扩散 + 原帧高频细节回注；
- embedding：detail + 嵌入域 v2 低频重写；
- full：detail + 时序一致性减法 + 嵌入域 v2 增强重写（视频）。

只使用公开预训练模型与本地素材；不接触任何平台线上系统。
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "research" / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ROOT / "backend" / "src"))

import cleanse_matrix_deep as matrix

from cthulhu_backend.transform import embedding_domain, purify


def _domain_for(scheme: str) -> str:
    if scheme == "wam":
        return "chroma"
    if scheme == "videoseal":
        return "luma"
    return "both"


def _apply_stack(
    frames: np.ndarray,
    *,
    scheme: str,
    stack: str,
    strength: float,
    steps: int,
    detail_strength: float,
    detail_sigma: float,
    temporal_strength: float,
    embedding_strength: float,
    video: bool,
) -> np.ndarray:
    detail = detail_strength if stack in {"detail", "embedding", "full"} else 0.0
    temporal = (
        temporal_strength
        if video and stack == "full"
        else 0.0
    )
    out = purify.purify_frames(
        frames,
        strength=strength,
        steps=steps,
        seed=20260909,
        detail=detail,
        detail_sigma=detail_sigma,
        temporal_strength=temporal,
    )
    if stack in {"embedding", "full"}:
        out = embedding_domain.attack(
            out,
            mode=_domain_for(scheme),
            strength=embedding_strength,
            rng=np.random.default_rng(20260909),
            variant="v2",
            block_jitter=stack == "full",
            multiscale=stack == "full",
            chroma_subsample=stack == "full",
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schemes", nargs="+", default=["trustmark", "mbrs", "wam", "videoseal"])
    parser.add_argument("--stacks", default="baseline,detail,embedding,full")
    parser.add_argument("--images", type=int, default=1)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--strengths", default="0.15,0.25")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--detail-strength", type=float, default=0.5)
    parser.add_argument("--detail-sigma", type=float, default=0.6)
    parser.add_argument("--temporal-strength", type=float, default=0.5)
    parser.add_argument("--embedding-strength", type=float, default=0.4)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "research" / "output" / "purify_stack_matrix",
    )
    args = parser.parse_args()
    stacks = [item for item in args.stacks.split(",") if item]
    strengths = [float(item) for item in args.strengths.split(",") if item]
    rng = np.random.default_rng(20260909)
    dev = matrix.device()
    builders = {
        "trustmark": lambda: matrix.build_trustmark(rng),
        "mbrs": lambda: matrix.build_mbrs(rng, dev),
        "wam": lambda: matrix.build_wam(rng, dev),
        "videoseal": lambda: matrix.build_videoseal(dev),
    }

    rows: list[dict[str, object]] = []
    for scheme_name in args.schemes:
        builder = builders.get(scheme_name)
        if builder is None:
            print(f"跳过未知方案：{scheme_name}")
            continue
        os.chdir(ROOT)
        try:
            built = builder()
            name, embed, decode, purify_size, kind = built[:5]
        except Exception as exc:  # noqa: BLE001 - 缺权重/仓库时单方案跳过
            print(f"[{scheme_name}] 加载失败，跳过：{exc}")
            continue

        samples: list[tuple[str, int, object]] = []
        if kind == "image":
            samples.append(("natural", 0, matrix.natural_still()))
            for index in range(args.images):
                samples.append(("simple", index, matrix.synthetic_image(index)))
        else:
            samples.append(("natural", 0, matrix.natural_video(args.frames)))
            samples.append(("simple", 0, matrix.simple_video(args.frames)))

        for content_kind, sample_index, cover in samples:
            if kind == "image":
                watermarked = embed(cover)
                clean_ba = decode(watermarked)
                source = np.asarray(matrix._resize(watermarked, purify_size))
                reference = np.asarray(watermarked)
                for strength in strengths:
                    for stack in stacks:
                        attacked = _apply_stack(
                            source[None],
                            scheme=name,
                            stack=stack,
                            strength=strength,
                            steps=args.steps,
                            detail_strength=args.detail_strength,
                            detail_sigma=args.detail_sigma,
                            temporal_strength=args.temporal_strength,
                            embedding_strength=args.embedding_strength,
                            video=False,
                        )[0]
                        image = Image.fromarray(attacked).resize(
                            (watermarked.width, watermarked.height), Image.BICUBIC
                        )
                        ba = decode(image)
                        psnr, ssim = matrix._quality(reference[None], np.asarray(image)[None])
                        rows.append(
                            {
                                "scheme": name,
                                "content": content_kind,
                                "sample": sample_index,
                                "stack": stack,
                                "strength": strength,
                                "clean_ba": round(clean_ba, 4),
                                "ba": round(ba, 4),
                                "psnr": round(psnr, 2),
                                "ssim": round(ssim, 4),
                            }
                        )
            else:
                watermarked, message = embed(cover)
                clean_ba, _ = decode(watermarked, message)
                for strength in strengths:
                    for stack in stacks:
                        attacked = _apply_stack(
                            watermarked,
                            scheme=name,
                            stack=stack,
                            strength=strength,
                            steps=args.steps,
                            detail_strength=args.detail_strength,
                            detail_sigma=args.detail_sigma,
                            temporal_strength=args.temporal_strength,
                            embedding_strength=args.embedding_strength,
                            video=True,
                        )
                        ba, per_frame = decode(attacked, message)
                        psnr, ssim = matrix._quality(watermarked, attacked)
                        rows.append(
                            {
                                "scheme": name,
                                "content": content_kind,
                                "sample": sample_index,
                                "stack": stack,
                                "strength": strength,
                                "clean_ba": round(clean_ba, 4),
                                "ba": round(ba, 4),
                                "per_frame_ba": round(per_frame, 4),
                                "psnr": round(psnr, 2),
                                "ssim": round(ssim, 4),
                            }
                        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.out.with_suffix(".csv")
    if rows:
        with csv_path.open("w", newline="") as handle:
            fieldnames = list(dict.fromkeys(key for row in rows for key in row))
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    md_path = args.out.with_suffix(".md")
    with md_path.open("w") as handle:
        handle.write("# 黑盒净化增强栈验收矩阵（研究）\n\n")
        handle.write(
            f"- 栈：{stacks}；强度：{strengths}；steps={args.steps}；"
            f"detail={args.detail_strength}；temporal={args.temporal_strength}；"
            f"embedding={args.embedding_strength}\n"
        )
        handle.write("- BA=比特准确率（0.5 为随机）；FNR=BA<0.6 的样本比例\n\n")
        handle.write("| 方案 | 栈 | FNR |\n| --- | --- | ---: |\n")
        for scheme in dict.fromkeys(str(row["scheme"]) for row in rows):
            for stack in stacks:
                subset = [
                    row
                    for row in rows
                    if row["scheme"] == scheme and row["stack"] == stack
                ]
                if not subset:
                    continue
                hits = sum(1 for row in subset if float(row["ba"]) < 0.6)
                handle.write(
                    f"| {scheme} | {stack} | {hits}/{len(subset)} = "
                    f"{hits / len(subset):.3f} |\n"
                )
    print(f"CSV 已写入：{csv_path}")
    print(f"报告已写入：{md_path}")
    for row in rows:
        print(
            f"{row['scheme']:<10} {row['stack']:<9} s={row['strength']:.2f} "
            f"BA={row['ba']:.3f} PSNR={row['psnr']:.2f} SSIM={row['ssim']:.4f}"
        )


if __name__ == "__main__":
    main()
