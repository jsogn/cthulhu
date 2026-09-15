"""对比平移抖动的轨迹设计：白噪声 vs 低频正弦漂移。

裁决口径：
1. 同幅度下四方案（SS/QIM/DWT/DFT）BER 不显著劣于白噪声基线；
2. pHash 一致率保持高位（几何武器不应误伤指纹）；
3. temporal_stability 与相邻帧最大位移明显下降（晕眩代理指标）；
4. 垂直幅度与水平一致（SS/DFT 按行提取，纵向错位不可压缩）。

只读实验：不改动任何业务代码，输出为终端表格。

用法：
  CTHULHU_BENCH_SRC=<720p 基准视频> uv run --project backend python backend/scripts/bench_jitter_trajectory.py
"""

from __future__ import annotations

import os
import subprocess

import numpy as np

from cthulhu_backend.evaluate import metrics
from cthulhu_backend.fingerprint import hashes
from cthulhu_backend.transform import extra_attacks
from cthulhu_backend.watermark import common, dft, dwt, qim, ss

SRC = os.environ.get("CTHULHU_BENCH_SRC", "")
BITS = common.payload_bits(1, 64)
REF = common.SYNC + BITS
SEED = 7


def decode(path: str, frames: int) -> np.ndarray:
    raw = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-i", path, "-frames:v", str(frames),
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ],
        check=True,
        capture_output=True,
    ).stdout
    return np.frombuffer(raw, np.uint8).reshape(frames, 720, 1280, 3)


def luma(rgb: np.ndarray) -> np.ndarray:
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def scheme_ber(frames: np.ndarray, extract) -> float:
    bits = [bit for f in frames for bit in extract(f)]
    return metrics.ber(REF * len(frames), bits)


def psnr(ref: np.ndarray, attacked: np.ndarray) -> float:
    mse = float(np.mean((ref - attacked) ** 2))
    return float("inf") if mse == 0 else float(10 * np.log10(1.0 / mse))


def phash_agreement(ref: np.ndarray, attacked: np.ndarray) -> float:
    agree = [
        1.0 - hashes.hamming_bits(hashes.phash(a), hashes.phash(b)) / 64.0
        for a, b in zip(ref, attacked)
    ]
    return float(np.mean(agree))


def displacement_stats(
    shape: tuple[int, ...], jitter: float, trajectory: str, seed: int = SEED,
) -> tuple[int, int, float]:
    """在 200 帧合成序列上重放偏移，计算晕眩代理指标（短切片下正弦的
    相邻帧差会随相位取样偏差，长序列才代表真实管线行为）。"""
    count, height, width = 200, shape[1], shape[2]
    rng = np.random.default_rng(seed)
    dx, dy = extra_attacks.jitter_offsets(
        count, width, height, jitter, rng, trajectory=trajectory,
    )
    max_ddx = int(np.abs(np.diff(dx)).max()) if count > 1 else 0
    max_ddy = int(np.abs(np.diff(dy)).max()) if count > 1 else 0
    return max_ddx, max_ddy, float(np.abs(dy).mean())


def main() -> None:
    if not SRC:
        raise SystemExit("请设置 CTHULHU_BENCH_SRC 指向基准 720p 视频后重跑")
    gray = luma(decode(SRC, 24).astype(np.float32) / 255.0)
    watermarks = {
        "SS": (np.stack([ss.embed(f, BITS, seed=0, alpha=0.25) for f in gray]),
               lambda f: ss.extract(f, len(BITS), seed=0)),
        "QIM": (np.stack([qim.embed(f, BITS, delta=20.0) for f in gray]),
                lambda f: qim.extract(f, len(BITS), delta=20.0)),
        "DWT": (np.stack([dwt.embed(f, BITS, seed=0) for f in gray]),
                lambda f: dwt.extract(f, len(BITS), seed=0)),
        "DFT": (np.stack([dft.embed(f, BITS, seed=0) for f in gray]),
                lambda f: dft.extract(f, len(BITS), seed=0)),
    }
    variants = [
        ("白噪声0.005", "noise", 0.005),
        ("白噪声0.01", "noise", 0.01),
        ("正弦漂移0.005", "sine", 0.005),
        ("正弦漂移0.008", "sine", 0.008),
    ]
    seeds = [7, 11, 23, 42, 99]
    header = (
        f"{'轨迹':14s} {'PSNR':>6s} {'pHash':>6s} {'稳定':>7s} "
        f"{'maxΔx':>6s} {'maxΔy':>6s} {'mean|dy|':>8s} "
        + " ".join(f"{k:>7s}" for k in watermarks)
    )
    print(header, flush=True)
    for label, trajectory, amp in variants:
        sums: dict[str, list[float]] = {key: [] for key in [*watermarks, "psnr", "phash", "stab"]}
        ddx, ddy, dy = [], [], []
        for seed in seeds:
            attack = lambda x, seed=seed, amp=amp, trajectory=trajectory: extra_attacks.translate_jitter(
                x, amp, np.random.default_rng(seed), trajectory=trajectory,
            )
            attacked = np.clip(attack(gray), 0, 1)
            sums["psnr"].append(psnr(gray, attacked))
            sums["phash"].append(phash_agreement(gray, attacked))
            sums["stab"].append(metrics.temporal_stability(attacked))
            for key, (wm, extract) in watermarks.items():
                wm_attacked = np.clip(attack(wm), 0, 1)
                sums[key].append(scheme_ber(wm_attacked, extract))
            md_x, md_y, m_dy = displacement_stats(gray.shape, amp, trajectory, seed)
            ddx.append(md_x)
            ddy.append(md_y)
            dy.append(m_dy)
        cols = [f"{np.mean(sums[key]):7.3f}" for key in watermarks]
        print(
            f"{label:14s} {np.mean(sums['psnr']):6.1f} "
            f"{np.mean(sums['phash']):6.3f} {np.mean(sums['stab']):7.4f} "
            f"{np.mean(ddx):6.1f} {np.mean(ddy):6.1f} {np.mean(dy):8.2f} "
            + " ".join(cols),
            flush=True,
        )


if __name__ == "__main__":
    main()
