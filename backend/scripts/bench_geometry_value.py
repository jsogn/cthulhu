"""几何武器价值验证：对基准库水印的 BER、PSNR、pHash 一致率，与旋转对照。"""

from __future__ import annotations

import subprocess

import numpy as np

from cthulhu_backend.evaluate import metrics
from cthulhu_backend.fingerprint import hashes
from cthulhu_backend.transform import extra_attacks, strategies
from cthulhu_backend.watermark import common, dft, dwt, qim, ss

SRC = "/Users/alone/Downloads/暗水印测试/AD-下载8.mp4"
BITS = common.payload_bits(1, 64)
REF = common.SYNC + BITS


def decode(path: str, frames: int) -> np.ndarray:
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-frames:v", str(frames),
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        check=True, capture_output=True,
    ).stdout
    return np.frombuffer(raw, np.uint8).reshape(frames, 1280, 720, 3)


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


def main() -> None:
    gray = luma(decode(SRC, 24).astype(np.float32) / 255.0)
    rng = np.random.default_rng(7)
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
    attacks = {
        "旋转0.4°(对照)": lambda x: strategies.rotate_de_sync(x, list(range(len(x))), 0.4),
        "扭曲0.003": lambda x: extra_attacks.local_warp(x, 0.003, rng),
        "扭曲0.005": lambda x: extra_attacks.local_warp(x, 0.005, rng),
        "扭曲0.006": lambda x: extra_attacks.local_warp(x, 0.006, rng),
        "剪切0.01": lambda x: extra_attacks.perspective_shear(x, 0.01, rng),
        "剪切0.02": lambda x: extra_attacks.perspective_shear(x, 0.02, rng),
        "抖动0.005": lambda x: extra_attacks.translate_jitter(x, 0.005, rng),
        "抖动0.01": lambda x: extra_attacks.translate_jitter(x, 0.01, rng),
        "抖动0.02": lambda x: extra_attacks.translate_jitter(x, 0.02, rng),
    }
    header = f"{'攻击':12s} {'PSNR':>7s} {'pHash':>7s} " + " ".join(f"{k:>8s}" for k in watermarks)
    print(header, flush=True)
    for name, attack in attacks.items():
        attacked = np.clip(attack(gray), 0, 1)
        cols = []
        for wm, extract in watermarks.values():
            wm_attacked = np.clip(attack(wm), 0, 1)
            cols.append(f"{scheme_ber(wm_attacked, extract):8.3f}")
        print(
            f"{name:12s} {psnr(gray, attacked):7.1f} {phash_agreement(gray, attacked):7.3f} "
            + " ".join(cols),
            flush=True,
        )
    # 重编码鲁棒性：胜出组合攻击后过 x264→x265 CRF28→x264 链。
    import sys

    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    from bench_ss_robust import reencode_chain

    print("--- 重编码后", flush=True)
    for label, attack in {
        "抖动0.01": lambda x: extra_attacks.translate_jitter(x, 0.01, rng),
        "扭曲0.006": lambda x: extra_attacks.local_warp(x, 0.006, rng),
        "剪切0.02": lambda x: extra_attacks.perspective_shear(x, 0.02, rng),
    }.items():
        cols = []
        for wm, extract in watermarks.values():
            attacked = np.clip(attack(wm), 0, 1)
            chained = reencode_chain(attacked, f"/tmp/geo-{label}")
            cols.append(f"{scheme_ber(chained, extract):8.3f}")
        print(f"{label:12s} " + " ".join(cols), flush=True)


if __name__ == "__main__":
    main()
