"""像素重写（伪超分 sr_rewrite / CLAHE）对基准库各水印的破坏力验证。

口径与通杀矩阵一致：嵌入 → 单武器攻击 → 提取 → BER。BER 接近 0.5 才视为
破坏；若攻击后与嵌入后相比几乎不动，说明该武器对暗水印无效。

历史记录脚本（不可直接运行）：该武器与 `regenerate.sr_rewrite/clahe_rewrite`
已随「弱武器裁决」从生产代码删除（见 `docs/研究-通杀验收.md`），保留本文件
仅为留下当时的实测口径；如需复跑，从对应历史提交取回这两个函数。
"""

from __future__ import annotations

import numpy as np

from cthulhu_backend import samples
from cthulhu_backend.evaluate import metrics
from cthulhu_backend.transform import regenerate
from cthulhu_backend.watermark import common, dft, dwt, qim, ss

BITS = common.payload_bits(1, 64)
REF = common.SYNC + BITS


def ber(frames: np.ndarray, extract) -> float:
    bits = [bit for frame in frames for bit in extract(frame)]
    return metrics.ber(REF * len(frames), bits)


def to_u8(frames: np.ndarray) -> np.ndarray:
    return (np.clip(frames, 0, 1) * 255).round().astype(np.uint8)


def to_f32(frames: np.ndarray) -> np.ndarray:
    return frames.astype(np.float32) / 255.0


def main() -> None:
    if not hasattr(regenerate, "sr_rewrite"):
        print("跳过：sr_rewrite/clahe_rewrite 已随「弱武器裁决」从生产代码移除。")
        print("本脚本仅作历史记录；复跑需先从历史提交取回这两个函数。")
        return
    clean = samples.make_cut_video(2, 12, 320, 240, seed=2)
    schemes = [
        ("SS α=0.25", lambda f: ss.embed(f, BITS, seed=0, alpha=0.25),
         lambda f: ss.extract(f, len(BITS), seed=0)),
        ("QIM Δ=20", lambda f: qim.embed(f, BITS, delta=20.0),
         lambda f: qim.extract(f, len(BITS), delta=20.0)),
        ("DWT α=0.08", lambda f: dwt.embed(f, BITS, seed=0, alpha=0.08),
         lambda f: dwt.extract(f, len(BITS), seed=0)),
        ("DFT α=4.0", lambda f: dft.embed(f, BITS, seed=0, alpha=4.0),
         lambda f: dft.extract(f, len(BITS), seed=0)),
    ]
    print(f"{'方案':12s} {'嵌入后':>8s} {'sr':>8s} {'clahe':>8s} {'sr+clahe':>9s}")
    for name, embed_fn, extract_fn in schemes:
        watermarked = np.stack([embed_fn(f) for f in clean])
        base = ber(watermarked, extract_fn)
        sr = ber(to_f32(regenerate.sr_rewrite(to_u8(watermarked))), extract_fn)
        clahe = ber(to_f32(regenerate.clahe_rewrite(to_u8(watermarked))), extract_fn)
        both = ber(
            to_f32(regenerate.clahe_rewrite(regenerate.sr_rewrite(to_u8(watermarked)))),
            extract_fn,
        )
        print(f"{name:12s} {base:8.3f} {sr:8.3f} {clahe:8.3f} {both:9.3f}")


if __name__ == "__main__":
    main()
