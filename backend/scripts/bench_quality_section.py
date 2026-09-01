"""画质优化区的真实作用实测：门控的衰减系数 + 空间降噪对 SS 的破坏力。"""

from __future__ import annotations

import subprocess

import numpy as np

from cthulhu_backend import samples
from cthulhu_backend.attacks import spatial as spatial_attacks
from cthulhu_backend.evaluate import metrics
from cthulhu_backend.transform import extra_attacks, regenerate
from cthulhu_backend.watermark import common, ss

BITS = common.payload_bits(1, 64)
REF = common.SYNC + BITS


def decode(path: str, count: int) -> np.ndarray:
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-frames:v", str(count),
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        check=True, capture_output=True,
    ).stdout
    return np.frombuffer(raw, np.uint8).reshape(count, 1280, 720, 3)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    return float(10 * np.log10(255.0**2 / mse))


def ss_ber(frames: np.ndarray) -> float:
    bits = [bit for f in frames for bit in ss.extract(f, len(BITS), seed=0)]
    return metrics.ber(REF * len(frames), bits)


def main() -> None:
    frames = decode("/Users/alone/Downloads/暗水印测试/AD-下载8.mp4", 24)
    rng = np.random.default_rng(7)

    # 均衡档 numpy 武器栈（不含神经/人脸），门控前后对比。
    params = extra_attacks.AssaultParams(
        requant=64, noise=0.003, dct_step=12.0,
        temporal_sub=0.6, fft_phase=0.5, dwt_detail=0.8,
    )
    attacked = extra_attacks.apply(frames, params, rng)
    raw_psnr = psnr(frames, attacked)
    gated = extra_attacks.quality_gate(frames, attacked, 38.0, 0.94)
    gated_psnr = psnr(frames, gated)
    delta = gated.astype(np.float32) - frames.astype(np.float32)
    full = attacked.astype(np.float32) - frames.astype(np.float32)
    k = float(np.sqrt(np.mean(delta**2) / max(np.mean(full**2), 1e-12)))
    print(f"均衡栈(无门控) PSNR {raw_psnr:.1f}dB → 门控38dB 后 {gated_psnr:.1f}dB, 有效强度系数 k≈{k:.3f}")

    # 门控对 SS 破坏力的净影响。
    watermarked = np.stack([ss.embed(f.astype(np.float32) / 255.0, BITS, seed=0, alpha=0.25) for f in frames])
    wm_u8 = (np.clip(watermarked, 0, 1) * 255).round().astype(np.uint8)
    att = extra_attacks.apply(wm_u8, params, rng)
    att_g = extra_attacks.quality_gate(wm_u8, att, 38.0, 0.94)
    print(f"SS BER: 嵌入后 {ss_ber(watermarked):.3f} | 无门控攻击 {ss_ber(att.astype(np.float32)/255.0):.3f} | 门控后 {ss_ber(att_g.astype(np.float32)/255.0):.3f}")

    # 空间降噪单独对 SS 的破坏。
    denoised = spatial_attacks.gaussian(watermarked, sigma=1.4)
    print(f"SS BER: 嵌入后 {ss_ber(watermarked):.3f} | σ1.4 空间降噪后 {ss_ber(denoised):.3f} (PSNR {10*np.log10(1.0/float(np.mean((watermarked-denoised)**2))):.1f}dB)")


if __name__ == "__main__":
    main()
