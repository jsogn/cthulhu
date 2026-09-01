"""DWT/DFT 置零式攻击原语验证：对比现有随机化攻击的破坏力与 PSNR。"""

from __future__ import annotations

import numpy as np

from cthulhu_backend import samples
from cthulhu_backend.evaluate import metrics
from cthulhu_backend.transform import regenerate
from cthulhu_backend.watermark import common, dft, dwt

BITS = common.payload_bits(1, 64)
REF = common.SYNC + BITS


def dwt_zero(frames: np.ndarray, strength: float) -> np.ndarray:
    out = []
    for frame in frames:
        coeffs, (h2, w2) = dwt._decompose(frame)
        half_h, half_w = h2 // 2, w2 // 2
        coeffs[half_h:, half_w:] *= 1.0 - strength
        recon = dwt._reconstruct(coeffs)
        canvas = frame.copy()
        canvas[:h2, :w2] = recon
        out.append(np.clip(canvas, 0, 1))
    return np.stack(out)


def dft_zero(frames: np.ndarray, strength: float) -> np.ndarray:
    out = []
    for frame in frames:
        spectrum = np.fft.rfft2(frame)
        mask = dft._band_mask(spectrum.shape)
        spectrum[mask] *= 1.0 - strength
        out.append(np.clip(np.fft.irfft2(spectrum, s=frame.shape), 0, 1))
    return np.stack(out)


def ber(frames: np.ndarray, extract, seed: int | None = None) -> float:
    bits = [
        bit
        for f in frames
        for bit in (extract(f, len(BITS), seed=seed) if seed is not None else extract(f, len(BITS)))
    ]
    return metrics.ber(REF * len(frames), bits)


def psnr(ref: np.ndarray, attacked: np.ndarray) -> float:
    mse = float(np.mean((ref - attacked) ** 2))
    return float(10 * np.log10(1.0 / mse))


def main() -> None:
    clean = samples.make_cut_video(2, 12, 320, 240, seed=2)
    dwt_wm = np.stack([dwt.embed(f, BITS, seed=0) for f in clean])
    dft_wm = np.stack([dft.embed(f, BITS, seed=0) for f in clean])
    print(f"DWT 嵌入后 {ber(dwt_wm, dwt.extract, 0):.3f} | DFT 嵌入后 {ber(dft_wm, dft.extract, 0):.3f}")
    print("--- DWT", flush=True)
    for strength in (0.5, 0.8, 1.0):
        rand = regenerate.dwt_detail(dwt_wm, strength, np.random.default_rng(7))
        zero = dwt_zero(dwt_wm, strength)
        print(
            f"  s={strength}  随机化 BER={ber(rand, dwt.extract, 0):.3f}({psnr(dwt_wm, rand):.1f}dB)"
            f" | 置零 BER={ber(zero, dwt.extract, 0):.3f}({psnr(dwt_wm, zero):.1f}dB)",
            flush=True,
        )
    print("--- DFT", flush=True)
    for strength in (0.5, 0.8, 1.0):
        phase = regenerate.fft_phase(dft_wm, strength, np.random.default_rng(7))
        zero = dft_zero(dft_wm, strength)
        print(
            f"  s={strength}  相位随机 BER={ber(phase, dft.extract, 0):.3f}({psnr(dft_wm, phase):.1f}dB)"
            f" | 置零 BER={ber(zero, dft.extract, 0):.3f}({psnr(dft_wm, zero):.1f}dB)",
            flush=True,
        )
    # 重编码鲁棒性：攻击 → x264→x265 CRF28→x264 → 提取。
    import sys

    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    from bench_ss_robust import reencode_chain

    print("--- 重编码后（s=1.0）", flush=True)
    for name, wm, extract, att_rand, att_zero in (
        ("DWT", dwt_wm, dwt.extract,
         regenerate.dwt_detail(dwt_wm, 1.0, np.random.default_rng(7)), dwt_zero(dwt_wm, 1.0)),
        ("DFT", dft_wm, dft.extract,
         regenerate.fft_phase(dft_wm, 1.0, np.random.default_rng(7)), dft_zero(dft_wm, 1.0)),
    ):
        chain_rand = reencode_chain(att_rand, f"/tmp/{name}-rand")
        chain_zero = reencode_chain(att_zero, f"/tmp/{name}-zero")
        seed = 0
        print(
            f"  {name}  随机化重编码后 BER={ber(chain_rand, extract, seed):.3f}"
            f" | 置零重编码后 BER={ber(chain_zero, extract, seed):.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
