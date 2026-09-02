"""SS 攻击的重编码鲁棒化探索：候选原语 → 模拟平台重编码 → 提取 BER。

口径：嵌入 → 候选攻击 → H.264→H.265(CRF28)→H.264 重编码链 → 提取。
目标：攻击 + 重编码后 BER 仍 ≥0.4（当前已知失败模式：反相关图案被平滑
重编码抹掉，BER 回落到 0.1~0.15）。
"""

from __future__ import annotations

import os
import subprocess
import tempfile

import numpy as np
from scipy.ndimage import gaussian_filter

from cthulhu_backend import services
from cthulhu_backend.evaluate import metrics
from cthulhu_backend.media import ffmpeg
from cthulhu_backend.transform import regenerate
from cthulhu_backend.watermark import common, ss

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


def ber(frames: np.ndarray, n_bits: int, seed: int) -> float:
    bits = [bit for f in frames for bit in ss.extract(f, n_bits, seed=seed)]
    return metrics.ber(REF * len(frames), bits)


def reencode_chain(frames: np.ndarray, prefix: str) -> np.ndarray:
    """x264 CRF23 → x265 CRF28 → x264 CRF23，模拟平台强平滑重编码。"""
    paths = [f"{prefix}-{i}.mp4" for i in range(3)]
    ffmpeg.encode_video(frames, paths[0], fps=30, crf=23, codec="libx264")
    for i, (codec, crf) in enumerate([("libx265", 28), ("libx264", 23)], start=1):
        subprocess.run(
            ["ffmpeg", "-v", "error", "-i", paths[i - 1], "-c:v", codec,
             "-preset", "veryfast", "-crf", str(crf), "-an", "-y", paths[i]],
            check=True, capture_output=True,
        )
    out, _ = ffmpeg.decode_video(paths[2], grayscale=True)
    return out


def shift_vertical(frames: np.ndarray, px: int) -> np.ndarray:
    return np.roll(frames, px, axis=1)


def smooth_field(shape: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    field = rng.standard_normal((shape[0] // 64 + 2, shape[1] // 64 + 2))
    big = np.kron(field, np.ones((64, 64)))[: shape[0], : shape[1]]
    return gaussian_filter(big.astype(np.float32), sigma=8)


def main() -> None:
    rgb = decode(SRC, 24)
    gray = luma(rgb.astype(np.float32) / 255.0)
    rng = np.random.default_rng(7)
    for alpha in (0.08, 0.25):
        marked = np.stack([ss.embed(f, BITS, seed=0, alpha=alpha) for f in gray])
        print(f"--- SS α={alpha} 嵌入后 BER={ber(marked, 64, 0):.3f}", flush=True)
        with tempfile.TemporaryDirectory() as tmp:
            reencoded = reencode_chain(marked, os.path.join(tmp, "wm"))
            print(f"    仅重编码链        BER={ber(reencoded, 64, 0):.3f}")
            attacks = {
                "噪声σ0.02": marked + rng.standard_normal(marked.shape).astype(np.float32) * 0.02,
                "噪声σ0.04": marked + rng.standard_normal(marked.shape).astype(np.float32) * 0.04,
                "跨帧β1.5": regenerate.temporal_subtract(marked, beta=1.5),
                "平滑随机场0.04": np.stack(
                    [f + smooth_field(f.shape, rng) * 0.04 for f in marked]
                ),
                "垂直位移8px": shift_vertical(marked, 8),
                "垂直位移16px": shift_vertical(marked, 16),
            }
            for name, attacked in attacks.items():
                out = reencode_chain(np.clip(attacked, 0, 1), os.path.join(tmp, "atk"))
                print(f"    {name:16s} 攻击+重编码 BER={ber(out, 64, 0):.3f}", flush=True)
    # β 阈值（numpy 跨帧，裸水印帧）。
    marked = np.stack([ss.embed(f, BITS, seed=0, alpha=0.08) for f in gray])
    print("--- numpy 跨帧 β 阈值（SS α=0.08，攻击+重编码）", flush=True)
    for beta in (0.8, 1.0, 1.2, 1.5):
        out = reencode_chain(
            np.clip(regenerate.temporal_subtract(marked, beta=beta), 0, 1), "beta"
        )
        print(f"    β={beta}  BER={ber(out, 64, 0):.3f}", flush=True)
    # 真实管线：均衡预置，原生跨帧 β 升档，输出再过重编码链。
    base = {
        "reorder": False, "speed": 1.0, "recrop": 0.0, "perturb": 0.0,
        "regrade": False, "audio_remix": False, "echo_defeat": False,
        "audio_strong": False, "sharpness": False, "color_restore": False,
        "denoise": False, "rotate": 0.0, "requant": 0, "noise": 0.0,
        "dct_step": 0.0, "fft_phase": 0.0, "dwt_detail": 0.0,
        "native_temporal": False, "skip_vmaf": True, "seed": 0,
    }
    print("--- 最小管线（numpy 跨帧 β1.2，门控开/关）→ 重编码链", flush=True)
    with tempfile.TemporaryDirectory() as tmp:
        wm_path = os.path.join(tmp, "wm.mp4")
        ffmpeg.encode_video(marked, wm_path, fps=30, crf=23)
        # 假设 1：首轮 CRF23 已磨掉高频图案 → 在解码帧上做跨帧过减。
        coded, _ = ffmpeg.decode_video(wm_path)
        attacked_coded = np.clip(regenerate.temporal_subtract(coded, beta=1.5), 0, 1)
        chain_coded = reencode_chain(attacked_coded, os.path.join(tmp, "coded"))
        print(
            f"    解码帧过减β1.5 后链 BER={ber(chain_coded, 64, 0):.3f}",
            flush=True,
        )
        # 假设 3：管线走 RGB uint8 解码，与灰度解码路径不同。
        coded_rgb, _ = ffmpeg.decode_video(wm_path, grayscale=False, out_dtype="float32")
        attacked_rgb = regenerate.temporal_subtract(coded_rgb, beta=1.5)
        chain_rgb = reencode_chain(
            np.clip(attacked_rgb, 0, 1),
            os.path.join(tmp, "rgb"),
        )
        print(
            f"    RGB解码帧过减β1.5 后链 BER={ber(luma(chain_rgb), 64, 0):.3f}",
            flush=True,
        )
        # 假设 2：多一次输出编码把反图案磨掉 → 直接帧域过减后再 CRF23+链。
        direct = np.clip(regenerate.temporal_subtract(marked, beta=1.5), 0, 1)
        direct_path = os.path.join(tmp, "direct.mp4")
        ffmpeg.encode_video(direct, direct_path, fps=30, crf=23)
        direct_coded, _ = ffmpeg.decode_video(direct_path)
        direct_chain = reencode_chain(direct_coded, os.path.join(tmp, "direct-chain"))
        print(
            f"    直接过减β1.5→CRF23→链 BER={ber(direct_chain, 64, 0):.3f}",
            flush=True,
        )
        for beta, gate, native in (
            (1.2, True, False), (1.2, True, True), (1.5, True, False),
        ):
            tag = f"b{beta}-gate" + ("-native" if native else "")
            out_path = os.path.join(tmp, f"pipe-{tag}.mp4")
            services.run_desensitize(
                wm_path, out_path,
                **{**base, "temporal_sub": beta, "quality_protect": gate,
                   "psnr_target": 38.0, "ssim_target": 0.94,
                   "native_temporal": native},
            )
            cleaned, _ = ffmpeg.decode_video(out_path)
            chained = reencode_chain(cleaned, os.path.join(tmp, f"chain-{tag}"))
            print(
                f"    {tag}  管线后={ber(cleaned, 64, 0):.3f}  再重编码={ber(chained, 64, 0):.3f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
