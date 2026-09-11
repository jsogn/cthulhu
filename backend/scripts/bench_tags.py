"""单武器耗时(ms/帧)与 PSNR 实测，为 UI 的耗时/画质标签提供数据锚。

用法（720p 竖屏素材，默认取 24 帧）：
  uv run --project backend python backend/scripts/bench_tags.py <视频路径>
"""

from __future__ import annotations

import subprocess
import sys
import time

import numpy as np

from cthulhu_backend.fingerprint import adversarial
from cthulhu_backend.transform import (
    extra_attacks,
    regenerate,
    strategies,
)
from cthulhu_backend.transform import (
    video as video_transform,
)
from cthulhu_backend.watermark import common as watermark_common


def decode(path: str, count: int) -> np.ndarray:
    cmd = [
        "ffmpeg", "-v", "error", "-i", path,
        "-frames:v", str(count), "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    # 先按 720×1280 读取；非该尺寸时用 ffprobe 修正。
    try:
        frames = np.frombuffer(raw, np.uint8).reshape(count, 1280, 720, 3)
    except ValueError:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "csv=p=0", path],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        w, h = (int(x) for x in probe.split(","))
        frames = np.frombuffer(raw, np.uint8).reshape(count, h, w, 3)
    return frames


def psnr(reference: np.ndarray, attacked: np.ndarray) -> float:
    a = reference.astype(np.float32)
    b = np.asarray(attacked, dtype=np.float32)
    if b.max() <= 1.0 + 1e-6:
        b = b * 255.0
    mse = float(np.mean((a - b) ** 2))
    return float("inf") if mse == 0 else float(10 * np.log10(255.0**2 / mse))


def bench(name: str, frames: np.ndarray, fn) -> None:
    start = time.perf_counter()
    out = np.asarray(fn())
    elapsed = (time.perf_counter() - start) * 1000 / len(frames)
    print(f"{name:22s} {elapsed:8.2f} ms/帧  PSNR {psnr(frames, out):6.2f} dB", flush=True)


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else (
        "/Users/alone/Downloads/暗水印测试/AD-下载8.mp4"
    )
    frames = decode(path, 24)
    rng = np.random.default_rng(7)
    spoof_bits = watermark_common.payload_bits(1234, 64)

    print(f"素材 {path} 帧数 {len(frames)} 尺寸 {frames.shape[1:3]}", flush=True)
    cases = [
        ("几何微旋转 0.4°", lambda: strategies.rotate_de_sync(
            frames, list(range(len(frames))), 0.4)),
        ("色彩微扰", lambda: video_transform.regrade(
            frames.astype(np.float32) / 255.0, rng)),
        ("伪水印注入 Δ6", lambda: strategies._embed_qim_frames(
            frames.astype(np.float32) / 255.0, spoof_bits, 6)),
        ("跨帧估计 0.6", lambda: regenerate.temporal_subtract(frames, 0.6)),
        ("FFT 相位 0.5", lambda: regenerate.fft_phase(frames, 0.5, rng)),
        ("小波细节 0.8", lambda: regenerate.dwt_detail(frames, 0.8, rng)),
        ("DCT 重量化 12", lambda: extra_attacks.dct_requant(frames, 12.0)),
        ("pHash ε=0.08", lambda: adversarial.attack_frames(frames, epsilon=0.08)),
    ]
    for name, fn in cases:
        bench(name, frames, fn)

    # 模型类武器较重，用更少帧测；模型缺失时跳过。
    try:
        bench("神经 DINOv2 0.05(6帧)", frames[:6],
              lambda: regenerate.copy_attack(frames[:6], 0.05, rng))
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"神经对抗跳过: {exc}", flush=True)
    try:
        bench("人脸 0.04(8帧)", frames[:8],
              lambda: regenerate.face_perturb(frames[:8], 0.04, rng))
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"人脸跳过: {exc}", flush=True)


if __name__ == "__main__":
    main()
