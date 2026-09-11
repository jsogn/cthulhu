"""研究脚本共用脚手架：帧读写、方案构建、BA 与画质指标。

存在的理由：43 个脚本里各自复制帧读取/方案构建/指标，口径不同曾产出过
「PSNR 68dB」这类假读数（审计 R3 研究侧）。本模块把口径钉死在一处：

- 帧一律是 uint8 RGB `(N,H,W,3)`，读取走 pyav 之外的唯一路径 ffmpeg；
- 基准方案（SS/QIM/DWT/DFT）与比特一致率（BA/BER）来自生产 watermark 包；
- 画质指标直接用中立层 `quality`（`evaluate.metrics` 亦由其提供，与后端清洗
  报告同源），不引入 skimage。

新脚本一律基于本模块；历史脚本按需迁移（不强制一次改完）。

用法：
    from _harness import SCHEMES, ba, quality, read_video, transcode, write_video
"""

from __future__ import annotations

import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path

import numpy as np

from cthulhu_backend import quality as quality_metrics
from cthulhu_backend.media import ffmpeg as backend_ffmpeg
from cthulhu_backend.watermark import common, dft, dwt, qim, ss

BITS = common.payload_bits(1, 64)
REF = common.SYNC + BITS

SCHEMES: dict[str, tuple[Callable, Callable]] = {
    "ss-a0.25": (
        lambda frame: ss.embed(frame, BITS, seed=0, alpha=0.25),
        lambda frame: ss.extract(frame, len(BITS), seed=0),
    ),
    "qim-d20": (
        lambda frame: qim.embed(frame, BITS, delta=20.0),
        lambda frame: qim.extract(frame, len(BITS), delta=20.0),
    ),
    "dwt": (
        lambda frame: dwt.embed(frame, BITS, seed=0),
        lambda frame: dwt.extract(frame, len(BITS), seed=0),
    ),
    "dft-a4": (
        lambda frame: dft.embed(frame, BITS, seed=0, alpha=4.0),
        lambda frame: dft.extract(frame, len(BITS), seed=0),
    ),
}


def to_u8(frames: np.ndarray) -> np.ndarray:
    """统一到研究口径：uint8 RGB。float [0,1] 输入自动放大。"""
    work = np.asarray(frames)
    if work.dtype == np.uint8:
        return work
    if work.size == 0:
        return work.astype(np.uint8)
    scaled = work if float(work.max()) > 1.5 else work * 255.0
    return np.clip(scaled, 0, 255).round().astype(np.uint8)


def to_float(frames: np.ndarray) -> np.ndarray:
    """统一到嵌入器口径：float32 [0,1]。"""
    return to_u8(frames).astype(np.float32) / 255.0


def video_info(path: str | Path) -> dict:
    return backend_ffmpeg.video_info(str(path))


def read_video(path: str | Path, count: int | None = None) -> np.ndarray:
    """解码为 uint8 RGB (N,H,W,3)；count 限制帧数（None = 全片）。"""
    info = video_info(path)
    width, height = int(info["width"]), int(info["height"])
    proc = subprocess.run(
        [
            backend_ffmpeg.FFMPEG_BIN,
            "-v",
            "error",
            "-i",
            str(path),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        capture_output=True,
        check=True,
    )
    frame_bytes = width * height * 3
    total = len(proc.stdout) // frame_bytes
    frames = np.frombuffer(proc.stdout, dtype=np.uint8)[: total * frame_bytes]
    frames = frames.reshape(total, height, width, 3)
    if total == 0:
        raise RuntimeError("输入视频没有可解码帧")
    return frames if count is None else frames[:count]


def write_video(
    frames: np.ndarray,
    path: str | Path,
    fps: float = 30.0,
    *,
    codec: str = "ffv1",
    crf: int | None = None,
    audio_source: str | Path | None = None,
) -> None:
    """写中间视频：默认 FFV1 无损；给 crf 时走 libx264 有损重编码。"""
    work = to_u8(frames)
    height, width = work.shape[1:3]
    cmd = [
        backend_ffmpeg.FFMPEG_BIN,
        "-y",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps:.6f}",
        "-i",
        "-",
    ]
    if audio_source is not None:
        cmd += ["-i", str(audio_source), "-map", "0:v:0", "-map", "1:a?", "-shortest"]
    cmd += ["-c:v", codec]
    if crf is not None:
        cmd += ["-preset", "veryfast", "-crf", str(crf)]
    if audio_source is not None:
        cmd += ["-c:a", "copy"]
    cmd.append(str(path))
    subprocess.run(cmd, input=np.ascontiguousarray(work).tobytes(), capture_output=True, check=True)


def transcode(frames: np.ndarray, *, crf: int = 23, fps: float = 30.0) -> np.ndarray:
    """真实 H.264 CRF 链路：扰动能否穿过平台常规重编码（返回 uint8 RGB）。"""
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "src.mkv"
        encoded = Path(tmp) / "out.mp4"
        write_video(frames, source, fps=fps)
        subprocess.run(
            [
                backend_ffmpeg.FFMPEG_BIN,
                "-y",
                "-v",
                "error",
                "-i",
                str(source),
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                str(crf),
                str(encoded),
            ],
            capture_output=True,
            check=True,
        )
        return read_video(encoded)


def embed(scheme: str, frames: np.ndarray) -> np.ndarray:
    """嵌入基准水印；输入/输出都是研究口径的 uint8 RGB（内部转 [0,1] 浮点）。"""
    embed_frame, _ = SCHEMES[scheme]
    work = to_float(frames)
    return to_u8(np.stack([embed_frame(frame) for frame in work]))


def extract_bits(scheme: str, frames: np.ndarray) -> np.ndarray:
    """逐帧提取并展平为比特序列（与 REF 前 N 位比较即可得 BA/BER）。"""
    _, extract = SCHEMES[scheme]
    if len(frames) == 0:
        return np.empty((0,), dtype=np.uint8)
    work = to_float(frames)
    return np.asarray([bit for frame in work for bit in extract(frame)], dtype=np.uint8)


def frame_ba(scheme: str, frames: np.ndarray, reference: np.ndarray | None = None) -> float:
    """逐帧 BA 的均值：每帧内部做同步头对齐搜索（与生产口径一致）。"""
    _, extract = SCHEMES[scheme]
    ref = list(REF if reference is None else reference)
    work = to_float(frames)
    if len(work) == 0:
        return 1.0
    scores = [1.0 - common.bit_error_rate(ref, list(extract(frame))) for frame in work]
    return float(np.mean(scores))


def ba(bits: np.ndarray, reference: np.ndarray | None = None) -> float:
    """比特一致率（BA）；0.5 表示已被打到随机水平。"""
    ref = np.asarray(REF if reference is None else reference)
    count = min(len(ref), len(bits))
    if count == 0:
        return 1.0
    return float(np.mean(ref[:count] == bits[:count]))


def ber(bits: np.ndarray, reference: np.ndarray | None = None) -> float:
    """比特错误率；脚本里统一用它而不是各写一份。"""
    return 1.0 - ba(bits, reference)


def self_ba(bits_before: np.ndarray, bits_after: np.ndarray) -> float:
    """与攻击前自身的比特一致率；0.5 表示比特已被打成随机。"""
    count = min(len(bits_before), len(bits_after))
    if count == 0:
        return 1.0
    return float(np.mean(np.asarray(bits_before)[:count] == np.asarray(bits_after)[:count]))


def quality(reference: np.ndarray, attacked: np.ndarray) -> tuple[float, float]:
    """逐帧 PSNR/SSIM 的均值（与生产清洗报告同源，不依赖 skimage）。"""
    ref = to_u8(reference)
    atk = to_u8(attacked)
    count = min(len(ref), len(atk))
    if count == 0:
        return float("inf"), 1.0
    psnr_values = [quality_metrics.psnr(ref[index], atk[index]) for index in range(count)]
    ssim_values = [quality_metrics.ssim(ref[index], atk[index]) for index in range(count)]
    return float(np.mean(psnr_values)), float(np.mean(ssim_values))
