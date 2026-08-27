"""基于 ffmpeg/ffprobe 的视频文件 I/O 与 VMAF 计算。"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from fractions import Fraction

import numpy as np
from scipy.ndimage import zoom

from cthulhu_backend.sample_prep.align import estimate_shift

# 打包后可指向内置的静态 ffmpeg 目录；缺省回退系统 PATH。
FFMPEG_DIR = os.environ.get("CTHULHU_FFMPEG_DIR", "")
_CUSTOM_DIR = ""


def _find_binary(name: str) -> str:
    """解析可执行文件路径。

    桌面应用由 Finder 启动时，进程 PATH 只有系统默认目录，不含
    /opt/homebrew/bin，因此不能只依赖 PATH 查找，需显式检查常见安装位置。
    """
    for base in (FFMPEG_DIR, _CUSTOM_DIR):
        if base:
            for candidate_name in (name, f"{name}.exe"):
                candidate = os.path.join(base, candidate_name)
                if os.access(candidate, os.X_OK):
                    return candidate
    candidates = [shutil.which(name)]
    if sys.platform == "darwin":
        candidates += [
            os.path.join(prefix, name)
            for prefix in ("/opt/homebrew/bin", "/usr/local/bin", "/opt/local/bin")
        ]
    for candidate in candidates:
        if candidate and os.access(candidate, os.X_OK):
            return candidate
    return name


def set_custom_dir(directory: str) -> None:
    """指定用户选择或自动安装的 ffmpeg 目录，并即时刷新二进制路径。"""
    global _CUSTOM_DIR
    _CUSTOM_DIR = directory or ""
    _refresh_binaries()


def _refresh_binaries() -> None:
    global FFMPEG_BIN, FFPROBE_BIN
    FFMPEG_BIN = _find_binary("ffmpeg")
    FFPROBE_BIN = _find_binary("ffprobe")


FFMPEG_BIN = _find_binary("ffmpeg")
FFPROBE_BIN = _find_binary("ffprobe")


def has_ffmpeg() -> bool:
    return os.access(FFMPEG_BIN, os.X_OK) and os.access(FFPROBE_BIN, os.X_OK)


def has_encoder(codec: str) -> bool:
    """检查当前 FFmpeg 是否包含指定视频编码器（如 libx265）。"""
    result = subprocess.run(
        [FFMPEG_BIN, "-hide_banner", "-encoders"],
        capture_output=True,
        text=True,
        check=False,
    )
    return codec in result.stdout


_FILTER_CACHE: set[str] | None = None

# 解码/编码的统一内存闸门：任何单次帧缓冲超过该字节数直接拒绝执行，
# 从根上保证不会因全片 float64 加载把机器内存打爆。
_MAX_FRAME_BYTES = int(os.environ.get("CTHULHU_MAX_FRAME_BYTES", str(16 * 1024**3)))


def _check_frame_budget(frames: int, height: int, width: int, channels: int, extra_bytes_per_px: int) -> None:
    """按预估帧数与 dtype 检查瞬时内存，超限时给出可操作错误。"""
    total_bytes = frames * height * width * channels * (extra_bytes_per_px + 1)
    if total_bytes >= _MAX_FRAME_BYTES:
        raise ValueError(
            f"拒绝执行：预计瞬时内存约 {total_bytes / 1024**3:.1f} GiB，超过上限 "
            f"{_MAX_FRAME_BYTES / 1024**3:.1f} GiB。请传入 max_frames 或 vf=scale 降采样，"
            "或改用 decode_sampled / decode_video_range / StreamingDecoder；"
            "如确需提高上限，可设置环境变量 CTHULHU_MAX_FRAME_BYTES。"
        )


def has_filter(name: str) -> bool:
    """检查当前 FFmpeg 是否包含指定视频滤镜（结果缓存）。"""
    global _FILTER_CACHE
    if _FILTER_CACHE is None:
        result = subprocess.run(
            [FFMPEG_BIN, "-hide_banner", "-filters"],
            capture_output=True,
            text=True,
            check=False,
        )
        _FILTER_CACHE = {
            parts[1] for line in result.stdout.splitlines() if len(parts := line.split()) >= 2
        }
    return name in _FILTER_CACHE


def probe(path: str) -> dict:
    """ffprobe JSON：流信息、容器格式与时长。"""
    out = subprocess.run(
        [FFPROBE_BIN, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return json.loads(out)


def video_info(path: str) -> dict:
    info = probe(path)
    stream = next(s for s in info["streams"] if s["codec_type"] == "video")
    fps = Fraction(stream.get("avg_frame_rate") or "30/1")
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": float(fps),
        "pix_fmt": stream.get("pix_fmt"),
        "codec": stream.get("codec_name"),
        "format": info["format"].get("format_name"),
        "duration": float(info["format"].get("duration") or 0),
    }


def decode_video(
    path: str,
    grayscale: bool = True,
    hwaccel: bool = False,
    max_frames: int | None = None,
    vf: str | None = None,
    out_dtype: str = "float64",
) -> tuple[np.ndarray, dict]:
    """解码为帧数组（灰度 float64 [0,1]，形状 (F,H,W)），返回 (frames, info)。"""
    info = video_info(path)
    channels = 1 if grayscale else 3
    scale = re.search(r"scale=(\d+):(\d+)", vf or "")
    width = int(scale.group(1)) if scale else info["width"]
    height = int(scale.group(2)) if scale else info["height"]
    expected_frames = max_frames
    if expected_frames is None:
        duration = float(info.get("duration") or 0)
        expected_frames = max(1, round(duration * info["fps"])) if duration > 0 else 1
    _check_frame_budget(expected_frames, height, width, channels, np.dtype(out_dtype).itemsize)
    pix_fmt = "gray" if grayscale else "rgb24"
    cmd = [FFMPEG_BIN, "-v", "error"]
    if hwaccel:
        cmd += ["-hwaccel", "videotoolbox" if sys.platform == "darwin" else "auto"]
    cmd += ["-i", path]
    if vf:
        cmd += ["-vf", vf]
    if max_frames:
        cmd += ["-frames:v", str(max_frames)]
    cmd += ["-f", "rawvideo", "-pix_fmt", pix_fmt, "-"]
    raw = subprocess.run(
        cmd,
        capture_output=True,
        check=True,
    ).stdout
    total = len(raw) // (height * width * channels)
    arr = np.frombuffer(raw, dtype=np.uint8)
    shape = (total, height, width) if grayscale else (total, height, width, 3)
    frames = arr.reshape(shape).astype(out_dtype)
    frames /= 255.0
    return frames, info


def decode_video_range(
    path: str,
    start_frame: int,
    count: int,
    grayscale: bool = True,
) -> tuple[np.ndarray, dict]:
    """解码指定帧区间（从 start_frame 起 count 帧），返回 float32 帧数组。

    用于分块流式处理：内存只占一块，不再整片驻留。
    """
    info = video_info(path)
    channels = 1 if grayscale else 3
    _check_frame_budget(count, info["height"], info["width"], channels, np.dtype(np.float32).itemsize)
    start = max(0.0, start_frame / info["fps"])
    pix_fmt = "gray" if grayscale else "rgb24"
    cmd = [
        FFMPEG_BIN, "-v", "error",
        "-i", path,
        "-ss", str(start),
        "-frames:v", str(count),
        "-f", "rawvideo", "-pix_fmt", pix_fmt, "-",
    ]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    height, width = info["height"], info["width"]
    total = len(raw) // (height * width * channels)
    arr = np.frombuffer(raw, dtype=np.uint8)
    shape = (total, height, width) if grayscale else (total, height, width, 3)
    frames = arr.reshape(shape).astype(np.float32)
    frames /= 255.0
    return frames, info


class StreamingDecoder:
    """从指定帧区间顺序流式解码：单次 FFmpeg 进程 + 单次 seek。

    与逐块调用 decode_video_range 的差别：只 seek 一次到区间起点，之后
    顺序读取，把「每块都从头解码再丢弃」的 O(N²) 开销降为 O(N)。
    调用方必须按帧顺序完整消费区间，不支持随机跳转。
    """

    def __init__(
        self,
        path: str,
        start_frame: int,
        count: int,
        grayscale: bool = True,
        pix_fmt: str | None = None,
    ) -> None:
        self.info = video_info(path)
        start = max(0.0, start_frame / self.info["fps"])
        pix_fmt = pix_fmt or ("gray" if grayscale else "rgb24")
        if pix_fmt == "gray":
            self.frame_bytes = self.info["height"] * self.info["width"]
        elif pix_fmt == "yuv420p":
            self.frame_bytes = self.info["height"] * self.info["width"] * 3 // 2
        else:
            self.frame_bytes = self.info["height"] * self.info["width"] * 3
        self.grayscale = grayscale
        self.pix_fmt = pix_fmt
        self._remaining = max(0, count)
        cmd = [
            FFMPEG_BIN, "-v", "error",
            "-i", path,
            "-ss", str(start),
            "-frames:v", str(self._remaining),
            "-f", "rawvideo", "-pix_fmt", pix_fmt, "-",
        ]
        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
        self._closed = False

    def read(self, count: int, dtype: str = "float32") -> np.ndarray:
        """读取至多 count 帧（可能少于请求量）。

        dtype 为 "float32" 时返回 [0,1] 浮点帧；"uint8" 时直接返回
        原始灰度字节，省去一次缩放转换，供 uint8 端到端管线使用。
        """
        if self._closed or self._remaining <= 0 or count <= 0:
            empty_dtype = np.uint8 if dtype == "uint8" else np.float32
            shape = self._frame_shape(0)
            return np.empty(shape, dtype=empty_dtype)
        want = min(count, self._remaining)
        raw = bytearray()
        needed = want * self.frame_bytes
        while len(raw) < needed:
            chunk = self._proc.stdout.read(needed - len(raw))
            if not chunk:
                break
            raw.extend(chunk)
        total = len(raw) // self.frame_bytes
        self._remaining -= total
        if total == 0:
            empty_dtype = np.uint8 if dtype == "uint8" else np.float32
            shape = self._frame_shape(0)
            return np.empty(shape, dtype=empty_dtype)
        buffer = np.frombuffer(bytes(raw[: total * self.frame_bytes]), dtype=np.uint8)
        shape = self._frame_shape(total)
        frames = buffer.reshape(shape)
        if dtype == "uint8":
            return frames
        return frames.astype(np.float32) / 255.0

    def _frame_shape(self, count: int) -> tuple[int, ...]:
        """按像素格式给出帧数组形状；yuv420p 返回紧凑平面字节。"""
        h, w = self.info["height"], self.info["width"]
        if self.pix_fmt == "gray":
            return (count, h, w)
        if self.pix_fmt == "yuv420p":
            return (count, h * w * 3 // 2)
        return (count, h, w, 3)

    def close(self) -> None:
        """结束解码进程并释放资源（幂等）。"""
        if self._closed:
            return
        self._closed = True
        try:
            self._proc.stdout.close()
        except OSError:
            pass
        try:
            self._proc.terminate()
            self._proc.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            self._proc.kill()


def decode_sampled(
    path: str,
    cap: int = 200,
    grayscale: bool = True,
    return_starts: bool = False,
    scale_long_edge: int | None = None,
):
    """跨全片抽样解码至多 cap 帧（关键帧 seek，长视频不再整段解码）。

    等间隔取 10 个位置，每个位置连续解码 cap/10 帧；仅解码 cap 帧左右，
    相比 fps 滤镜全片解码，长片（8 分钟以上）耗时下降一个量级。
    return_starts=True 时额外返回每个窗口首帧的全局帧号（用于镜头边界映射）。
    scale_long_edge 指定输出长边像素（等比缩放），把降采样下沉到解码器。
    """
    info = video_info(path)
    in_w, in_h = info["width"], info["height"]
    out_w, out_h = in_w, in_h
    scale_filter = None
    if scale_long_edge and max(in_w, in_h) > scale_long_edge:
        if in_w >= in_h:
            out_w, out_h = scale_long_edge, max(2, round(scale_long_edge * in_h / in_w) // 2 * 2)
        else:
            out_h, out_w = scale_long_edge, max(2, round(scale_long_edge * in_w / in_h) // 2 * 2)
        scale_filter = f"scale={out_w}:{out_h}:flags=bicubic"

    total = max(1, round(info["duration"] * info["fps"]))
    if total <= cap:
        frames, _ = decode_video(
            path,
            grayscale=grayscale,
            out_dtype="float32",
            vf=scale_filter,
        )
        frames = frames[:cap]
        return (frames, info, [0]) if return_starts else (frames, info)

    pix_fmt = "gray" if grayscale else "rgb24"
    channels = 1 if grayscale else 3
    # 每窗口一次子进程；窗口数固定为 6，在长片省时与短片子进程开销间取平衡。
    positions = min(6, cap)
    per = max(1, cap // positions)
    parts: list[np.ndarray] = []
    starts: list[int] = []
    for index in range(positions):
        start_frame = int(total * (index + 0.5) / positions)
        cmd = [
            FFMPEG_BIN, "-v", "error",
            "-ss", str(start_frame / info["fps"]),
            "-i", path,
            *(["-vf", scale_filter] if scale_filter else []),
            "-frames:v", str(per),
            "-f", "rawvideo", "-pix_fmt", pix_fmt, "-",
        ]
        raw = subprocess.run(cmd, capture_output=True, check=True).stdout
        count = len(raw) // (out_h * out_w * channels)
        if count == 0:
            continue
        shape = (
            (count, out_h, out_w)
            if grayscale
            else (count, out_h, out_w, 3)
        )
        part = np.frombuffer(raw, dtype=np.uint8).reshape(shape).astype(np.float32) / 255.0
        parts.append(part)
        starts.append(start_frame)
    if not parts:
        shape = (
            (0, out_h, out_w)
            if grayscale
            else (0, out_h, out_w, 3)
        )
        frames = np.zeros(shape, dtype=np.float32)
    else:
        frames = np.concatenate(parts, axis=0)[:cap]
    return (frames, info, starts) if return_starts else (frames, info)


class StreamingEncoder:
    """把 float32 灰度帧分块写入长期 FFmpeg 编码进程。

    只编码视频轨；音轨在 finish 后由调用方单独 mux，避免双输入共享
    stdin 造成的交错死锁。内存占用与视频总长度无关。
    """

    def __init__(
        self,
        path: str,
        width: int,
        height: int,
        fps: float,
        *,
        codec: str = "libx264",
        hardware: bool = False,
        crf: int = 23,
        preset: str = "medium",
        color: bool = False,
        gop: int | None = None,
        bitrate_kbps: int | None = None,
        out_size: tuple[int, int] | None = None,
        hw_quality: int = 70,
        filters: list[str] | None = None,
        input_pix_fmt: str | None = None,
        threads: int | None = None,
        stop=None,
    ) -> None:
        if hardware and sys.platform == "darwin":
            codec = "hevc_videotoolbox" if codec == "libx265" else "h264_videotoolbox"
        cmd = [
            FFMPEG_BIN, "-y", "-v", "error",
            "-f", "rawvideo", "-pix_fmt", input_pix_fmt or ("rgb24" if color else "gray"),
            "-s", f"{width}x{height}", "-r", str(fps),
            "-i", "-",
            "-c:v", codec,
        ]
        if hardware and sys.platform == "darwin":
            if bitrate_kbps:
                cmd += ["-b:v", f"{bitrate_kbps}k"]
            else:
                cmd += ["-q:v", str(hw_quality)]
        elif bitrate_kbps:
            cmd += [
                "-b:v", f"{bitrate_kbps}k",
                "-maxrate", f"{bitrate_kbps}k",
                "-bufsize", f"{bitrate_kbps * 2}k",
            ]
        else:
            cmd += ["-preset", preset, "-crf", str(crf)]
        if gop:
            cmd += ["-g", str(gop)]
        if threads:
            cmd += ["-threads", str(threads)]
        chain = list(filters or [])
        if out_size:
            chain.append(f"scale={out_size[0]}:{out_size[1]}")
        if chain:
            cmd += ["-vf", ",".join(chain)]
        cmd += ["-pix_fmt", "yuv420p", path]
        self._stop = stop
        self._finished = False
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def write(self, frames: np.ndarray) -> None:
        if self._stop and self._stop():
            raise InterruptedError("任务已取消")
        if frames.dtype == np.uint8:
            payload = np.ascontiguousarray(frames).tobytes()
        else:
            payload = (np.clip(frames, 0, 1) * 255).round().astype(np.uint8).tobytes()
        try:
            self._proc.stdin.write(payload)
        except BrokenPipeError as exc:
            # ffmpeg 提前退出时把真实 stderr 带上，避免只看到无意义的 Broken pipe。
            self._proc.wait()
            stderr = ""
            if self._proc.stderr is not None:
                stderr = self._proc.stderr.read().decode(errors="ignore").strip()
            detail = stderr[-500:] if stderr else "编码进程意外退出"
            raise RuntimeError(f"编码进程意外退出：{detail}") from exc

    def finish(self) -> None:
        if self._stop and self._stop():
            self.abort()
            raise InterruptedError("任务已取消")
        self._proc.stdin.close()
        returncode = self._proc.wait()
        stderr = self._proc.stderr.read().decode(errors="ignore")
        self._finished = True
        if returncode != 0:
            raise subprocess.CalledProcessError(returncode, "ffmpeg", stderr=stderr)

    def abort(self) -> None:
        if self._finished:
            return
        self._proc.terminate()
        try:
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._proc.kill()
        self._finished = True


def encode_video(
    frames: np.ndarray,
    path: str,
    fps: float = 30.0,
    crf: int = 23,
    codec: str = "libx264",
    hardware: bool = False,
    gop: int | None = None,
    bitrate_kbps: int | None = None,
    out_size: tuple[int, int] | None = None,
    hw_quality: int = 70,
) -> None:
    """把帧数组编码为 MP4（yuv420p）；灰度 (F,H,W) 或彩色 (F,H,W,3) 均可。"""
    color = frames.ndim == 4
    _, height, width = frames.shape[:3]
    channels = frames.shape[3] if color else 1
    _check_frame_budget(len(frames), height, width, channels, frames.dtype.itemsize)
    raw = np.clip(frames, 0, 1)
    payload = (raw * 255).round().astype(np.uint8).tobytes()
    if hardware and sys.platform == "darwin":
        codec = "hevc_videotoolbox" if codec == "libx265" else "h264_videotoolbox"
    cmd = [
        FFMPEG_BIN, "-y", "-v", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24" if color else "gray",
        "-s", f"{width}x{height}", "-r", str(fps),
        "-i", "-",
        "-c:v", codec,
    ]
    if hardware and sys.platform == "darwin":
        # VideoToolbox 用质量标度（0~100，越高画质越好）；有码率要求时按码率。
        if bitrate_kbps:
            cmd += ["-b:v", f"{bitrate_kbps}k"]
        else:
            cmd += ["-q:v", str(hw_quality)]
    elif bitrate_kbps:
        cmd += [
            "-b:v", f"{bitrate_kbps}k",
            "-maxrate", f"{bitrate_kbps}k",
            "-bufsize", f"{bitrate_kbps * 2}k",
        ]
    else:
        cmd += ["-preset", "medium", "-crf", str(crf)]
    if gop:
        cmd += ["-g", str(gop)]
    if out_size:
        cmd += ["-vf", f"scale={out_size[0]}:{out_size[1]}"]
    cmd += ["-pix_fmt", "yuv420p", path]
    subprocess.run(
        cmd,
        input=payload,
        capture_output=True,
        check=True,
    )


def decode_audio(
    path: str,
    sample_rate: int = 16000,
    max_seconds: float | None = None,
) -> tuple[np.ndarray, int] | None:
    """解码音轨为单声道 float64 [-1,1]；无音轨时返回 None。

    max_seconds 只解码开头片段，供检测抽样使用，避免长视频整段解码拖慢任务。
    """
    info = probe(path)
    if not any(stream.get("codec_type") == "audio" for stream in info["streams"]):
        return None
    cmd = [FFMPEG_BIN, "-v", "error", "-i", path]
    if max_seconds:
        cmd += ["-t", str(max_seconds)]
    cmd += ["-f", "s16le", "-ac", "1", "-ar", str(sample_rate), "-"]
    raw = subprocess.run(
        cmd,
        capture_output=True,
        check=True,
    ).stdout
    signal = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    return signal, sample_rate


def encode_video_with_audio(
    frames: np.ndarray,
    audio: np.ndarray,
    path: str,
    fps: float = 30.0,
    sample_rate: int = 16000,
    crf: int = 23,
    codec: str = "libx264",
    hardware: bool = False,
    gop: int | None = None,
    bitrate_kbps: int | None = None,
    out_size: tuple[int, int] | None = None,
    hw_quality: int = 70,
) -> None:
    """把灰度帧数组与单声道音频编码为带音轨的 MP4。"""
    _, height, width = frames.shape
    _check_frame_budget(len(frames), height, width, 1, frames.dtype.itemsize)
    video_payload = (np.clip(frames, 0, 1) * 255).round().astype(np.uint8).tobytes()
    audio_payload = (np.clip(audio, -1, 1) * 32767).round().astype(np.int16).tobytes()
    if hardware and sys.platform == "darwin":
        codec = "hevc_videotoolbox" if codec == "libx265" else "h264_videotoolbox"
    cmd = [
        FFMPEG_BIN, "-y", "-v", "error",
        "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{width}x{height}", "-r", str(fps),
        "-i", "-",
        "-f", "s16le", "-ar", str(sample_rate), "-ac", "1",
        "-i", "-",
        "-c:v", codec,
    ]
    if hardware and sys.platform == "darwin":
        if bitrate_kbps:
            cmd += ["-b:v", f"{bitrate_kbps}k"]
        else:
            cmd += ["-q:v", str(hw_quality)]
    elif bitrate_kbps:
        cmd += [
            "-b:v", f"{bitrate_kbps}k",
            "-maxrate", f"{bitrate_kbps}k",
            "-bufsize", f"{bitrate_kbps * 2}k",
        ]
    else:
        cmd += ["-preset", "medium", "-crf", str(crf)]
    if gop:
        cmd += ["-g", str(gop)]
    if out_size:
        cmd += ["-vf", f"scale={out_size[0]}:{out_size[1]}"]
    cmd += ["-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", "-shortest", path]
    subprocess.run(
        cmd,
        input=video_payload + audio_payload,
        capture_output=True,
        check=True,
    )


def vmaf_score(distorted: str, reference: str, subsample: int | None = None) -> float | None:
    """计算两段视频的 VMAF 均值；subsample 抽样取帧可大幅提速。"""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        log_path = tmp.name
    try:
        filter_spec = f"libvmaf=log_fmt=json:log_path={log_path}"
        if subsample:
            filter_spec += f":n_subsample={subsample}"
        subprocess.run(
            [
                FFMPEG_BIN, "-v", "error",
                "-i", distorted, "-i", reference,
                "-lavfi", filter_spec,
                "-f", "null", "-",
            ],
            capture_output=True,
            check=False,
        )
        with open(log_path, encoding="utf-8") as fh:
            data = json.load(fh)
        return float(data["pooled_metrics"]["vmaf"]["mean"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    finally:
        try:
            os.unlink(log_path)
        except OSError:
            pass


def aligned_vmaf(
    distorted: str,
    reference: str,
    sample_frames: int = 16,
    fps: float = 30.0,
) -> float | None:
    """空间配准后计算 VMAF：平移对齐、裁共同区域、时间抽样后重编码再评分。

    用于重采样/重构图等造成内容错位的场景，避免朴素 VMAF 因错位虚高失真。
    """
    # 只解码开头的抽样帧做位移估计，避免长视频全片二次解码（原为性能热点）。
    d_frames, _ = decode_video(distorted, max_frames=sample_frames)
    r_frames, _ = decode_video(reference, max_frames=sample_frames)
    if d_frames.shape[1:] != r_frames.shape[1:]:
        factors = (
            d_frames.shape[1] / r_frames.shape[1],
            d_frames.shape[2] / r_frames.shape[2],
        )
        r_frames = np.stack([zoom(frame, factors, order=1) for frame in r_frames])
    n = min(len(d_frames), len(r_frames))
    if n < 2:
        return None
    d_frames, r_frames = d_frames[:n], r_frames[:n]
    dy, dx = estimate_shift(np.median(r_frames, axis=0), np.median(d_frames, axis=0))
    margin = max(8, abs(dy) + 1, abs(dx) + 1)
    # 周期纹理会产生歧义配准峰，位移可能异常大；公共区域不足时不强行配准。
    if 2 * margin >= r_frames.shape[1] or 2 * margin >= r_frames.shape[2]:
        return None
    aligned = np.stack([np.roll(np.roll(frame, dy, axis=0), dx, axis=1) for frame in d_frames])
    ref_crop = r_frames[:, margin:-margin, margin:-margin]
    dist_crop = aligned[:, margin:-margin, margin:-margin]
    indices = np.linspace(0, n - 1, min(sample_frames, n)).astype(int)
    with tempfile.TemporaryDirectory() as tmp:
        ref_path = os.path.join(tmp, "ref.mp4")
        dist_path = os.path.join(tmp, "dist.mp4")
        encode_video(ref_crop[indices], ref_path, fps=fps, crf=18)
        encode_video(dist_crop[indices], dist_path, fps=fps, crf=18)
        return vmaf_score(dist_path, ref_path)


def repair_delogo(
    path: str,
    output: str,
    regions: list[dict],
    crf: int = 23,
    progress_cb=None,
    stop=None,
    pause=None,
) -> None:
    """FFmpeg delogo 可见水印修复：按归一化区域与时间范围逐段抹除并重编码。

    经 -progress 输出汇报真实进度；stop 触发时终止进程，pause 触发时
    挂起/恢复进程（POSIX 平台），让取消与暂停即时生效。
    """
    if stop and stop():
        raise InterruptedError("任务已取消")
    info = video_info(path)
    filters = []
    for region in regions:
        x = round(region["x"] * info["width"])
        y = round(region["y"] * info["height"])
        w = max(4, round(region["w"] * info["width"]))
        h = max(4, round(region["h"] * info["height"]))
        enable = ""
        if region.get("start") is not None or region.get("end") is not None:
            start = region.get("start", 0)
            end = region.get("end", info["duration"])
            enable = f":enable='between(t,{start},{end})'"
        filters.append(f"delogo=x={x}:y={y}:w={w}:h={h}:show=0{enable}")
    cmd = [
        FFMPEG_BIN, "-y", "-v", "error",
        "-progress", "pipe:1", "-nostats",
        "-i", path, "-vf", ",".join(filters),
    ]
    cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p"]
    if any(stream.get("codec_type") == "audio" for stream in probe(path)["streams"]):
        cmd += ["-c:a", "copy"]
    cmd += [output]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    total = float(info.get("duration") or 0)
    last_percent = 0
    for raw_line in proc.stdout:
        if stop and stop():
            proc.terminate()
            proc.wait()
            raise InterruptedError("任务已取消")
        if pause is not None and pause.is_set():
            if progress_cb:
                progress_cb(last_percent, "已暂停")
            if hasattr(signal, "SIGSTOP"):
                proc.send_signal(signal.SIGSTOP)
            while pause.is_set() and not (stop and stop()):
                time.sleep(0.1)
            if stop and stop():
                proc.terminate()
                proc.wait()
                raise InterruptedError("任务已取消")
            if hasattr(signal, "SIGSTOP"):
                proc.send_signal(signal.SIGCONT)
            if progress_cb:
                progress_cb(last_percent, "已恢复")
        line = raw_line.decode(errors="ignore").strip()
        if line.startswith("out_time_us=") and total > 0:
            elapsed = int(line.split("=", 1)[1]) / 1_000_000
            percent = min(92, max(0, round(elapsed / total * 100)))
            if progress_cb and percent != last_percent:
                last_percent = percent
                progress_cb(percent, "逐帧修复中")
    returncode = proc.wait()
    stderr = proc.stderr.read().decode(errors="ignore")
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, cmd, stderr=stderr)


def extract_frame(path: str, time_seconds: float) -> bytes:
    """按时间点抽取单帧，返回 PNG 字节。"""
    result = subprocess.run(
        [
            FFMPEG_BIN, "-y", "-v", "error",
            "-ss", str(time_seconds), "-i", path,
            "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-",
        ],
        capture_output=True,
        check=True,
    )
    if not result.stdout.startswith(b"\x89PNG"):
        raise RuntimeError("抽帧失败：输出不是 PNG")
    return result.stdout


def extract_thumbnail(path: str, width: int = 320) -> bytes:
    """抽取首帧附近等比缩放缩略图，返回 JPEG 字节（供素材封面使用）。"""
    result = subprocess.run(
        [
            FFMPEG_BIN, "-y", "-v", "error",
            "-ss", "0.1", "-i", path,
            "-frames:v", "1",
            "-vf", f"scale={width}:-2",
            "-f", "image2pipe", "-vcodec", "mjpeg", "-q:v", "3", "-",
        ],
        capture_output=True,
        check=True,
    )
    if not result.stdout.startswith(b"\xff\xd8"):
        raise RuntimeError("缩略图生成失败：输出不是 JPEG")
    return result.stdout


def benchmark(path: str, max_frames: int = 600) -> dict:
    """软/硬件编解码耗时对比（秒）。"""
    started = time.perf_counter()
    frames, _ = decode_video(path, hwaccel=False, max_frames=max_frames, out_dtype="float32")
    decode_sw = time.perf_counter() - started
    frame_count = len(frames)

    started = time.perf_counter()
    frames_hw, _ = decode_video(path, hwaccel=True, max_frames=max_frames, out_dtype="float32")
    decode_hw = time.perf_counter() - started

    sw_path = os.path.join(tempfile.gettempdir(), "cthulhu-bench-sw.mp4")
    hw_path = os.path.join(tempfile.gettempdir(), "cthulhu-bench-hw.mp4")
    started = time.perf_counter()
    encode_video(frames, sw_path, fps=30, hardware=False)
    encode_sw = time.perf_counter() - started
    del frames
    started = time.perf_counter()
    encode_video(frames_hw, hw_path, fps=30, hardware=True)
    encode_hw = time.perf_counter() - started
    del frames_hw

    return {
        "frames": frame_count,
        "decode_sw_s": round(decode_sw, 3),
        "decode_hw_s": round(decode_hw, 3),
        "encode_sw_s": round(encode_sw, 3),
        "encode_hw_s": round(encode_hw, 3),
    }
