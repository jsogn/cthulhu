"""GUI 后端服务层：把研究核心封装为可复用的业务操作。"""

from __future__ import annotations

import math
import os
import platform
import queue
import shutil
import subprocess
import tempfile
import threading
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import pairwise
from pathlib import Path

import numpy as np
from scipy.ndimage import zoom
from scipy.signal import resample_poly

from cthulhu_backend import db, samples
from cthulhu_backend.bitstream import analyze as bitstream_analyze
from cthulhu_backend.evaluate import metrics
from cthulhu_backend.fingerprint import adversarial
from cthulhu_backend.media import container, ffmpeg
from cthulhu_backend.numeric import channel_stats
from cthulhu_backend.similarity import embedding
from cthulhu_backend.transform import audio as audio_transform
from cthulhu_backend.transform import extra_attacks, shots, strategies
from cthulhu_backend.watermark import common as watermark_common
from cthulhu_backend.watermark import detect

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".flv", ".ts", ".webm", ".m4v"}
MAX_WORKING_BYTES = 512 * 1024**3 // 2  # 低内存机器兜底：256MB


def _memory_budget_bytes() -> int:
    """按物理内存自适应工作预算（256MB~32GB）。

    大内存按 55% 取用，小内存（<4GB）按 40% 取用，避免预算超过物理内存。
    """
    try:
        total = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, ValueError, OSError):
        total = 0
    if total <= 0:
        return MAX_WORKING_BYTES
    fraction = 0.55 if total >= 4 * 1024**3 else 0.4
    return max(MAX_WORKING_BYTES, min(32 * 1024**3, int(total * fraction)))

DEMO_LIBRARY_DIR = Path(os.environ.get("CTHULHU_DEMO_LIBRARY", str(Path(__file__).resolve().parents[2] / "data" / "demo-library")))
# 演示素材默认关闭：仅在显式设置 CTHULHU_DEMO_LIBRARY=1 时生成，
# 避免正式使用时素材库里出现无法删除的合成演示视频。
ENABLE_DEMO = os.environ.get("CTHULHU_DEMO_LIBRARY") == "1"

DEMO_MATERIALS = [
    ("短剧_第3集_片段A.mp4", 24, 480, 270, 101, 1.0),
    ("直播切片_美妆_0712.mp4", 30, 480, 270, 102, 1.5),
    ("产品展示_精华液.mp4", 18, 384, 216, 103, 0.8),
    ("短剧_第7集_高光.mp4", 22, 480, 270, 104, 1.8),
    ("达人素材_口播A.mp4", 28, 270, 480, 105, 1.2),
    ("千川_素材B_15s.mp4", 16, 480, 270, 106, 2.0),
    ("竞品参考_切片.mp4", 20, 480, 270, 107, 1.4),
    ("口播_剧情号_0812.mp4", 26, 384, 216, 108, 0.9),
]


def ensure_demo_library() -> int:
    """启动时生成演示素材库（幂等），返回可用视频数。"""
    if not ENABLE_DEMO:
        return 0
    DEMO_LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    for name, frames, width, height, seed, motion in DEMO_MATERIALS:
        target = DEMO_LIBRARY_DIR / name
        if target.exists() and target.stat().st_size > 0:
            continue
        video = samples.make_video_frames(frames, width, height, seed, motion_speed=motion)
        audio = samples.make_audio(frames / 30.0, seed=seed + 1000)
        ffmpeg.encode_video_with_audio(video, audio, str(target), fps=30)
    return len(list(DEMO_LIBRARY_DIR.glob("*.mp4")))


def demo_library() -> dict:
    """返回规范演示素材清单（固定顺序，不包含处理产物与预置报告）。"""
    if not ENABLE_DEMO:
        return {"directory": True, "files": []}
    files = []
    for name, *_ in DEMO_MATERIALS:
        path = DEMO_LIBRARY_DIR / name
        if not path.exists():
            continue
        files.append(
            {
                "path": str(path),
                "name": name,
                "size": path.stat().st_size,
                "video": ffmpeg.video_info(str(path)),
                "error": None,
                "report": None,
            }
        )
    return {"directory": True, "files": files, "valid": len(files), "invalid": 0}


def _require_file(path: str) -> str:
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"文件不存在：{path}")
    return str(target)


def run_detect(path: str, progress_cb=None, should_stop=None) -> dict:
    """检测：容器/元数据/SEI + 压缩域（码流层）分析，按阶段汇报进度。"""
    path = _require_file(path)

    def step(percent: int, note: str) -> None:
        if progress_cb:
            progress_cb(percent, note)

    def check_cancelled() -> None:
        if should_stop and should_stop():
            raise InterruptedError("任务已取消")

    check_cancelled()
    step(5, "读取视频信息")
    probe = ffmpeg.video_info(path)
    check_cancelled()
    step(20, "扫描容器元数据")
    container_report = container.scan_mp4(path)
    sei_count = container.count_sei(path)
    check_cancelled()
    step(50, "码流层分析")
    bitstream = bitstream_analyze.analyze(path, max_frames=300)
    report = {
        "probe": probe,
        "container": container_report,
        "sei_count": sei_count,
        "bitstream": bitstream,
    }
    check_cancelled()
    step(78, "盲检测抽样")
    try:
        # 盲检测仅抽样前 300 帧，避免长视频全量解码拖垮导入。
        frames, _ = ffmpeg.decode_video(path, max_frames=300)
        check_cancelled()
        step(90, "音频分析")
        blind: dict = detect.video_scores(frames)
        # 检测只做抽样，取前 120 秒音轨即可，避免长视频整段解码。
        audio = ffmpeg.decode_audio(path, max_seconds=120)
        check_cancelled()
        if audio is not None:
            signal, sample_rate = audio
            blind["echo"] = detect.audio_scores(signal, sample_rate)["echo"]
        report["blind"] = blind
    except InterruptedError:
        raise
    except Exception:  # noqa: BLE001 - 盲检测失败不影响压缩域报告
        report["blind"] = None
    step(95, "写入报告")
    db.save_report(path, report)
    return report


def run_blind(path: str) -> dict | None:
    """仅盲检测抽样：空间/频域置信度与音频回声（清洗产物残留复检用）。"""
    try:
        frames, _ = ffmpeg.decode_video(path, max_frames=300)
        blind: dict = detect.video_scores(frames)
        audio = ffmpeg.decode_audio(path, max_seconds=120)
        if audio is not None:
            signal, sample_rate = audio
            blind["echo"] = detect.audio_scores(signal, sample_rate)["echo"]
        return blind
    except Exception:  # noqa: BLE001 - 残留复检失败不影响任务成功状态
        return None


def _temporal_aligned_vmaf(
    ref: np.ndarray,
    mov: np.ndarray,
    matches: np.ndarray,
    sample_pairs: int = 8,
    fps: float = 30.0,
) -> float | None:
    """时间对齐 VMAF：把处理帧匹配回最近原始帧，抽样重编码后评分。"""
    if len(mov) < 2:
        return None
    indices = np.linspace(0, len(mov) - 1, min(sample_pairs, len(mov))).astype(int)
    ref_pairs: list[np.ndarray] = []
    mov_pairs: list[np.ndarray] = []
    for index in indices:
        candidate = mov[index]
        reference = ref[int(matches[index])]
        if candidate.shape != reference.shape:
            factors = (
                reference.shape[0] / candidate.shape[0],
                reference.shape[1] / candidate.shape[1],
            )
            candidate = zoom(candidate, factors, order=1)
        ref_pairs.append(reference)
        mov_pairs.append(candidate)
    with tempfile.TemporaryDirectory() as tmp:
        ref_path = os.path.join(tmp, "ref.mp4")
        dist_path = os.path.join(tmp, "dist.mp4")
        ffmpeg.encode_video(np.asarray(ref_pairs), ref_path, fps=fps, crf=18)
        ffmpeg.encode_video(np.asarray(mov_pairs), dist_path, fps=fps, crf=18)
        return ffmpeg.vmaf_score(dist_path, ref_path)


def _prefetch_batches(
    decoder: ffmpeg.StreamingDecoder,
    seg_len: int,
    chunk: int,
    dtype: str,
):
    """顺序解码预取生成器：后台线程提前读下一块，隐藏解码等待。

    生产者把解码结果放入有界队列（最多 2 块在途），主线程边变换边消费，
    使解码与变换/编码重叠；块序与逐块同步读取完全一致。
    """
    batch_queue: queue.Queue = queue.Queue(maxsize=1)

    def producer() -> None:
        try:
            pos = 0
            while pos < seg_len:
                batch = decoder.read(min(chunk, seg_len - pos), dtype=dtype)
                if len(batch) == 0:
                    break
                batch_queue.put((pos, batch))
                pos += len(batch)
        except BaseException as exc:  # noqa: BLE001 - 异常传给主线程统一处理
            batch_queue.put(exc)
        finally:
            batch_queue.put(None)

    threading.Thread(target=producer, daemon=True).start()
    while True:
        item = batch_queue.get()
        if item is None:
            break
        if isinstance(item, BaseException):
            raise item
        yield item


def _requant_yuv(yuv: np.ndarray, levels: int) -> np.ndarray:
    """YUV420p 紧凑平面上的像素重量化（与 RGB 重量化同口径量化步长）。"""
    if levels <= 1:
        return yuv
    step = 255.0 / (levels - 1)
    return (np.round(yuv.astype(np.float32) / step) * step).round().astype(np.uint8)


def _process_segment_yuv(
    path: str,
    start_frame: int,
    count: int,
    output_path: str,
    width: int,
    height: int,
    fps: float,
    *,
    chunk: int,
    requant_eff: int,
    phash_attack: bool,
    phash_epsilon: float,
    phash_iters: int,
    attack_workers: int | None,
    filters: list[str] | None,
    codec: str,
    hardware: bool,
    crf: int,
    preset: str,
    gop: int | None,
    threads: int | None,
    progress=None,
    stop=None,
) -> int:
    """单段 YUV420p 快路径：解码→重量化/签名攻击→编码到独立文件。"""
    y_plane = height * width
    decoder = ffmpeg.StreamingDecoder(path, start_frame, count, pix_fmt="yuv420p")
    encoder = ffmpeg.StreamingEncoder(
        output_path,
        width,
        height,
        fps,
        codec=codec,
        hardware=hardware,
        crf=crf,
        preset=preset,
        gop=gop,
        color=True,
        filters=filters,
        input_pix_fmt="yuv420p",
        threads=threads,
        stop=stop,
    )
    out_count = 0
    try:
        for _, batch in _prefetch_batches(decoder, count, chunk, "uint8"):
            if stop and stop():
                raise InterruptedError("任务已取消")
            if requant_eff > 0:
                batch = _requant_yuv(batch, requant_eff)
            if phash_attack:
                if not batch.flags.writeable:
                    batch = np.array(batch, dtype=np.uint8, copy=True)
                y = batch[:, :y_plane].reshape(len(batch), height, width)
                attacked = adversarial.attack_frames(
                    y, epsilon=phash_epsilon, iterations=phash_iters, workers=attack_workers
                )
                batch[:, :y_plane] = attacked.reshape(len(batch), -1)
            encoder.write(batch)
            out_count += len(batch)
            if progress:
                progress(len(batch))
    finally:
        decoder.close()
        encoder.finish()
    return out_count


def _concat_video_segments(segments: list[str], output: str) -> None:
    """把同参数字视频段用 concat demuxer 无损拼接（-c copy）。"""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
        for segment in segments:
            escaped = segment.replace("'", "'\\''")
            handle.write(f"file '{escaped}'\n")
        list_path = handle.name
    try:
        subprocess.run(
            [
                ffmpeg.FFMPEG_BIN, "-y", "-v", "error",
                "-f", "concat", "-safe", "0", "-i", list_path,
                "-c", "copy", "-an", output,
            ],
            capture_output=True,
            check=True,
        )
    finally:
        try:
            os.unlink(list_path)
        except OSError:
            pass


def _transcode_chain(path: str, final_codec: str, check_cancelled) -> None:
    """编码域组合拳：跨 codec 二次转码，破坏量化/GOP 域的脆弱相关。

    第一遍换 codec（H.264→H.265，不可用则同 codec）高 CRF 粗量化，
    第二遍转回目标 codec。音轨直接复制。产物替换回原路径。
    """
    mid = path + ".chain.mp4"
    final = path + ".final.mp4"
    mid_codec = (
        "libx265" if final_codec != "libx265" and ffmpeg.has_encoder("libx265") else final_codec
    )
    try:
        subprocess.run(
            [
                ffmpeg.FFMPEG_BIN, "-y", "-v", "error",
                "-i", path, "-c:v", mid_codec, "-preset", "veryfast",
                "-crf", "28", "-c:a", "copy", mid,
            ],
            capture_output=True,
            check=True,
        )
        check_cancelled()
        subprocess.run(
            [
                ffmpeg.FFMPEG_BIN, "-y", "-v", "error",
                "-i", mid, "-c:v", final_codec, "-preset", "veryfast",
                "-crf", "23", "-c:a", "copy", final,
            ],
            capture_output=True,
            check=True,
        )
        check_cancelled()
        os.replace(final, path)
    finally:
        for leftover in (mid, final):
            try:
                os.unlink(leftover)
            except OSError:
                pass


def run_similarity(a: str, b: str) -> dict:
    a, b = _require_file(a), _require_file(b)
    frames_a, _ = ffmpeg.decode_video(a)
    frames_b, _ = ffmpeg.decode_video(b)
    return embedding.similarity_report(frames_a, frames_b)


def scan_import_path(path: str) -> dict:
    """递归扫描导入目标（文件或目录），返回视频文件清单与有效性统计。"""
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"路径不存在：{path}")
    if target.is_file():
        entries = [target]
    else:
        entries = sorted(
            p for p in target.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
        )
    files = []
    for entry in entries:
        item: dict = {
            "path": str(entry),
            "name": entry.name,
            "size": entry.stat().st_size,
            "video": None,
            "error": None,
        }
        try:
            item["video"] = ffmpeg.video_info(str(entry))
        except Exception as exc:  # noqa: BLE001 - 单个文件失败不影响整批
            item["error"] = str(exc)
        files.append(item)
    return {
        "directory": target.is_dir(),
        "files": files,
        "valid": sum(1 for item in files if item["video"]),
        "invalid": sum(1 for item in files if not item["video"]),
    }


def run_desensitize(
    path: str,
    output: str,
    *,
    reorder: bool = False,
    speed: float = 1.0,
    recrop: float = 0.0,
    regrade: bool = True,
    perturb: float = 0.0,
    audio_remix: bool = True,
    sharpness: bool = True,
    color_restore: bool = True,
    denoise: bool = False,
    anti_reembed: bool = False,
    banner: str = "",
    seed: int = 0,
    codec: str = "libx264",
    lossless: bool = False,
    preset: str = "veryfast",
    spoof: bool = False,
    bitrate_kbps: int | None = None,
    gop: int | None = None,
    resolution: str | None = None,
    fps_out: float | None = None,
    hardware: bool = False,
    transform_strategy: str | None = None,
    phash_attack: bool = False,
    phash_epsilon: float = 0.03,
    phash_iters: int = 120,
    multi_hash_attack: bool = False,
    shot_retime: bool = False,
    shot_retime_min: float = 0.98,
    shot_retime_max: float = 1.04,
    cut_jitter: int = 0,
    audio_strong: bool = False,
    echo_defeat: bool = False,
    skip_vmaf: bool = False,
    filter_scale: int = 0,
    rotate: float = 0.0,
    transcode_chain: bool = False,
    median: int = 0,
    noise: float = 0.0,
    requant: int = 0,
    dct_step: float = 0.0,
    drop_every: int = 0,
    jitter: float = 0.0,
    perspective: float = 0.0,
    warp: float = 0.0,
    mirror: bool = False,
    subtract_beta: float = 0.0,
    saliency: int = 0,
    chroma_levels: int = 0,
    native_filters: bool = False,
    detail_protect: float = 0.0,
    progress_cb=None,
    should_stop=None,
) -> dict:
    """内容脱敏：分析遍 + 分块流式处理遍，内存与视频总长度解耦。"""
    path = _require_file(path)
    output = os.path.expanduser(output)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    def check_cancelled() -> None:
        if should_stop and should_stop():
            raise InterruptedError("任务已取消")

    if progress_cb:
        progress_cb(5, "准备处理")
    if not ffmpeg.has_encoder(codec):
        raise ValueError(f"当前 FFmpeg 缺少视频编码器 {codec}，请更换输出编码或安装完整版 FFmpeg")
    info = ffmpeg.video_info(path)
    fps_in = float(info["fps"])
    total_in = max(1, round(info["duration"] * fps_in))
    check_cancelled()

    # 回声水印清除：音画按同一 factor 同步放慢，音频用 atempo 保调拉伸，
    # 移动回声时延以破坏检测，同时保持音画内容对齐。
    if echo_defeat:
        speed = speed * 0.97
        if speed < 0.5:
            raise ValueError("echo_defeat 需要有效变速 factor ≥ 0.5")
        audio_tempo = speed
    else:
        audio_tempo = None

    # ---------- 分析遍：抽样镜头边界 + 预生成逐帧随机参数 ----------
    if progress_cb:
        progress_cb(8, "扫描镜头结构")
    need_shots = reorder or shot_retime or cut_jitter > 0
    # 分析抽样只为镜头检测与色彩还原服务；两者都关闭时跳过整段抽样解码。
    if need_shots or color_restore:
        sampled, _, sampled_starts = ffmpeg.decode_sampled(path, cap=400, return_starts=True)
    else:
        sampled = np.empty((0,), dtype=np.float32)
        sampled_starts = [0]
    if need_shots and len(sampled) > 2 and sampled_starts:
        # 抽样按窗口返回；逐窗口检测切点再映射回全局帧号，避免窗口拼接处的假切点。
        per_window = max(1, len(sampled) // len(sampled_starts))
        boundaries = []
        for window, start in enumerate(sampled_starts):
            segment = sampled[window * per_window : (window + 1) * per_window]
            if len(segment) < 2:
                continue
            for cut in shots.detect_cuts(segment):
                if 0 < cut < len(segment):
                    boundaries.append(start + cut)
        boundaries = sorted({boundary for boundary in boundaries if 0 < boundary < total_in})
        boundaries = [0] + boundaries + [total_in]
    else:
        boundaries = [0, total_in]
    shot_ranges = list(pairwise(boundaries))

    rng = np.random.default_rng(seed)
    # 逐镜头对抗参数按原镜头序号确定性生成：切点漂移 + 逐镜头变速因子。
    shot_offsets: list[tuple[int, int]] = []
    shot_factors: list[float] = []
    for start, end in shot_ranges:
        length = end - start
        drop_start = int(rng.integers(0, cut_jitter + 1)) if cut_jitter > 0 else 0
        drop_end = int(rng.integers(0, cut_jitter + 1)) if cut_jitter > 0 else 0
        if drop_start + drop_end >= length:
            drop_start = min(drop_start, max(0, length - 1))
            drop_end = 0
        shot_offsets.append((drop_start, drop_end))
        shot_factors.append(
            float(rng.uniform(shot_retime_min, shot_retime_max)) if shot_retime else 1.0
        )
    order = rng.permutation(len(shot_ranges)) if reorder else np.arange(len(shot_ranges))
    # 重排后的输入区间；seg_factors 为该段最终变速因子（全局 speed × 逐镜头）。
    segments: list[tuple[int, int, int]] = []
    seg_factors: list[float] = []
    seg_out_lens: list[int] = []
    cursor = 0
    for shot_index in order:
        orig_start, orig_end = shot_ranges[int(shot_index)]
        drop_start, drop_end = shot_offsets[int(shot_index)]
        eff_start = orig_start + drop_start
        eff_len = (orig_end - orig_start) - drop_start - drop_end
        factor = speed * shot_factors[int(shot_index)]
        retimed = speed != 1.0 or shot_retime
        out_len = max(1, round(eff_len / factor)) if retimed else eff_len
        segments.append((cursor, eff_start, eff_len))
        seg_factors.append(factor)
        seg_out_lens.append(out_len)
        cursor += out_len
    total_out = cursor

    gammas = np.ones(total_out, dtype=np.float32)
    deltas = np.zeros(total_out, dtype=np.float32)
    if regrade:
        gamma_strength = 0.03 + 0.2 * perturb
        brightness = 0.02 + 0.06 * perturb
        # 时间平滑：调光参数沿低频轨迹变化，避免逐帧独立随机造成的暗部闪烁。
        # 幅度与旧实现一致（gamma ±gamma_strength、亮度 ±brightness），对抗
        # 语义不变，只是相邻帧连续过渡。
        t = np.arange(total_out, dtype=np.float32)
        period_a = float(rng.uniform(80.0, 180.0))
        period_b = float(rng.uniform(80.0, 180.0))
        period_c = float(rng.uniform(80.0, 180.0))
        period_d = float(rng.uniform(80.0, 180.0))
        phase_a = float(rng.uniform(0.0, 2.0 * np.pi))
        phase_b = float(rng.uniform(0.0, 2.0 * np.pi))
        phase_c = float(rng.uniform(0.0, 2.0 * np.pi))
        phase_d = float(rng.uniform(0.0, 2.0 * np.pi))
        gammas = 1.0 + gamma_strength * (
            0.6 * np.sin(2.0 * np.pi * t / period_a + phase_a)
            + 0.4 * np.sin(2.0 * np.pi * t / period_b + phase_b)
        ).astype(np.float32)
        deltas = brightness * (
            0.6 * np.sin(2.0 * np.pi * t / period_c + phase_c)
            + 0.4 * np.sin(2.0 * np.pi * t / period_d + phase_d)
        ).astype(np.float32)

    mid_rng = np.random.default_rng(seed)
    spoof_bits = None
    if spoof:
        spoof_rng = np.random.default_rng(seed ^ 0x5F3759DF)
        spoof_bits = watermark_common.payload_bits(int(spoof_rng.integers(0, 2**31)), 64)
    assault_rng = np.random.default_rng(seed ^ 0xA55A55A5)
    # FFmpeg 原生滤镜迁移：可平移的原语始终交给编码器滤镜链，numpy 侧跳过，
    # 不依赖对抗档位开关，默认路径即可获得成倍提速。
    native_chain: list[str] = []
    sharpness_eff = sharpness
    noise_eff = noise
    requant_eff = requant
    denoise_eff = denoise
    # 几何/调光下沉到编码器滤镜链：仅在无重排、无变速的快路径上启用，
    # 避免输出帧号与原帧号不一致时表达式错位；其余路径保持 numpy 实现。
    use_native_geometry = (
        not reorder and speed == 1.0 and not shot_retime and cut_jitter == 0
    )
    recrop_eff = recrop
    regrade_eff = regrade
    rotate_eff = rotate
    if use_native_geometry:
        if recrop > 0 and ffmpeg.has_filter("crop") and ffmpeg.has_filter("scale"):
            recrop_eff = 0.0
        if regrade and ffmpeg.has_filter("eq"):
            regrade_eff = False
        if rotate > 0 and ffmpeg.has_filter("rotate"):
            rotate_eff = 0.0

    def geometry_chain(offset: int) -> list[str]:
        """几何/调光原生滤镜链；offset 用于分段并行时保持全局帧号连续。"""
        chain: list[str] = []
        if not use_native_geometry:
            return chain
        frame_var = f"(n+{offset})"
        if recrop > 0 and ffmpeg.has_filter("crop") and ffmpeg.has_filter("scale"):
            frame_w, frame_h = info["width"], info["height"]
            crop_x = int(frame_w * recrop)
            # 只裁左/下两边，保护顶部与右侧（短剧剧名常见位置）；
            # crop_y 按同比例推导，保证 x/y 缩放一致不变形。
            crop_y = round(frame_h * crop_x / frame_w)
            crop_w = max(2, frame_w - crop_x)
            crop_h = max(2, frame_h - crop_y)
            if crop_w % 2:
                crop_w -= 1
            if crop_h % 2:
                crop_h -= 1
            chain.append(
                f"crop={crop_w}:{crop_h}:{crop_x}:0,scale={frame_w}:{frame_h}:flags=bilinear"
            )
        if filter_scale > 0:
            frame_w, frame_h = info["width"], info["height"]
            scale_h = max(2, round(frame_h * filter_scale / frame_w))
            chain.append(f"scale={filter_scale}:{scale_h}:flags=bicubic")
        if regrade and ffmpeg.has_filter("eq"):
            gamma_expr = (
                f"1+{gamma_strength:.5f}*(0.6*sin(2*PI*{frame_var}/{period_a:.1f}+{phase_a:.4f})"
                f"+0.4*sin(2*PI*{frame_var}/{period_b:.1f}+{phase_b:.4f}))"
            )
            brightness_expr = (
                f"{brightness:.5f}*(0.6*sin(2*PI*{frame_var}/{period_c:.1f}+{phase_c:.4f})"
                f"+0.4*sin(2*PI*{frame_var}/{period_d:.1f}+{phase_d:.4f}))"
            )
            chain.append(f"eq=gamma='{gamma_expr}':brightness='{brightness_expr}'")
        if rotate > 0 and ffmpeg.has_filter("rotate"):
            # numpy 侧为 max_angle(度) * sin(2π n / 200)；ffmpeg rotate 用弧度。
            angle_rad = rotate * np.pi / 180.0
            chain.append(
                f"rotate=a='{angle_rad:.6f}*sin(2*PI*{frame_var}/200)':c=black:bilinear=1"
            )
        return chain

    base_filters: list[str] = []
    if sharpness and ffmpeg.has_filter("unsharp"):
        base_filters.append("unsharp=5:5:0.25:3:3:0.0")
        sharpness_eff = False
    if denoise and ffmpeg.has_filter("removegrain"):
        base_filters.append("removegrain=4")
        denoise_eff = False
    if noise > 0 and ffmpeg.has_filter("noise"):
        base_filters.append(f"noise=alls={max(1, round(noise * 255))}:allf=t")
        noise_eff = 0.0
    if requant > 0 and ffmpeg.has_filter("posterize"):
        base_filters.append(f"posterize={requant}")
        requant_eff = 0
    if filter_scale > 0 and use_native_geometry:
        base_filters.append(f"scale={info['width']}:{info['height']}:flags=bicubic")
    native_chain = geometry_chain(0) + base_filters
    assault_params = extra_attacks.AssaultParams(
        mirror=mirror,
        jitter=jitter,
        perspective=perspective,
        warp=warp,
        median=median,
        noise=noise_eff,
        subtract_beta=subtract_beta,
        requant=requant_eff,
        dct_step=dct_step,
        chroma_levels=chroma_levels,
        drop_every=drop_every,
    )
    saliency_obj = extra_attacks.saliency_layout(seed, saliency)

    if color_restore:
        color_sampled, _ = ffmpeg.decode_sampled(path, cap=40, grayscale=False)
        ref_mean, ref_std = channel_stats(color_sampled)
        del color_sampled
    else:
        ref_mean = np.zeros(3, dtype=np.float32)
        ref_std = np.zeros(3, dtype=np.float32)
    del sampled

    # 变换策略：与分块无关的选项与上下文一次性组装，分块内只调用 apply。
    strategy = strategies.get_strategy(transform_strategy)
    transform_options = strategies.TransformOptions(
        recrop=recrop_eff,
        regrade=regrade_eff,
        anti_reembed=anti_reembed,
        color_restore=color_restore,
        sharpness=sharpness_eff,
        denoise=denoise_eff,
        spoof=spoof,
    )
    transform_context = strategies.TransformContext(
        gammas=gammas,
        deltas=deltas,
        banner=banner,
        seed=seed,
        ref_mean=ref_mean,
        ref_std=ref_std,
        mid_rng=mid_rng,
        spoof_bits=spoof_bits,
    )

    crf = 0 if lossless else 23
    if lossless:
        # 无损档以 CRF=0 为准，显式码率与无损互斥，避免被编码器忽略成非无损。
        bitrate_kbps = None
    output_fps = fps_out or fps_in
    out_size = None
    if resolution and "x" in resolution:
        try:
            width, height = (int(part) for part in resolution.split("x", 1))
            out_size = (width - width % 2, height - height % 2)
        except ValueError:
            out_size = None

    # 分块大小：按内存预算自适应（float32 每帧 4 字节，预留 8 倍中间量余量）。
    budget = _memory_budget_bytes()
    adaptive_parallelism = min(4, max(2, os.cpu_count() or 2))
    try:
        concurrency = max(1, int(db.load_settings().get("parallelism", adaptive_parallelism)))
    except (TypeError, ValueError):
        concurrency = adaptive_parallelism
    # 预算按并行任务数均分，保证并行度再高也不会叠加超内存。
    budget //= concurrency
    frame_dtype = getattr(strategy, "frame_dtype", "float32")
    frame_bytes = info["width"] * info["height"] * 3 * (1 if frame_dtype == "uint8" else 4)
    # 中间量按 12 倍预留余量，分块上限 240，低内存机器自动变小块。
    chunk = max(8, min(240, int(budget // 12 // max(frame_bytes, 1))))
    # pHash/多哈希对抗在批级持有大量 float32 中间量，收窄分块避免长片 OOM。
    if phash_attack or multi_hash_attack:
        chunk = min(chunk, 48)

    # 音轨独立准备（整段重混后统一 mux）。
    audio_signal = None
    sample_rate = 16000
    if audio_remix:
        decoded_audio = ffmpeg.decode_audio(path)
        if decoded_audio is not None:
            signal, sample_rate = decoded_audio
            audio_rng = np.random.default_rng(seed ^ 0x9E3779B9)
            if audio_strong:
                signal = audio_transform.remix_strong(
                    signal, sample_rate, audio_rng,
                    pitch_ratio=0.985, eq_db=4.0, noise_floor=0.003,
                )
            else:
                # 等长重混：不改内容时间线，音画同步只由末尾按实际帧数对齐兜底。
                signal = audio_transform.remix(signal, sample_rate, audio_rng)
            audio_signal = signal

    # ---------- 处理遍：逐块解码 → 变换 → 流式编码 ----------
    temp_video = output + ".video.mp4"
    # YUV420p 快路径：仅当所有像素变换都已下沉原生、且无任何 RGB 专属
    # 选项时启用，原始帧体积减半、省去 RGB↔YUV 转换。
    use_yuv_path = (
        use_native_geometry
        and getattr(strategy, "name", "") == "fast"
        and not color_restore
        and not banner
        and not anti_reembed
        and not spoof
        and detail_protect <= 0
        and saliency == 0
        and not mirror
        and jitter <= 0
        and perspective <= 0
        and warp <= 0
        and median <= 0
        and subtract_beta <= 0
        and dct_step <= 0
        and chroma_levels <= 0
        and drop_every <= 0
        and not multi_hash_attack
    )
    # YUV 快路径自建分段/单段编码器写视频轨；RGB 路径用共享编码器。
    encoder = (
        None
        if use_yuv_path
        else ffmpeg.StreamingEncoder(
            temp_video,
            info["width"],
            info["height"],
            output_fps,
            codec=codec,
            hardware=hardware,
            crf=crf,
            preset=preset,
            gop=gop,
            bitrate_kbps=bitrate_kbps,
            out_size=out_size,
            color=True,
            filters=native_chain or None,
            stop=should_stop,
        )
    )
    out_index = 0

    def apply_attacks(frames, output_ids):
        """pHash 攻击、几何去同步与细节保护等与分块无关的后置变换。"""
        protected = frames.copy() if detail_protect > 0 else None
        if rotate_eff > 0:
            frames = strategies.rotate_de_sync(frames, output_ids, rotate_eff)
        if assault_params.enabled:
            frames = extra_attacks.apply(frames, assault_params, assault_rng)
        if phash_attack:
            frames = adversarial.attack_frames(
                frames, epsilon=phash_epsilon, iterations=phash_iters
            )
        elif multi_hash_attack:
            frames = adversarial.attack_frames_joint(
                frames, epsilon=phash_epsilon, iterations=phash_iters
            )
        if protected is not None:
            frames = extra_attacks.protect_details(frames, protected, detail_protect)
        if saliency_obj is not None:
            frames = extra_attacks.salient_overlay(frames, saliency_obj)
        return frames

    try:
        if use_yuv_path:
            # YUV420p 快路径：解码/写回体积减半，静态变换全在编码器滤镜链，
            # numpy 侧只剩像素重量化与 Y 平面签名攻击。长片按 CPU 核数分
            # 2~3 段并发编码（滤镜链单线程是瓶颈），段间用帧号偏移保持
            # eq/rotate 轨迹连续，任一段失败回退串行。
            frame_h, frame_w = info["height"], info["width"]
            cpu = os.cpu_count() or 1
            max_k = 3 if cpu >= 12 else (2 if cpu >= 8 else 1)
            k = max(1, min(max_k, total_out // 800))
            if total_out < 2400:
                k = 1
            override_k = os.environ.get("CTHULHU_SEGMENT_K")
            if override_k:
                try:
                    k = max(1, min(int(override_k), max_k, total_out))
                except ValueError:
                    pass
            if k <= 1:
                out_index = _process_segment_yuv(
                    path, 0, total_out, temp_video,
                    frame_w, frame_h, output_fps,
                    chunk=chunk, requant_eff=requant_eff, phash_attack=phash_attack,
                    phash_epsilon=phash_epsilon, phash_iters=phash_iters,
                    attack_workers=None, filters=native_chain, codec=codec, hardware=hardware,
                    crf=crf, preset=preset, gop=gop, threads=None,
                    stop=should_stop,
                )
            else:
                seg_paths = [f"{temp_video}.seg{index}.mp4" for index in range(k)]
                seg_len = math.ceil(total_out / k)
                seg_threads = max(2, cpu // k)
                done = [0]
                progress_lock = threading.Lock()

                def seg_progress(delta: int) -> None:
                    with progress_lock:
                        done[0] += delta
                        current = done[0]
                    if progress_cb and total_out:
                        progress_cb(
                            8 + int(current / total_out * 82),
                            f"处理中 {current}/{total_out} 帧",
                        )

                try:
                    with ThreadPoolExecutor(max_workers=k) as pool:
                        futures = []
                        for index in range(k):
                            start = index * seg_len
                            count = min(seg_len, total_out - start)
                            filters = geometry_chain(start) + base_filters
                            futures.append(
                                pool.submit(
                                    _process_segment_yuv,
                                    path, start, count, seg_paths[index],
                                    frame_w, frame_h, output_fps,
                                    chunk=chunk, requant_eff=requant_eff,
                                    phash_attack=phash_attack,
                                    phash_epsilon=phash_epsilon,
                                    phash_iters=phash_iters, attack_workers=2,
                                    filters=filters,
                                    codec=codec, hardware=hardware, crf=crf,
                                    preset=preset, gop=gop, threads=seg_threads,
                                    progress=seg_progress if progress_cb else None,
                                    stop=should_stop,
                                )
                            )
                        for future in as_completed(futures):
                            out_index += future.result()
                    _concat_video_segments(seg_paths, temp_video)
                except InterruptedError:
                    raise
                except BaseException:  # noqa: BLE001 - 段级失败统一回退串行
                    # 任一段失败回退串行，保证任务可用；清理半成品段文件。
                    for seg_path in seg_paths:
                        try:
                            os.unlink(seg_path)
                        except OSError:
                            pass
                    out_index = _process_segment_yuv(
                        path, 0, total_out, temp_video,
                        frame_w, frame_h, output_fps,
                        chunk=chunk, requant_eff=requant_eff, phash_attack=phash_attack,
                        phash_epsilon=phash_epsilon, phash_iters=phash_iters,
                        attack_workers=None, filters=native_chain, codec=codec, hardware=hardware,
                        crf=crf, preset=preset, gop=gop, threads=None,
                        stop=should_stop,
                    )
                else:
                    for seg_path in seg_paths:
                        try:
                            os.unlink(seg_path)
                        except OSError:
                            pass
        elif speed == 1.0 and not shot_retime and cut_jitter == 0:
            # 默认快路径：按重排后的镜头顺序流式处理，每个镜头只 seek 一次，
            # 消除逐块 decode_video_range 从头重复解码丢弃的 O(N²) 开销。
            for seg_start, orig_start, seg_len in segments:
                check_cancelled()
                decoder = ffmpeg.StreamingDecoder(path, orig_start, seg_len, grayscale=False)
                try:
                    for pos, batch in _prefetch_batches(decoder, seg_len, chunk, frame_dtype):
                        check_cancelled()
                        output_ids = list(range(seg_start + pos, seg_start + pos + len(batch)))
                        frames = strategy.apply(
                            batch, output_ids, transform_context, transform_options
                        )
                        frames = apply_attacks(frames, output_ids)
                        encoder.write(frames)
                        out_index += len(frames)
                        if progress_cb and total_out:
                            progress_cb(
                                8 + int(out_index / total_out * 82),
                                f"处理中 {out_index}/{total_out} 帧",
                            )
                finally:
                    decoder.close()
        else:
            # O(N) 流式变速路径：逐镜头因子映射，每个镜头只 seek 一次。
            # 避免逐块随机访问解码的 O(N²) 开销。
            for si, (seg_start, orig_start, seg_len) in enumerate(segments):
                check_cancelled()
                factor = seg_factors[si]
                seg_out_len = seg_out_lens[si]
                seg_out_index = 0
                decoder = ffmpeg.StreamingDecoder(path, orig_start, seg_len, grayscale=False)
                try:
                    for pos, batch in _prefetch_batches(decoder, seg_len, chunk, frame_dtype):
                        batch_len = len(batch)
                        output_ids: list[int] = []
                        while seg_out_index < seg_out_len:
                            source = int(np.floor(seg_out_index * factor))
                            if source >= pos + batch_len:
                                break
                            output_ids.append(seg_start + seg_out_index)
                            seg_out_index += 1
                        if output_ids:
                            sources = np.clip(
                                np.floor(
                                    (np.asarray(output_ids) - seg_start) * factor
                                ).astype(np.int64)
                                - pos,
                                0,
                                batch_len - 1,
                            )
                            frames = np.asarray(batch)[sources]
                            frames = strategy.apply(
                                frames, output_ids, transform_context, transform_options
                            )
                            frames = apply_attacks(frames, output_ids)
                            encoder.write(frames)
                            out_index += len(frames)
                            if progress_cb and total_out:
                                progress_cb(
                                    8 + int(out_index / total_out * 82),
                                    f"处理中 {out_index}/{total_out} 帧",
                                )
                finally:
                    decoder.close()
        if encoder is not None:
            encoder.finish()
        if audio_signal is not None:
            # 以实际写出的帧数对齐音轨时长：等长重混下仅当视频做统一变速
            # 时才需要拉伸，且拉伸比例与视频统一变速因子一致，保持内容对齐。
            if not echo_defeat:
                target_len = round(out_index / output_fps * sample_rate)
                if target_len != len(audio_signal):
                    audio_signal = resample_poly(audio_signal, target_len, len(audio_signal))
            audio_payload = (
                (np.clip(audio_signal, -1, 1) * 32767).round().astype(np.int16).tobytes()
            )
            mux_cmd = [
                ffmpeg.FFMPEG_BIN, "-y", "-v", "error",
                "-i", temp_video,
                "-f", "s16le", "-ar", str(sample_rate), "-ac", "1", "-i", "-",
            ]
            if echo_defeat:
                # atempo 保调拉伸放在 mux 滤镜里完成，与视频 factor 一致，
                # Python 侧不重采样，避免二次变速造成漂移。
                mux_cmd += ["-af", f"atempo={audio_tempo:.5f}"]
            mux_cmd += ["-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "-shortest", output]
            subprocess.run(
                mux_cmd,
                input=audio_payload,
                capture_output=True,
                check=True,
            )
            os.unlink(temp_video)
        else:
            os.replace(temp_video, output)
        if progress_cb:
            progress_cb(95, "编码完成")
        check_cancelled()
        if transcode_chain:
            _transcode_chain(output, codec, check_cancelled)
    except BaseException:
        if encoder is not None:
            encoder.abort()
        try:
            os.unlink(temp_video)
        except OSError:
            pass
        raise

    # ---------- 指标遍：抽样重解码后计算，避免整片驻留 ----------
    if progress_cb:
        progress_cb(97, "计算画质指标")
    original_sampled, _ = ffmpeg.decode_sampled(path, cap=120)
    processed_sampled, _ = ffmpeg.decode_sampled(output, cap=120)
    ref_s, mov_s, matches = metrics.temporal_match(original_sampled, processed_sampled)
    stability_in = metrics.temporal_stability(original_sampled)
    stability_out = metrics.temporal_stability(processed_sampled)
    # 几何去同步（旋转）使逐像素画质指标失去对齐口径，数值会误导用户。
    quality_na = rotate > 0
    return {
        "input": path,
        "output": output,
        "transform_strategy": strategy.name,
        "preset": preset,
        "frames": total_in,
        "order_disruption": round(
            metrics.order_disruption(original_sampled, processed_sampled), 4,
        ),
        # 与自身比较恒为 1，直接给出常量，省去一次全量 embedding。
        "similarity_before": {
            "content_cosine": 1.0,
            "motion_cosine": 1.0,
            "dhash_agreement": 1.0,
            "ssim_mean": 1.0,
            "reduction": {"content": 0.0, "motion": 0.0, "dhash": 0.0},
        },
        "similarity_after": embedding.similarity_report(original_sampled, processed_sampled),
        "psnr_db": None if quality_na else round(metrics.matched_psnr(ref_s, mov_s, matches), 2),
        "ssim": None if quality_na else round(metrics.matched_ssim(ref_s, mov_s, matches), 4),
        # 重排/变速后时间轴错位，朴素 VMAF 恒近 0 无参考价值，改用对齐分。
        "vmaf": None,
        "vmaf_aligned": (
            None
            if (quality_na or skip_vmaf)
            else _temporal_aligned_vmaf(ref_s, mov_s, matches, fps=output_fps)
        ),
        "quality_metrics_na": quality_na,
        "stability_ratio": round(stability_out / max(stability_in, 1e-9), 3),
        "export_health": _export_health(output),
    }


def run_repair(
    path: str,
    output: str,
    regions: list[dict],
    crf: int = 23,
    progress_cb=None,
    stop=None,
    pause=None,
) -> dict:
    """可见水印区域修复（FFmpeg delogo）。"""
    path = _require_file(path)
    output = os.path.expanduser(output)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    ffmpeg.repair_delogo(path, output, regions, crf, progress_cb=progress_cb, stop=stop, pause=pause)
    if stop and stop():
        raise InterruptedError("任务已取消")
    return {
        "input": path,
        "output": output,
        "regions": len(regions),
        "vmaf": None if (stop and stop()) else ffmpeg.vmaf_score(output, path),
        "vmaf_aligned": None if (stop and stop()) else ffmpeg.aligned_vmaf(output, path),
    }


def export_outputs(paths: list[str], export_dir: str) -> dict:
    """把源视频已生成的清洗/修复产物复制到导出目录，并导出产物记录清单。

    产物按清洗优先、修复其次的规则匹配，同一素材存在多个产物时取最新修改的。
    源视频尚未处理时归入 missing，由前端提示用户先执行清洗或修复。
    """
    export_root = Path(os.path.expanduser(export_dir or "~/导出/暗水印清洗"))
    export_root.mkdir(parents=True, exist_ok=True)
    exported: list[dict] = []
    missing: list[str] = []
    for source in paths:
        source_path = Path(source)
        # 产物可能在源目录或导出目录，统一走记录汇总；取最新一份导出。
        existing = [
            candidate
            for candidate in _product_candidates(source)
            if _product_kind(candidate) in {"cleaned", "repaired"}
        ]
        if not existing:
            missing.append(source_path.name)
            continue
        product = max(existing, key=lambda candidate: candidate.stat().st_mtime)
        destination = export_root / product.name
        shutil.copy2(product, destination)
        exported.append({"source": product.name, "dest": str(destination)})
    return {"exported": exported, "missing": missing}


def _export_health(path: str) -> dict:
    """导出转码健康检查：编码/分辨率/时长/体积等是否适合二次剪辑。"""
    try:
        probe = ffmpeg.probe(path)
        video = next((s for s in probe["streams"] if s.get("codec_type") == "video"), None)
        audio = next((s for s in probe["streams"] if s.get("codec_type") == "audio"), None)
        fmt = probe.get("format", {})
        return {
            "video_codec": video.get("codec_name") if video else None,
            "audio_codec": audio.get("codec_name") if audio else None,
            "pix_fmt": video.get("pix_fmt") if video else None,
            "width": video.get("width") if video else None,
            "height": video.get("height") if video else None,
            "duration": fmt.get("duration"),
            "size_bytes": os.path.getsize(path),
        }
    except Exception:  # noqa: BLE001 - 健康检查失败不阻断主流程
        return {}


def _product_candidates(source: str) -> list[Path]:
    """汇总某源素材的全部产物路径：优先 variants 记录，兼容源目录旧产物。"""
    paths = {
        Path(os.path.expanduser(record["output"]))
        for record in db.list_variants(source=source)
        if record.get("output") and Path(os.path.expanduser(record["output"])).is_file()
    }
    source_path = Path(source)
    for pattern in (
        f"{source_path.stem}_清洗*.mp4",
        f"{source_path.stem}_cleaned*.mp4",
        f"{source_path.stem}_repaired*.mp4",
        f"{source_path.stem}_候选*.mp4",
    ):
        for candidate in source_path.parent.glob(pattern):
            if candidate.is_file():
                paths.add(candidate)
    return list(paths)


def _product_kind(path: Path) -> str:
    name = path.name
    if "_repaired" in name:
        return "repaired"
    if "_清洗" in name:
        return "cleaned"
    if "_候选" in name:
        return "candidate"
    return "cleaned"


def list_outputs(source: str) -> dict:
    """按源素材匹配清洗/修复产物（含导出目录），最新在前。

    候选产物属未来规划能力，当前不在界面展示。
    """
    source = _require_file(source)
    outputs = [
        {
            "kind": _product_kind(candidate),
            "path": str(candidate),
            "name": candidate.name,
            "size": candidate.stat().st_size,
            "mtime": candidate.stat().st_mtime,
        }
        for candidate in _product_candidates(source)
        if _product_kind(candidate) != "candidate"
    ]
    outputs.sort(key=lambda item: item["mtime"], reverse=True)
    return {"source": source, "outputs": outputs}


def library_output_counts() -> dict[str, int]:
    """素材库每个源素材的产物数量（清洗/修复两类合计）。"""
    counts: dict[str, int] = {}
    for item in db.list_library():
        counts[item["path"]] = sum(
            1
            for candidate in _product_candidates(item["path"])
            if _product_kind(candidate) != "candidate"
        )
    return counts


def delete_output(path: str) -> bool:
    """删除一个处理产物文件；只允许产物命名（清洗/修复/候选），防止误删源素材。"""
    target = Path(path)
    if not target.is_file():
        return False
    name = target.name
    if not any(part in name for part in ("_清洗", "_cleaned", "_repaired", "_候选")):
        raise ValueError("仅允许删除清洗/修复/候选产物")
    target.unlink()
    db.delete_variant_by_output(str(target))
    return True


def record_variant(
    source: str,
    output: str,
    options: dict,
    seed: int,
    template_id: str | None = None,
    metrics: dict | None = None,
) -> dict:
    """把产物参数与指标写入 variants 数据表，A/B 追溯用。"""
    return db.create_variant(
        source=source,
        output=output,
        options=options,
        seed=seed,
        template_id=template_id,
        metrics=metrics,
    )


# ---------- 视频处理引擎（FFmpeg）自动安装 ----------

FFMPEG_DOWNLOADS: dict[str, dict[str, str]] = {
    "arm64": {
        "ffmpeg": "https://ffmpeg.martin-riedl.de/redirect/latest/macos/arm64/release/ffmpeg.zip",
        "ffprobe": "https://ffmpeg.martin-riedl.de/redirect/latest/macos/arm64/release/ffprobe.zip",
    },
    "x86_64": {
        "ffmpeg": "https://evermeet.cx/ffmpeg/getrelease/ffmpeg/zip",
        "ffprobe": "https://evermeet.cx/ffmpeg/getrelease/ffprobe/zip",
    },
}

_ffmpeg_install_state: dict[str, str] = {"status": "idle", "detail": ""}


def ffmpeg_install_state() -> dict:
    """返回一键安装的当前状态（idle / downloading / installed / failed）。"""
    return dict(_ffmpeg_install_state)


def install_ffmpeg(target_dir: str) -> dict:
    """在后台线程下载静态 ffmpeg/ffprobe 并安装到目标目录。"""
    if _ffmpeg_install_state.get("status") == "downloading":
        return ffmpeg_install_state()
    _ffmpeg_install_state.update({"status": "downloading", "detail": ""})
    threading.Thread(target=_install_worker, args=(target_dir,), daemon=True).start()
    return ffmpeg_install_state()


def _install_worker(target_dir: str) -> None:
    try:
        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)
        arch = platform.machine()
        sources = FFMPEG_DOWNLOADS.get(arch) or FFMPEG_DOWNLOADS["arm64"]
        for name, url in sources.items():
            archive = target / f"{name}.zip"
            with urllib.request.urlopen(url, timeout=900) as response, open(archive, "wb") as output:
                shutil.copyfileobj(response, output)
            _extract_binary(archive, target, name)
            archive.unlink(missing_ok=True)
        if not (os.access(target / "ffmpeg", os.X_OK) and os.access(target / "ffprobe", os.X_OK)):
            raise RuntimeError("下载内容不完整，请重试")
        ffmpeg.set_custom_dir(str(target))
        db.save_settings({"ffmpeg_dir": str(target)})
        _ffmpeg_install_state.update({"status": "installed", "detail": str(target)})
    except Exception as exc:  # noqa: BLE001 - 下载失败需完整呈现给界面
        if isinstance(exc, (urllib.error.URLError, urllib.error.HTTPError)):
            detail = "下载服务器暂不可用，请稍后重试或选择本机已有组件"
        else:
            detail = str(exc)
        _ffmpeg_install_state.update({"status": "failed", "detail": detail})


def _extract_binary(archive: Path, target: Path, name: str) -> None:
    """从 zip 中取出指定可执行文件并重命名到目标目录。"""
    with zipfile.ZipFile(archive) as zipped:
        candidates = [item for item in zipped.namelist() if item.rstrip("/").endswith(f"/{name}") or item == name]
        if not candidates:
            raise RuntimeError(f"压缩包中未找到 {name}")
        extracted = target / candidates[0]
        zipped.extract(candidates[0], target)
        destination = target / name
        if extracted != destination:
            os.replace(extracted, destination)
        os.chmod(destination, 0o755)
