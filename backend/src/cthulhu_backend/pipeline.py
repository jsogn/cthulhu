"""脱敏管线核心：分析遍、逐块变换、流式编码与音视频 mux。

从 services.py 拆出的纯管线实现，不承担素材库/产物管理/对外入口职责；
run_desensitize 等入口仍留在 services 做编排与指标收口，避免业务服务
模块继续膨胀，也让管线部分可以独立阅读与回归。
"""

from __future__ import annotations

import math
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from itertools import pairwise
from typing import Any

import numpy as np
from scipy.signal import resample_poly

from cthulhu_backend import db
from cthulhu_backend.cache import analysis_cache
from cthulhu_backend.fingerprint import adversarial
from cthulhu_backend.media import ffmpeg
from cthulhu_backend.numeric import channel_stats
from cthulhu_backend.schemas import DesensitizeOptions
from cthulhu_backend.transform import audio as audio_transform
from cthulhu_backend.transform import extra_attacks, profile, purify, regenerate, shots, strategies
from cthulhu_backend.watermark import common as watermark_common

MAX_WORKING_BYTES = 256 * 1024**2  # 低内存机器兜底：256MiB


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
    stop = threading.Event()

    def put_or_exit(item) -> None:
        """投递一项；消费方停止后立即放弃，避免线程永久阻塞持有帧块。"""
        while not stop.is_set():
            try:
                batch_queue.put(item, timeout=0.5)
                return
            except queue.Full:
                continue

    def producer() -> None:
        try:
            pos = 0
            while pos < seg_len and not stop.is_set():
                batch = decoder.read(min(chunk, seg_len - pos), dtype=dtype)
                if len(batch) == 0:
                    break
                put_or_exit((pos, batch))
                pos += len(batch)
        except BaseException as exc:  # noqa: BLE001 - 异常传给主线程统一处理
            put_or_exit(exc)
        finally:
            put_or_exit(None)

    threading.Thread(target=producer, daemon=True, name="cthulhu-prefetch").start()
    try:
        while True:
            item = batch_queue.get()
            if item is None:
                break
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        # 消费方提前退出（取消/异常）时，通知生产者停止并等待其释放
        # 当前帧块；生产者靠超时 put 及时察觉，无需等待其退出。
        stop.set()


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
    pause=None,
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
            if pause is not None:
                while pause.is_set():
                    if stop and stop():
                        raise InterruptedError("任务已取消")
                    time.sleep(0.2)
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


@dataclass
class _DesensitizeState:
    """分析遍产物：编码遍与指标遍共享的派生状态。"""

    info: dict
    fps_in: float
    total_in: int
    total_out: int
    speed: float
    audio_tempo: float | None
    needs_audio_alignment: bool
    segments: list[tuple[int, int, int]]
    seg_factors: list[float]
    seg_out_lens: list[int]
    strategy: Any
    transform_options: Any
    shot_options: list[Any] | None
    seg_shot_indices: list[int] | None
    transform_context: Any
    frame_dtype: str
    native_chain: list[str]
    native_complex: str | None
    base_filters: list[str]
    use_native_geometry: bool
    use_yuv_path: bool
    chunk: int
    crf: int
    bitrate_kbps: int | None
    output_fps: float
    out_size: tuple[int, int] | None
    requant_eff: int
    phash_attack: bool
    phash_epsilon: float
    phash_iters: int
    multi_hash_attack: bool
    dhash_attack: bool
    quality_protect: bool
    psnr_target: float
    ssim_target: float
    assault_rng: Any
    assault_params: Any
    saliency_obj: Any
    recrop: float
    filter_scale: int
    regrade: bool
    rotate: float
    rotate_eff: float
    gamma_strength: float
    brightness: float
    period_a: float
    period_b: float
    period_c: float
    period_d: float
    phase_a: float
    phase_b: float
    phase_c: float
    phase_d: float
    purify_note: str
    purify_enabled: bool
    profile_note: str = "off"
    profile_metrics: dict | None = None


def _geometry_chain(state: _DesensitizeState, offset: int) -> list[str]:
    """几何/调光原生滤镜链；offset 用于分段并行时保持全局帧号连续。"""
    chain: list[str] = []
    if not state.use_native_geometry:
        return chain
    frame_var = f"(n+{offset})"
    if state.recrop > 0 and ffmpeg.has_filter("crop") and ffmpeg.has_filter("scale"):
        frame_w, frame_h = state.info["width"], state.info["height"]
        crop_x = int(frame_w * state.recrop)
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
    if state.filter_scale > 0:
        frame_w, frame_h = state.info["width"], state.info["height"]
        scale_h = max(2, round(frame_h * state.filter_scale / frame_w))
        chain.append(f"scale={state.filter_scale}:{scale_h}:flags=bicubic")
    if state.regrade and ffmpeg.has_filter("eq"):
        gamma_expr = (
            f"1+{state.gamma_strength:.5f}*(0.6*sin(2*PI*{frame_var}/{state.period_a:.1f}+{state.phase_a:.4f})"
            f"+0.4*sin(2*PI*{frame_var}/{state.period_b:.1f}+{state.phase_b:.4f}))"
        )
        brightness_expr = (
            f"{state.brightness:.5f}*(0.6*sin(2*PI*{frame_var}/{state.period_c:.1f}+{state.phase_c:.4f})"
            f"+0.4*sin(2*PI*{frame_var}/{state.period_d:.1f}+{state.phase_d:.4f}))"
        )
        chain.append(f"eq=gamma='{gamma_expr}':brightness='{brightness_expr}'")
    if state.rotate > 0 and ffmpeg.has_filter("rotate"):
        # numpy 侧为 max_angle(度) * sin(2π n / 200)；ffmpeg rotate 用弧度。
        angle_rad = state.rotate * np.pi / 180.0
        chain.append(
            f"rotate=a='{angle_rad:.6f}*sin(2*PI*{frame_var}/200)':c=black:bilinear=1"
        )
    return chain


def _apply_attacks(
    state: _DesensitizeState,
    frames: np.ndarray,
    output_ids: list[int],
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """pHash 攻击、几何去同步与画质门控等与分块无关的后置变换。"""
    assault_rng = rng if rng is not None else state.assault_rng
    original = frames.copy() if state.quality_protect else None
    if state.rotate_eff > 0:
        frames = strategies.rotate_de_sync(frames, output_ids, state.rotate_eff)
    if state.assault_params.enabled:
        frames = extra_attacks.apply(frames, state.assault_params, assault_rng)
    if state.phash_attack:
        frames = adversarial.attack_frames(
            frames, epsilon=state.phash_epsilon, iterations=state.phash_iters
        )
    elif state.dhash_attack:
        frames = adversarial.attack_frames_dhash(
            frames, epsilon=state.phash_epsilon, iterations=state.phash_iters
        )
    elif state.multi_hash_attack:
        frames = adversarial.attack_frames_joint(
            frames, epsilon=state.phash_epsilon, iterations=state.phash_iters
        )
    # 门控参考帧在策略层之后取得，因此只约束后续攻击层（旋转/底层原语/哈希），
    # 不会把净化结果拉回原图；净化自身的画质代价由指标遍如实报告。
    if original is not None:
        frames = extra_attacks.quality_gate(
            original, frames, state.psnr_target, state.ssim_target
        )
    if state.saliency_obj is not None:
        frames = extra_attacks.salient_overlay(frames, state.saliency_obj)
    return frames


def _parallel_workers() -> int:
    """武器阶段进程池规模：默认关闭（8 线程已饱和带宽型武器）。

    仅当显式设置 CTHULHU_PROCESS_WORKERS 时才启用多进程，供更高核数的
    机器或未来计算型武器使用；进程内线程数保持 8 以不拖慢逐帧武器。
    """
    raw = os.environ.get("CTHULHU_PROCESS_WORKERS")
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return 1


def _purify_progress(
    base: int,
    count: int,
    fraction: float,
    total_out: int,
) -> tuple[int, str]:
    """把批次内进度换算成整段视频的全局帧数与百分比。"""
    total = max(1, int(total_out))
    bounded = max(0.0, min(1.0, float(fraction)))
    processed = round(base + bounded * max(1, count))
    processed = max(0, min(total, processed))
    percent = 8 + int(processed / total * 82)
    return min(90, percent), f"潜空间净化 {processed}/{total}"


def _transform_chunk_worker(payload: dict) -> np.ndarray:
    """进程池工作项：解码块 → 变换 → 攻击，返回按序待编码的帧。"""
    import dataclasses as dc

    state = payload["state"]
    if _WORKER_FIELDS:
        for attr, value in _WORKER_FIELDS.items():
            if value is not None:
                setattr(state.assault_params, attr, value)
    frames = payload["batch"]
    output_ids = payload["output_ids"]
    rng = np.random.default_rng(payload["chunk_seed"])
    ctx = state.transform_context
    if ctx.mid_rng is not None:
        ctx = dc.replace(ctx, mid_rng=np.random.default_rng(payload["chunk_seed"] ^ 0x51A))
    options = payload.get("options", state.transform_options)
    frames = state.strategy.apply(frames, output_ids, ctx, options)
    return _apply_attacks(state, frames, output_ids, rng=rng)


_WORKER_FIELDS: dict[str, np.ndarray] = {}


def _worker_init(fields: dict[str, np.ndarray]) -> None:
    """进程池初始化：一次性注入首块算好的 SPSA 场，避免逐任务序列化。"""
    global _WORKER_FIELDS
    _WORKER_FIELDS = fields
    # 进程级并行接管后，进程内的帧级线程池应退化为单线程，避免 48 线程
    # 抢占 14 核造成的调度抖动。
    os.environ["CTHULHU_TRANSFORM_THREADS"] = "8"


def _collect_spsa_fields(state: _DesensitizeState, rng: np.random.Generator) -> dict:
    """取出首块算好的 copy/face 一次性 SPSA 场。"""

    cache = regenerate._FIELD_CACHE
    key = id(rng)
    fields: dict[str, np.ndarray] = {}
    for name, attr in (
        ("copy", "copy_field"),
        ("facefield", "face_field"),
    ):
        field = cache.get((name, key))
        if field is not None:
            fields[attr] = field
    return fields


def _plan_shots(
    sampled: np.ndarray,
    sampled_starts: list[int],
    total_in: int,
    need_shots: bool,
) -> list[tuple[int, int]]:
    """抽样窗口内检测切点并映射回全局帧号，返回相邻切点区间。"""
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
    return list(pairwise(boundaries))


def _build_segments(
    shot_ranges: list[tuple[int, int]],
    opts: DesensitizeOptions,
    speed: float,
    rng: np.random.Generator,
) -> tuple[list[tuple[int, int, int]], list[float], list[int], list[int], int]:
    """逐镜头切点漂移/变速与重排，输出编码区间（消耗 rng 流，顺序敏感）。"""
    shot_offsets: list[tuple[int, int]] = []
    shot_factors: list[float] = []
    for start, end in shot_ranges:
        length = end - start
        drop_start = int(rng.integers(0, opts.cut_jitter + 1)) if opts.cut_jitter > 0 else 0
        drop_end = int(rng.integers(0, opts.cut_jitter + 1)) if opts.cut_jitter > 0 else 0
        if drop_start + drop_end >= length:
            drop_start = min(drop_start, max(0, length - 1))
            drop_end = 0
        shot_offsets.append((drop_start, drop_end))
        shot_factors.append(
            float(rng.uniform(opts.shot_retime_min, opts.shot_retime_max))
            if opts.shot_retime
            else 1.0
        )
    order = rng.permutation(len(shot_ranges)) if opts.reorder else np.arange(len(shot_ranges))
    segments: list[tuple[int, int, int]] = []
    seg_factors: list[float] = []
    seg_out_lens: list[int] = []
    seg_shot_indices: list[int] = []
    cursor = 0
    # 累计取整（Bresenham）：逐镜头独立 round 会把每段的舍入误差留成常驻偏差，
    # 等长镜头下最多每段 ±0.5 帧，50 段就能累积成半秒音画不同步。改为「先累计
    # 精确输出长度、再取整到帧」，任意切点处的偏差都被夹在一帧以内。
    exact_out = 0.0
    for shot_index in order:
        orig_start, orig_end = shot_ranges[int(shot_index)]
        drop_start, drop_end = shot_offsets[int(shot_index)]
        eff_start = orig_start + drop_start
        eff_len = (orig_end - orig_start) - drop_start - drop_end
        factor = speed * shot_factors[int(shot_index)]
        retimed = speed != 1.0 or opts.shot_retime
        if retimed:
            exact_out += eff_len / factor
            out_len = max(1, round(exact_out) - cursor)
        else:
            out_len = eff_len
            exact_out = cursor + eff_len
        segments.append((cursor, eff_start, eff_len))
        seg_factors.append(factor)
        seg_out_lens.append(out_len)
        seg_shot_indices.append(int(shot_index))
        cursor += out_len
    return segments, seg_factors, seg_out_lens, seg_shot_indices, cursor


def _regrade_curves(
    rng: np.random.Generator,
    total_out: int,
    perturb: float,
    regrade: bool,
) -> tuple[np.ndarray, np.ndarray, float, float, tuple[float, float, float, float], tuple[float, float, float, float]]:
    """调光曲线：低频平滑的 gamma/亮度轨迹（确定性，消耗 rng 流）。"""
    gammas = np.ones(total_out, dtype=np.float32)
    deltas = np.zeros(total_out, dtype=np.float32)
    gamma_strength = 0.03 + 0.2 * perturb
    brightness = 0.02 + 0.06 * perturb
    periods = (0.0, 0.0, 0.0, 0.0)
    phases = (0.0, 0.0, 0.0, 0.0)
    if regrade:
        # 时间平滑：调光参数沿低频轨迹变化，避免逐帧独立随机造成的暗部闪烁。
        # 幅度与旧实现一致（gamma ±gamma_strength、亮度 ±brightness），对抗
        # 语义不变，只是相邻帧连续过渡。
        t = np.arange(total_out, dtype=np.float32)
        periods = tuple(float(rng.uniform(80.0, 180.0)) for _ in range(4))
        phases = tuple(float(rng.uniform(0.0, 2.0 * np.pi)) for _ in range(4))
        period_a, period_b, period_c, period_d = periods
        phase_a, phase_b, phase_c, phase_d = phases
        gammas = 1.0 + gamma_strength * (
            0.6 * np.sin(2.0 * np.pi * t / period_a + phase_a)
            + 0.4 * np.sin(2.0 * np.pi * t / period_b + phase_b)
        ).astype(np.float32)
        deltas = brightness * (
            0.6 * np.sin(2.0 * np.pi * t / period_c + phase_c)
            + 0.4 * np.sin(2.0 * np.pi * t / period_d + phase_d)
        ).astype(np.float32)
    return gammas, deltas, gamma_strength, brightness, periods, phases


def _prepare_desensitize(
    path: str,
    opts: DesensitizeOptions,
    progress_cb,
    check_cancelled,
) -> _DesensitizeState:
    """分析遍：镜头边界、逐镜头参数、变换上下文与编码参数一次性组装。"""
    info = ffmpeg.video_info(path)
    fps_in = float(info["fps"])
    total_in = max(1, round(info["duration"] * fps_in))
    check_cancelled()

    speed = opts.speed
    # 回声水印清除：音画按同一 factor 同步放慢，音频用 atempo 保调拉伸，
    # 移动回声时延以破坏检测，同时保持音画内容对齐。
    # 源无音轨时回声水印不存在：此时变速只会白白拉长成片，直接跳过。
    if opts.echo_defeat and ffmpeg.has_audio(path):
        speed = speed * 0.97
        if speed < 0.5:
            raise ValueError("echo_defeat 需要有效变速 factor ≥ 0.5")
        audio_tempo = speed
    else:
        audio_tempo = None

    # 音频处理开关全关时音轨原样透传；但任何会改变音画时长关系的操作
    # （变速/抽帧/帧重排/镜头变速/切点抖动/输出帧率）都必须强制对齐音轨，
    # 否则会与画面时长错位。
    needs_audio_alignment = (
        speed != 1.0
        or opts.reorder
        or opts.shot_retime
        or opts.cut_jitter > 0
        or opts.drop_every > 0
        or (opts.fps_out is not None and opts.fps_out != fps_in)
    )

    # ---------- 分析遍：抽样镜头边界 + 预生成逐帧随机参数 ----------
    if progress_cb:
        progress_cb(8, "扫描镜头结构")
    # 自动画像也按镜头工作：镜头边界仍用灰度 400 帧抽样检测（缓存优先），
    # 颜色画像另取 60 帧，避免把 400 帧彩色序列常驻内存。
    need_shots = (
        opts.reorder or opts.shot_retime or opts.cut_jitter > 0 or opts.auto_profile
    )
    sampled_gray = np.empty((0,), dtype=np.float32)
    sampled_starts = [0]
    if need_shots:
        shot_ranges = analysis_cache.get_shot_boundaries(path)
        if not (
            shot_ranges
            and shot_ranges[0][0] == 0
            and shot_ranges[-1][1] == total_in
        ):
            sampled_gray, _, sampled_starts = ffmpeg.decode_sampled(
                path, cap=400, return_starts=True
            )
            shot_ranges = _plan_shots(sampled_gray, sampled_starts, total_in, True)
            analysis_cache.put_shot_boundaries(path, shot_ranges)
    else:
        shot_ranges = _plan_shots(sampled_gray, sampled_starts, total_in, False)

    rng = np.random.default_rng(opts.seed)
    segments, seg_factors, seg_out_lens, seg_shot_indices, total_out = _build_segments(
        shot_ranges, opts, speed, rng
    )

    gammas, deltas, gamma_strength, brightness, periods, phases = _regrade_curves(
        rng, total_out, opts.perturb, opts.regrade
    )
    period_a, period_b, period_c, period_d = periods
    phase_a, phase_b, phase_c, phase_d = phases

    mid_rng = np.random.default_rng(opts.seed)
    spoof_bits = None
    if opts.spoof:
        spoof_rng = np.random.default_rng(opts.seed ^ 0x5F3759DF)
        spoof_bits = watermark_common.payload_bits(int(spoof_rng.integers(0, 2**31)), 64)
    assault_rng = np.random.default_rng(opts.seed ^ 0xA55A55A5)
    # FFmpeg 原生滤镜迁移：可平移的原语始终交给编码器滤镜链，numpy 侧跳过，
    # 不依赖对抗档位开关，默认路径即可获得成倍提速。
    sharpness_eff = opts.sharpness
    noise_eff = opts.noise
    requant_eff = opts.requant
    chroma_eff = opts.chroma_levels
    denoise_eff = opts.denoise
    # 几何/调光下沉到编码器滤镜链：仅在无重排、无变速的快路径上启用，
    # 避免输出帧号与原帧号不一致时表达式错位；其余路径保持 numpy 实现。
    use_native_geometry = (
        not opts.reorder and speed == 1.0 and not opts.shot_retime and opts.cut_jitter == 0
    )
    recrop_eff = opts.recrop
    regrade_eff = opts.regrade
    rotate_eff = opts.rotate
    if use_native_geometry:
        if opts.recrop > 0 and ffmpeg.has_filter("crop") and ffmpeg.has_filter("scale"):
            recrop_eff = 0.0
        if opts.regrade and ffmpeg.has_filter("eq"):
            regrade_eff = False
        if opts.rotate > 0 and ffmpeg.has_filter("rotate"):
            rotate_eff = 0.0

    base_filters: list[str] = []
    if opts.sharpness and ffmpeg.has_filter("unsharp"):
        base_filters.append("unsharp=5:5:0.25:3:3:0.0")
        sharpness_eff = False
    if opts.denoise and ffmpeg.has_filter("removegrain"):
        base_filters.append("removegrain=4")
        denoise_eff = False
    # 噪声/重量化下沉原生滤镜：二者近乎无损且远快于 numpy 路径；画质门控
    # 只负责覆盖 numpy 侧的重武器，这两项不参与门控（文档口径见门控说明）。
    if opts.noise > 0 and ffmpeg.has_filter("noise"):
        base_filters.append(f"noise=alls={max(1, round(opts.noise * 255))}:allf=t")
        noise_eff = 0.0
    # posterize 只支持 2~31 级，重量化档位（32~96）仍走 numpy 路径。
    if 0 < opts.requant <= 31 and ffmpeg.has_filter("posterize"):
        base_filters.append(f"posterize={opts.requant}")
        requant_eff = 0
    elif opts.requant > 31 and ffmpeg.has_filter("lut"):
        # 任意级数量化查表：与 numpy 查表实现效果等价（原型验证 QIM 破坏力一致）。
        scale = (opts.requant - 1) / 255.0
        expr = f"round(val*{scale:.6f})*{1.0 / scale:.6f}"
        base_filters.append(f"lut=r='{expr}':g='{expr}':b='{expr}'")
        requant_eff = 0
    if opts.chroma_levels > 0 and ffmpeg.has_filter("lutyuv"):
        # 色度量化原生下沉：YUV 域对 U/V 平面查表量化，Y 原样保留。
        levels = opts.chroma_levels
        scale = (levels - 1) / 255.0
        expr = f"128+round((val-128)*{scale:.6f})*{1.0 / scale:.6f}"
        base_filters.append(
            f"format=yuv444p,lutyuv=y='val':u='{expr}':v='{expr}',format=rgb24"
        )
        chroma_eff = 0
    # 跨帧估计的原生加速：ffmpeg 链（median+tmix+blend 两步重构）在原生几何
    # 快路径上启用；numpy 版作为重排/变速路径的兜底。
    temporal_eff = opts.temporal_sub
    native_temporal_ok = (
        opts.native_temporal
        and temporal_eff > 0
        and use_native_geometry
        and ffmpeg.has_filter("median")
        and ffmpeg.has_filter("tmix")
        and ffmpeg.has_filter("blend")
    )
    if native_temporal_ok:
        temporal_eff = 0.0
    if opts.filter_scale > 0 and use_native_geometry:
        base_filters.append(f"scale={info['width']}:{info['height']}:flags=bicubic")
    assault_params = extra_attacks.AssaultParams(
        jitter=opts.jitter,
        perspective=opts.perspective,
        warp=opts.warp,
        median=opts.median,
        noise=noise_eff,
        subtract_beta=opts.subtract_beta,
        requant=requant_eff,
        dct_step=opts.dct_step,
        chroma_levels=chroma_eff,
        drop_every=opts.drop_every,
        temporal_sub=temporal_eff,
        fft_phase=opts.fft_phase,
        fft_mag=opts.fft_mag,
        dwt_detail=opts.dwt_detail,
        hsv_jitter=opts.hsv_jitter,
        nonint_ratio=opts.nonint_ratio,
        flow_disturb=opts.flow_disturb,
        texture_inject=opts.texture_inject,
        multiscale=opts.multiscale,
        complexity_trap=opts.complexity_trap,
        face_perturb=opts.face_perturb,
        temporal_blur=opts.temporal_blur,
        copy_attack=opts.copy_attack,
    )
    saliency_obj = extra_attacks.saliency_layout(opts.seed, opts.saliency)

    sampled_color: np.ndarray | None = None
    color_starts: list[int] = [0]
    if opts.auto_profile:
        sampled_color, _, color_starts = ffmpeg.decode_sampled(
            path, cap=60, grayscale=False, return_starts=True
        )

    if opts.color_restore:
        cached_stats = analysis_cache.get_color_stats(path)
        if cached_stats is not None:
            ref_mean, ref_std = cached_stats
        else:
            color_sampled = sampled_color
            if color_sampled is None or len(color_sampled) == 0:
                color_sampled, _ = ffmpeg.decode_sampled(path, cap=40, grayscale=False)
            ref_mean, ref_std = channel_stats(color_sampled)
            if color_sampled is not sampled_color:
                del color_sampled
            analysis_cache.put_color_stats(path, ref_mean, ref_std)
    else:
        ref_mean = np.zeros(3, dtype=np.float32)
        ref_std = np.zeros(3, dtype=np.float32)
    del sampled_gray

    # 潜空间净化：可选依赖缺位时降级回经典档，结果里记录降级原因。
    purify_strength_eff = opts.purify_strength
    purify_detail_eff = opts.purify_detail
    purify_temporal_eff = opts.purify_temporal
    purify_note = "off"
    if opts.purify_strength > 0:
        available, reason = purify.preflight()
        if available:
            purify_note = "will_download" if reason == "will_download" else "applied"
        else:
            purify_strength_eff = 0.0
            purify_detail_eff = 0.0
            purify_temporal_eff = 0.0
            purify_note = f"skipped: {reason}"

    # 内容画像：黑盒按镜头做复杂度/运动/时序一致性自适应；嵌入域按已知方案映射。
    profile_note = "off"
    profile_metrics = None
    shot_profiles: list[profile.Profile] = []
    embedding_attack_eff = opts.embedding_attack
    if opts.auto_profile and sampled_color is not None and len(sampled_color) > 0:
        prof = profile.profile_frames(sampled_color)
        shot_profiles = profile.profile_shots(
            sampled_color,
            color_starts,
            shot_ranges,
            fallback=prof,
        )
        profile_metrics = {
            **prof.as_dict(),
            "shot_count": len(shot_profiles),
            "shots": [
                {"start": start, "end": end, **shot_prof.as_dict()}
                for (start, end), shot_prof in zip(
                    shot_ranges, shot_profiles, strict=False
                )
            ],
        }
        profile_note = "applied"
        embedding_attack_eff = prof.suggested_attack
        if purify_strength_eff > 0:
            # 自动画像只在用户预算内调强度与时序；细节回注/带宽是画质档位参数，
            # 不参与自适应（砍它会直接糊掉字幕，见 research §19.2）。
            purify_strength_eff = min(
                opts.purify_strength, prof.suggested_purify_strength
            )
            purify_temporal_eff = min(
                opts.purify_temporal, prof.suggested_purify_temporal
            )
    elif embedding_attack_eff == "auto":
        embedding_attack_eff = "both"
    if purify_strength_eff <= 0:
        purify_detail_eff = 0.0
        purify_temporal_eff = 0.0
    del sampled_color

    # 变换策略：与分块无关的选项与上下文一次性组装，分块内只调用 apply。
    strategy = strategies.get_strategy(opts.transform_strategy)
    transform_options = strategies.TransformOptions(
        recrop=recrop_eff,
        regrade=regrade_eff,
        anti_reembed=opts.anti_reembed,
        color_restore=opts.color_restore,
        sharpness=sharpness_eff,
        denoise=denoise_eff,
        spoof=opts.spoof,
        purify_strength=purify_strength_eff,
        purify_detail=purify_detail_eff,
        purify_detail_sigma=opts.purify_detail_sigma,
        purify_detail_wide=opts.purify_detail_wide,
        purify_temporal=purify_temporal_eff,
        purify_max_edge=opts.purify_max_edge,
        purify_batch=opts.purify_batch,
        embedding_attack=embedding_attack_eff,
        embedding_strength=opts.embedding_strength,
        embedding_variant=opts.embedding_variant,
        embedding_aggressive=opts.embedding_aggressive,
    )
    shot_options: list[strategies.TransformOptions] | None = None
    if opts.auto_profile and shot_profiles:
        shot_options = []
        for shot_prof in shot_profiles:
            shot_strength = (
                min(opts.purify_strength, shot_prof.suggested_purify_strength)
                if purify_strength_eff > 0
                else 0.0
            )
            shot_options.append(
                replace(
                    transform_options,
                    purify_strength=shot_strength,
                    purify_temporal=min(
                        opts.purify_temporal, shot_prof.suggested_purify_temporal
                    ),
                    embedding_attack=shot_prof.suggested_attack,
                )
            )
    transform_context = strategies.TransformContext(
        gammas=gammas,
        deltas=deltas,
        banner=opts.banner,
        seed=opts.seed,
        ref_mean=ref_mean,
        ref_std=ref_std,
        mid_rng=mid_rng,
        spoof_bits=spoof_bits,
    )

    crf = 0 if opts.lossless else 23
    bitrate_kbps = opts.bitrate_kbps
    if opts.lossless:
        # 无损档以 CRF=0 为准，显式码率与无损互斥，避免被编码器忽略成非无损。
        bitrate_kbps = None
    output_fps = opts.fps_out or fps_in
    out_size = None
    if opts.resolution and "x" in opts.resolution:
        try:
            width, height = (int(part) for part in opts.resolution.split("x", 1))
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
    if opts.phash_attack or opts.multi_hash_attack or opts.dhash_attack:
        chunk = min(chunk, 48)

    # YUV420p 快路径：仅当所有像素变换都已下沉原生、且无任何 RGB 专属
    # 选项时启用，原始帧体积减半、省去 RGB↔YUV 转换。
    use_yuv_path = (
        use_native_geometry
        and getattr(strategy, "name", "") == "fast"
        and not opts.color_restore
        and not opts.banner
        and not opts.anti_reembed
        and not opts.spoof
        and not opts.quality_protect
        and opts.saliency == 0
        and opts.jitter <= 0
        and opts.perspective <= 0
        and opts.warp <= 0
        and opts.median <= 0
        and opts.subtract_beta <= 0
        and opts.temporal_sub <= 0
        and opts.fft_phase <= 0
        and opts.fft_mag <= 0
        and opts.dwt_detail <= 0
        and opts.nonint_ratio <= 0
        and opts.flow_disturb <= 0
        and opts.texture_inject <= 0
        and opts.multiscale <= 0
        and opts.complexity_trap <= 0
        and opts.face_perturb <= 0
        and opts.temporal_blur <= 0
        and opts.copy_attack <= 0
        and opts.hsv_jitter <= 0
        and opts.dct_step <= 0
        and opts.chroma_levels <= 0
        and opts.drop_every <= 0
        and not opts.multi_hash_attack
        and not opts.dhash_attack
        and opts.purify_strength <= 0
        and opts.embedding_strength <= 0
    )

    state = _DesensitizeState(
        info=info,
        fps_in=fps_in,
        total_in=total_in,
        total_out=total_out,
        speed=speed,
        audio_tempo=audio_tempo,
        needs_audio_alignment=needs_audio_alignment,
        segments=segments,
        seg_factors=seg_factors,
        seg_out_lens=seg_out_lens,
        strategy=strategy,
        transform_options=transform_options,
        shot_options=shot_options,
        seg_shot_indices=seg_shot_indices,
        transform_context=transform_context,
        frame_dtype=frame_dtype,
        native_chain=[],
        native_complex=None,
        base_filters=base_filters,
        use_native_geometry=use_native_geometry,
        use_yuv_path=use_yuv_path,
        chunk=chunk,
        crf=crf,
        bitrate_kbps=bitrate_kbps,
        output_fps=output_fps,
        out_size=out_size,
        requant_eff=requant_eff,
        phash_attack=opts.phash_attack,
        phash_epsilon=opts.phash_epsilon,
        phash_iters=opts.phash_iters,
        multi_hash_attack=opts.multi_hash_attack,
        dhash_attack=opts.dhash_attack,
        quality_protect=opts.quality_protect,
        psnr_target=opts.psnr_target,
        ssim_target=opts.ssim_target,
        assault_rng=assault_rng,
        assault_params=assault_params,
        saliency_obj=saliency_obj,
        recrop=opts.recrop,
        filter_scale=opts.filter_scale,
        regrade=opts.regrade,
        rotate=opts.rotate,
        rotate_eff=rotate_eff,
        gamma_strength=gamma_strength,
        brightness=brightness,
        period_a=period_a,
        period_b=period_b,
        period_c=period_c,
        period_d=period_d,
        phase_a=phase_a,
        phase_b=phase_b,
        phase_c=phase_c,
        phase_d=phase_d,
        purify_note=purify_note,
        purify_enabled=purify_strength_eff > 0,
        profile_note=profile_note,
        profile_metrics=profile_metrics,
    )
    if native_temporal_ok:
        beta = opts.temporal_sub
        prefix = ",".join(_geometry_chain(state, 0) + base_filters)
        graph = "[0]" + prefix
        first = min(beta, 1.0)
        extra = max(0.0, beta - 1.0)
        if extra <= 1e-6:
            graph += (
                ("," if prefix else "")
                + f"split=3[a][b][c];[b]median=radius=1[m];"
                f"[m]tmix=frames=24[tm];[a]tmix=frames=24[ta];"
                f"[c][tm]blend=all_mode=addition:all_opacity={first}[s1];"
                f"[s1][ta]blend=all_mode=subtract:all_opacity={first}[out]"
            )
        else:
            # β>1：blend 的 opacity 上限为 1，把过减拆成两段「加/减重构」，
            # 两段合计仍为 β（每段都无带符号中间量，规避 8bit 负值钳零）。
            graph += (
                ("," if prefix else "")
                + f"split=3[a][b][c];[b]median=radius=1[m];"
                f"[m]split=2[m0][m1];[m0]tmix=frames=24[tm0];[m1]tmix=frames=24[tm1];"
                f"[a]split=2[a0][a1];[a0]tmix=frames=24[ta0];[a1]tmix=frames=24[ta1];"
                f"[c][tm0]blend=all_mode=addition:all_opacity={first}[s1];"
                f"[s1][ta0]blend=all_mode=subtract:all_opacity={first}[s2];"
                f"[s2][tm1]blend=all_mode=addition:all_opacity={extra}[s3];"
                f"[s3][ta1]blend=all_mode=subtract:all_opacity={extra}[out]"
            )
        state.native_complex = graph
        state.native_chain = []
    else:
        state.native_chain = _geometry_chain(state, 0) + state.base_filters
    return state


def _prepare_audio(
    path: str,
    opts: DesensitizeOptions,
    state: _DesensitizeState,
) -> tuple[np.ndarray | None, int]:
    """音轨独立准备：整段重混/回声扰动，供编码完成后统一 mux。"""
    sample_rate = 16000
    if not (
        opts.audio_remix
        or opts.echo_defeat
        or opts.lpc_attack > 0
        or state.needs_audio_alignment
    ):
        return None, sample_rate
    decoded_audio = ffmpeg.decode_audio(path)
    if decoded_audio is None:
        return None, sample_rate
    signal, sample_rate = decoded_audio
    audio_rng = np.random.default_rng(opts.seed ^ 0x9E3779B9)
    if opts.audio_remix and opts.audio_strong:
        signal = audio_transform.remix_strong(
            signal, sample_rate, audio_rng,
            pitch_ratio=0.985, eq_db=4.0, noise_floor=0.003,
        )
    elif opts.audio_remix:
        # 等长重混：不改内容时间线，音画同步只由末尾按实际帧数对齐兜底。
        signal = audio_transform.remix(signal, sample_rate, audio_rng)
    if opts.lpc_attack > 0:
        signal = audio_transform.lpc_whiten(
            signal, sample_rate, opts.lpc_attack, audio_rng
        )
    return signal, sample_rate


def _mux_output(
    path: str,
    output: str,
    temp_video: str,
    out_index: int,
    audio_signal: np.ndarray | None,
    sample_rate: int,
    opts: DesensitizeOptions,
    state: _DesensitizeState,
) -> None:
    """把视频轨与音轨合成为最终产物：重混走 s16le，未处理则原样透传。"""
    if audio_signal is not None:
        # 以实际写出的帧数对齐音轨时长：等长重混下仅当视频做统一变速
        # 时才需要拉伸，且拉伸比例与视频统一变速因子一致，保持内容对齐。
        # audio_tempo 是「音画同步变速是否真的生效」的唯一判据：源无音轨时
        # 即使勾了回声清除也不会变速，此处必须随之走等长重采样。
        tempo = state.audio_tempo
        if tempo is None:
            target_len = round(out_index / state.output_fps * sample_rate)
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
        if tempo is not None:
            # atempo 保调拉伸放在 mux 滤镜里完成，与视频 factor 一致，
            # Python 侧不重采样，避免二次变速造成漂移。
            mux_cmd += ["-af", f"atempo={tempo:.5f}"]
        mux_cmd += ["-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "-shortest", output]
        subprocess.run(
            mux_cmd,
            input=audio_payload,
            capture_output=True,
            check=True,
        )
        return
    # 未做音频处理时音轨原样透传；源无音轨则直接落盘视频。
    if ffmpeg.has_audio(path):
        copy_cmd = [
            ffmpeg.FFMPEG_BIN, "-y", "-v", "error",
            "-i", temp_video, "-i", path,
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy", "-c:a", "copy", "-shortest", output,
        ]
        try:
            subprocess.run(copy_cmd, capture_output=True, check=True)
        except subprocess.CalledProcessError:
            # 源音轨编码不适合 MP4 直接拷贝时，重编码为 AAC 保底。
            fallback_cmd = [
                ffmpeg.FFMPEG_BIN, "-y", "-v", "error",
                "-i", temp_video, "-i", path,
                "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
                "-shortest", output,
            ]
            subprocess.run(fallback_cmd, capture_output=True, check=True)
    else:
        shutil.move(temp_video, output)


def _encode_yuv_fast(
    path: str,
    temp_video: str,
    opts: DesensitizeOptions,
    state: _DesensitizeState,
    progress_cb,
    should_stop,
    pause,
) -> int:
    """YUV420p 快路径：静态变换全下沉编码器滤镜，长片分段并发，失败回退串行。"""
    frame_h, frame_w = state.info["height"], state.info["width"]
    cpu = os.cpu_count() or 1
    max_k = 3 if cpu >= 12 else (2 if cpu >= 8 else 1)
    k = max(1, min(max_k, state.total_out // 800))
    if state.total_out < 2400:
        k = 1
    override_k = os.environ.get("CTHULHU_SEGMENT_K")
    if override_k:
        try:
            k = max(1, min(int(override_k), max_k, state.total_out))
        except ValueError:
            pass
    if k <= 1:
        return _process_segment_yuv(
            path, 0, state.total_out, temp_video,
            frame_w, frame_h, state.output_fps,
            chunk=state.chunk, requant_eff=state.requant_eff,
            phash_attack=state.phash_attack,
            phash_epsilon=state.phash_epsilon, phash_iters=state.phash_iters,
            attack_workers=None, filters=state.native_chain, codec=opts.codec,
            hardware=opts.hardware, crf=state.crf, preset=opts.preset, gop=opts.gop,
            threads=None,
            stop=should_stop,
            pause=pause,
        )

    seg_paths = [f"{temp_video}.seg{index}.mp4" for index in range(k)]
    seg_len = math.ceil(state.total_out / k)
    seg_threads = max(2, cpu // k)
    done = [0]
    progress_lock = threading.Lock()

    def seg_progress(delta: int) -> None:
        with progress_lock:
            done[0] += delta
            current = done[0]
        if progress_cb and state.total_out:
            progress_cb(
                8 + int(current / state.total_out * 82),
                f"处理中 {current}/{state.total_out} 帧",
            )

    out_index = 0
    try:
        with ThreadPoolExecutor(max_workers=k) as pool:
            futures = []
            for index in range(k):
                start = index * seg_len
                count = min(seg_len, state.total_out - start)
                filters = _geometry_chain(state, start) + state.base_filters
                futures.append(
                    pool.submit(
                        _process_segment_yuv,
                        path, start, count, seg_paths[index],
                        frame_w, frame_h, state.output_fps,
                        chunk=state.chunk, requant_eff=state.requant_eff,
                        phash_attack=state.phash_attack,
                        phash_epsilon=state.phash_epsilon,
                        phash_iters=state.phash_iters, attack_workers=2,
                        filters=filters,
                        codec=opts.codec, hardware=opts.hardware,
                        crf=state.crf,
                        preset=opts.preset, gop=opts.gop, threads=seg_threads,
                        progress=seg_progress if progress_cb else None,
                        stop=should_stop,
                        pause=pause,
                    )
                )
            for future in as_completed(futures):
                out_index += future.result()
        _concat_video_segments(seg_paths, temp_video)
        return out_index
    except InterruptedError:
        raise
    except BaseException:  # noqa: BLE001 - 段级失败统一回退串行
        # 任一段失败回退串行，保证任务可用。
        return _process_segment_yuv(
            path, 0, state.total_out, temp_video,
            frame_w, frame_h, state.output_fps,
            chunk=state.chunk, requant_eff=state.requant_eff,
            phash_attack=state.phash_attack,
            phash_epsilon=state.phash_epsilon, phash_iters=state.phash_iters,
            attack_workers=None, filters=state.native_chain, codec=opts.codec,
            hardware=opts.hardware, crf=state.crf, preset=opts.preset,
            gop=opts.gop, threads=None,
            stop=should_stop,
            pause=pause,
        )
    finally:
        # 成功、失败、取消都要清理段文件，避免残留 .segN.mp4。
        for seg_path in seg_paths:
            try:
                os.unlink(seg_path)
            except OSError:
                pass


def _encode_desensitize(
    path: str,
    output: str,
    opts: DesensitizeOptions,
    state: _DesensitizeState,
    progress_cb,
    should_stop,
    pause,
) -> None:
    """处理遍：音轨准备 + 逐块解码→变换→流式编码 + 音画 mux。"""

    def check_cancelled() -> None:
        if should_stop and should_stop():
            raise InterruptedError("任务已取消")

    def check_pause() -> None:
        if pause is None:
            return
        while pause.is_set():
            if should_stop and should_stop():
                raise InterruptedError("任务已取消")
            time.sleep(0.2)

    # 音轨独立准备放入后台线程，与视频解码/变换/编码重叠；结果在 mux 前
    # 取回。内存峰值与旧串行实现一致（音频数组本来就会驻留到 mux）。
    # 取舍：坏音轨的失败时机从「编码前」后移到「mux 前」（罕见路径，
    # 错误与产物状态不变），换来音频阶段被完整隐藏。
    audio_box: list[tuple[np.ndarray | None, int] | BaseException] = []

    def prepare_audio_worker() -> None:
        try:
            audio_box.append(_prepare_audio(path, opts, state))
        except BaseException as exc:  # noqa: BLE001 - 统一在 mux 前抛给主线程
            audio_box.append(exc)

    audio_thread = threading.Thread(
        target=prepare_audio_worker,
        daemon=True,
        name="cthulhu-audio-prep",
    )
    audio_thread.start()

    # ---------- 处理遍：逐块解码 → 变换 → 流式编码 ----------
    # 中间文件放系统临时目录（每任务独立子目录），避免污染输出/素材目录；
    # 完成后整目录清理，跨卷移动由 shutil.move 兜底。
    task_temp = tempfile.mkdtemp(prefix="cthulhu-task-")
    temp_video = os.path.join(task_temp, "video.mp4")
    # YUV 快路径自建分段/单段编码器写视频轨；RGB 路径用共享编码器。
    encoder = (
        None
        if state.use_yuv_path
        else ffmpeg.StreamingEncoder(
            temp_video,
            state.info["width"],
            state.info["height"],
            state.output_fps,
            codec=opts.codec,
            hardware=opts.hardware,
            crf=state.crf,
            preset=opts.preset,
            gop=opts.gop,
            bitrate_kbps=state.bitrate_kbps,
            out_size=state.out_size,
            color=True,
            filters=None if state.native_complex else (state.native_chain or None),
            complex_filter=state.native_complex,
            stop=should_stop,
        )
    )
    out_index = 0
    # 净化控制面走线程本地，不进入 TransformContext：避免把 stop/pause 回调
    # 塞进多进程池的任务载荷导致 pickle 失败。base/count 由每批处理前更新。
    purify_progress_state = {"base": 0, "count": 1}

    def purify_progress(fraction: float, note: str) -> None:
        if not progress_cb or not state.total_out:
            return
        if note.startswith("下载"):
            percent = 8 + int(purify_progress_state["base"] / state.total_out * 82)
            progress_cb(min(90, percent), note)
        else:
            percent, global_note = _purify_progress(
                purify_progress_state["base"],
                purify_progress_state["count"],
                fraction,
                state.total_out,
            )
            progress_cb(percent, global_note)

    if state.purify_enabled:
        purify.set_control(
            should_stop=should_stop,
            pause=pause,
            progress=purify_progress,
        )

    try:
        if state.use_yuv_path:
            out_index = _encode_yuv_fast(
                path, temp_video, opts, state, progress_cb, should_stop, pause
            )
        elif state.speed == 1.0 and not opts.shot_retime and opts.cut_jitter == 0:
            # 默认快路径：按重排后的镜头顺序流式处理，每个镜头只 seek 一次，
            # 消除逐块 decode_video_range 从头重复解码丢弃的 O(N²) 开销。
            # 武器阶段 CPU 密集，多进程并行处理帧块；首块在主进程处理以
            # 预计算 deep/copy/face 的一次性 SPSA 场并分发到后续进程。
            # 净化管线持有大模型且推理不跨进程回流失败原因：净化任务强制单进程。
            worker_count = 1 if state.purify_enabled else _parallel_workers()
            par_chunk = max(8, state.chunk // worker_count) if worker_count > 1 else state.chunk
            for seg_index, (seg_start, orig_start, seg_len) in enumerate(state.segments):
                check_cancelled()
                seg_options = state.transform_options
                if state.shot_options is not None and state.seg_shot_indices is not None:
                    shot_index = state.seg_shot_indices[seg_index]
                    if 0 <= shot_index < len(state.shot_options):
                        seg_options = state.shot_options[shot_index]
                decoder = ffmpeg.StreamingDecoder(path, orig_start, seg_len, grayscale=False)
                pool = None
                try:
                    batches = _prefetch_batches(decoder, seg_len, par_chunk, state.frame_dtype)
                    base_seed = int(state.transform_context.seed) ^ (1000003 * seg_index)

                    def process(
                        batch: np.ndarray,
                        output_ids: list[int],
                        seed: int,
                        rng: np.random.Generator,
                        options: Any,
                    ):
                        import dataclasses as dc

                        ctx = state.transform_context
                        if ctx.mid_rng is not None:
                            ctx = dc.replace(ctx, mid_rng=np.random.default_rng(seed ^ 0x51A))
                        purify_progress_state["base"] = out_index
                        purify_progress_state["count"] = len(batch)
                        frames = state.strategy.apply(
                            batch, output_ids, ctx, options
                        )
                        return _apply_attacks(state, frames, output_ids, rng=rng)

                    def emit(frames: np.ndarray) -> None:
                        nonlocal out_index
                        encoder.write(frames)
                        out_index += len(frames)
                        if progress_cb and state.total_out:
                            progress_cb(
                                8 + int(out_index / state.total_out * 82),
                                f"处理中 {out_index}/{state.total_out} 帧",
                            )

                    # 首块在主进程处理：SPSA 场只在这里计算一次。
                    pos, batch = next(batches)
                    output_ids = list(range(seg_start + pos, seg_start + pos + len(batch)))
                    rng0 = np.random.default_rng(base_seed ^ pos)
                    emit(process(batch, output_ids, base_seed ^ pos, rng0, seg_options))
                    spsa_fields = _collect_spsa_fields(state, rng0)

                    if worker_count > 1:
                        import multiprocessing

                        pool = multiprocessing.get_context("spawn").Pool(
                            worker_count,
                            initializer=_worker_init,
                            initargs=(spsa_fields,),
                        )

                        def tasks(
                            batches=batches,
                            base_seed=base_seed,
                            seg_start=seg_start,
                            seg_options=seg_options,
                        ):
                            for pos, batch in batches:
                                check_cancelled()
                                seed = base_seed ^ pos
                                yield {
                                    "state": state,
                                    "batch": batch,
                                    "options": seg_options,
                                    "output_ids": list(
                                        range(seg_start + pos, seg_start + pos + len(batch))
                                    ),
                                    "chunk_seed": seed,
                                }

                        for frames in pool.imap(_transform_chunk_worker, tasks(), chunksize=4):
                            check_cancelled()
                            check_pause()
                            emit(frames)
                    else:
                        for pos, batch in batches:
                            check_cancelled()
                            check_pause()
                            output_ids = list(
                                range(seg_start + pos, seg_start + pos + len(batch))
                            )
                            emit(
                                process(
                                    batch,
                                    output_ids,
                                    base_seed ^ pos,
                                    rng0,
                                    seg_options,
                                )
                            )
                finally:
                    if pool is not None:
                        pool.terminate()
                        pool.join()
                        pool.close()
                    decoder.close()
        else:
            # O(N) 流式变速路径：逐镜头因子映射，每个镜头只 seek 一次。
            # 避免逐块随机访问解码的 O(N²) 开销。
            for si, (seg_start, orig_start, seg_len) in enumerate(state.segments):
                check_cancelled()
                seg_options = state.transform_options
                if state.shot_options is not None and state.seg_shot_indices is not None:
                    shot_index = state.seg_shot_indices[si]
                    if 0 <= shot_index < len(state.shot_options):
                        seg_options = state.shot_options[shot_index]
                factor = state.seg_factors[si]
                seg_out_len = state.seg_out_lens[si]
                seg_out_index = 0
                decoder = ffmpeg.StreamingDecoder(path, orig_start, seg_len, grayscale=False)
                try:
                    for pos, batch in _prefetch_batches(
                        decoder, seg_len, state.chunk, state.frame_dtype
                    ):
                        check_cancelled()
                        check_pause()
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
                            purify_progress_state["base"] = out_index
                            purify_progress_state["count"] = len(frames)
                            frames = state.strategy.apply(
                                frames, output_ids, state.transform_context, seg_options
                            )
                            frames = _apply_attacks(state, frames, output_ids)
                            encoder.write(frames)
                            out_index += len(frames)
                            if progress_cb and state.total_out:
                                progress_cb(
                                    8 + int(out_index / state.total_out * 82),
                                    f"处理中 {out_index}/{state.total_out} 帧",
                                )
                finally:
                    decoder.close()
        if encoder is not None:
            encoder.finish()
        # 视频完成后取回音轨结果：正常路径在这里才需要音频，后台准备
        # 已与整个视频阶段重叠；音频失败在此处抛出并走统一清理。
        audio_thread.join()
        audio_result = audio_box[0] if audio_box else None
        if isinstance(audio_result, BaseException):
            raise audio_result
        audio_signal, sample_rate = audio_result
        _mux_output(
            path, output, temp_video, out_index, audio_signal, sample_rate, opts, state
        )
        shutil.rmtree(task_temp, ignore_errors=True)
        if progress_cb:
            progress_cb(95, "编码完成")
        check_cancelled()
    except BaseException:
        if encoder is not None:
            encoder.abort()
        shutil.rmtree(task_temp, ignore_errors=True)
        raise
    finally:
        if state.purify_enabled:
            purify.clear_control()
