"""GUI 后端服务层：把研究核心封装为可复用的业务操作。"""

from __future__ import annotations

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
from cthulhu_backend.transform import shots, strategies
from cthulhu_backend.watermark import common as watermark_common
from cthulhu_backend.watermark import detect

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".flv", ".ts", ".webm", ".m4v"}
MAX_WORKING_BYTES = 2 * 1024**3


def _memory_budget_bytes() -> int:
    """按物理内存的 55% 自适应工作预算（2GB~32GB）。

    固定的 2GB 上限会把真实 1080p 素材误判为过大，
    改用实际内存自适应后，小内存机器仍受保护。
    """
    try:
        total = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, ValueError, OSError):
        total = 0
    if total <= 0:
        return MAX_WORKING_BYTES
    return max(MAX_WORKING_BYTES, min(32 * 1024**3, int(total * 0.55)))

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
    sample_pairs: int = 12,
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
    batch_queue: queue.Queue = queue.Queue(maxsize=2)

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
    reorder: bool = True,
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
    preset: str = "medium",
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
    rotate: float = 0.0,
    progress_cb=None,
    should_stop=None,
) -> dict:
    """内容脱敏：分析遍 + 分块流式处理遍，内存与视频总长度解耦。"""
    path = _require_file(path)
    output = os.path.expanduser(output)

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

    # ---------- 分析遍：抽样镜头边界 + 预生成逐帧随机参数 ----------
    if progress_cb:
        progress_cb(8, "扫描镜头结构")
    sampled, _ = ffmpeg.decode_sampled(path, cap=400)
    if reorder and len(sampled) > 2:
        stride = max(1, total_in // max(len(sampled), 1))
        boundaries = [min(total_in, cut * stride) for cut in shots.detect_cuts(sampled)]
        if not boundaries or boundaries[0] != 0:
            boundaries = [0] + boundaries
        if boundaries[-1] != total_in:
            boundaries.append(total_in)
        deduped: list[int] = []
        for boundary in boundaries:
            if not deduped or boundary > deduped[-1]:
                deduped.append(boundary)
        boundaries = deduped
    else:
        boundaries = [0, total_in]
    shot_ranges = list(pairwise(boundaries))

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(shot_ranges)) if reorder else np.arange(len(shot_ranges))
    # 重排后的输入区间（重排空间连续，便于按旧口径全局变速）。
    segments: list[tuple[int, int, int]] = []
    reord_total = 0
    for shot_index in order:
        orig_start, orig_end = shot_ranges[int(shot_index)]
        segments.append((reord_total, orig_start, orig_end - orig_start))
        reord_total += orig_end - orig_start
    total_out = max(1, round(reord_total / speed)) if speed != 1.0 else reord_total

    gammas = np.ones(total_out, dtype=np.float32)
    deltas = np.zeros(total_out, dtype=np.float32)
    if regrade:
        gamma_strength = 0.03 + 0.2 * perturb
        brightness = 0.02 + 0.06 * perturb
        for index in range(total_out):
            gammas[index] = rng.uniform(1.0 - gamma_strength, 1.0 + gamma_strength)
            deltas[index] = rng.uniform(-brightness, brightness)

    mid_rng = np.random.default_rng(seed)
    spoof_bits = None
    if spoof:
        spoof_rng = np.random.default_rng(seed ^ 0x5F3759DF)
        spoof_bits = watermark_common.payload_bits(int(spoof_rng.integers(0, 2**31)), 64)

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
        recrop=recrop,
        regrade=regrade,
        anti_reembed=anti_reembed,
        denoise=denoise,
        color_restore=color_restore,
        sharpness=sharpness,
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
    frame_dtype = getattr(strategy, "frame_dtype", "float32")
    frame_bytes = info["width"] * info["height"] * 3 * (1 if frame_dtype == "uint8" else 4)
    chunk = max(8, min(480, int(budget // 8 // max(frame_bytes, 1))))

    # 音轨独立准备（整段重混后统一 mux）。
    audio_signal = None
    sample_rate = 16000
    if audio_remix:
        decoded_audio = ffmpeg.decode_audio(path)
        if decoded_audio is not None:
            signal, sample_rate = decoded_audio
            audio_rng = np.random.default_rng(seed ^ 0x9E3779B9)
            signal = audio_transform.remix(signal, sample_rate, audio_rng, speed_factor=0.97)
            target_len = round(total_out / output_fps * sample_rate)
            if target_len != len(signal):
                signal = resample_poly(signal, target_len, len(signal))
            audio_signal = signal

    # ---------- 处理遍：逐块解码 → 变换 → 流式编码 ----------
    temp_video = output + ".video.mp4"
    encoder = ffmpeg.StreamingEncoder(
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
        stop=should_stop,
    )
    out_index = 0
    try:
        if speed == 1.0:
            # 默认路径：按重排后的镜头顺序流式处理，每个镜头只 seek 一次，
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
                        if rotate > 0:
                            frames = strategies.rotate_de_sync(frames, output_ids, rotate)
                        if phash_attack:
                            frames = adversarial.attack_frames(
                                frames, epsilon=phash_epsilon, iterations=phash_iters
                            )
                        encoder.write(frames)
                        out_index += len(batch)
                        if progress_cb and total_out:
                            progress_cb(
                                8 + int(out_index / total_out * 82),
                                f"处理中 {out_index}/{total_out} 帧",
                            )
                finally:
                    decoder.close()
        else:
            # 变速路径：保留逐块随机访问解码（地板映射要求精确帧对齐）。
            for reord_block_start in range(0, reord_total, chunk):
                check_cancelled()
                reord_block_end = min(reord_total, reord_block_start + chunk)
                block_parts: list[np.ndarray] = []
                for seg_start, orig_start, seg_len in segments:
                    overlap_start = max(reord_block_start, seg_start)
                    overlap_end = min(reord_block_end, seg_start + seg_len)
                    if overlap_start >= overlap_end:
                        continue
                    part, _ = ffmpeg.decode_video_range(
                        path,
                        orig_start + (overlap_start - seg_start),
                        overlap_end - overlap_start,
                        grayscale=False,
                    )
                    if len(part):
                        block_parts.append(part)
                if not block_parts:
                    continue
                block_frames = (
                    np.concatenate(block_parts, axis=0) if len(block_parts) > 1 else block_parts[0]
                )
                del block_parts
                # 变速：收集本块对应的全局输出帧号（floor(j*speed) 落在块内）。
                output_ids: list[int] = []
                while out_index < total_out:
                    if int(np.floor(out_index * speed)) >= reord_block_end:
                        break
                    output_ids.append(out_index)
                    out_index += 1
                if not output_ids:
                    continue
                source_ids = np.clip(
                    np.floor(np.asarray(output_ids, dtype=np.float64) * speed)
                    - reord_block_start,
                    0,
                    len(block_frames) - 1,
                ).astype(int)
                frames = block_frames[source_ids]
                del block_frames
                if frame_dtype == "uint8":
                    frames = (np.clip(frames, 0.0, 1.0) * 255.0).round().astype(np.uint8)
                frames = strategy.apply(frames, output_ids, transform_context, transform_options)
                if rotate > 0:
                    frames = strategies.rotate_de_sync(frames, output_ids, rotate)
                if phash_attack:
                    frames = adversarial.attack_frames(
                        frames, epsilon=phash_epsilon, iterations=phash_iters
                    )
                encoder.write(frames)
                if progress_cb and total_out:
                    progress_cb(
                        8 + int(out_index / total_out * 82),
                        f"处理中 {out_index}/{total_out} 帧",
                    )
        encoder.finish()
        if audio_signal is not None:
            audio_payload = (
                (np.clip(audio_signal, -1, 1) * 32767).round().astype(np.int16).tobytes()
            )
            subprocess.run(
                [
                    ffmpeg.FFMPEG_BIN, "-y", "-v", "error",
                    "-i", temp_video,
                    "-f", "s16le", "-ar", str(sample_rate), "-ac", "1", "-i", "-",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "-shortest", output,
                ],
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
    except BaseException:
        encoder.abort()
        try:
            os.unlink(temp_video)
        except OSError:
            pass
        raise

    # ---------- 指标遍：抽样重解码后计算，避免整片驻留 ----------
    if progress_cb:
        progress_cb(97, "计算画质指标")
    original_sampled, _ = ffmpeg.decode_sampled(path, cap=200)
    processed_sampled, _ = ffmpeg.decode_sampled(output, cap=200)
    ref_s, mov_s, matches = metrics.temporal_match(original_sampled, processed_sampled)
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
            if quality_na
            else _temporal_aligned_vmaf(ref_s, mov_s, matches, fps=output_fps)
        ),
        "quality_metrics_na": quality_na,
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
    """把源视频已生成的清洗/修复产物复制到导出目录。

    产物按清洗优先、修复其次的规则匹配，同一素材存在多个产物时取最新修改的。
    源视频尚未处理时归入 missing，由前端提示用户先执行清洗或修复。
    """
    export_root = Path(os.path.expanduser(export_dir or "~/导出/暗水印清洗"))
    export_root.mkdir(parents=True, exist_ok=True)
    exported: list[dict] = []
    missing: list[str] = []
    for source in paths:
        source_path = Path(source)
        candidates = [
            source_path.with_name(f"{source_path.stem}_cleaned.mp4"),
            source_path.with_name(f"{source_path.stem}_repaired.mp4"),
        ]
        existing = [candidate for candidate in candidates if candidate.exists()]
        if not existing:
            missing.append(source_path.name)
            continue
        product = max(existing, key=lambda candidate: candidate.stat().st_mtime)
        destination = export_root / product.name
        shutil.copy2(product, destination)
        exported.append({"source": product.name, "dest": str(destination)})
    return {"exported": exported, "missing": missing}


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
