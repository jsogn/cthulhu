"""GUI 后端服务层：把研究核心封装为可复用的业务操作。"""

from __future__ import annotations

import hashlib
import math
import os
import shutil
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.ndimage import zoom

from cthulhu_backend import db, samples
from cthulhu_backend.bitstream import analyze as bitstream_analyze
from cthulhu_backend.evaluate import metrics
from cthulhu_backend.media import container, ffmpeg
from cthulhu_backend.pipeline import (
    _DesensitizeState,
    _encode_desensitize,
    _mux_output,
    prepare_desensitize,
)
from cthulhu_backend.schemas import DesensitizeOptions
from cthulhu_backend.similarity import embedding
from cthulhu_backend.transform import purify
from cthulhu_backend.watermark import detect

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".flv", ".ts", ".webm", ".m4v"}

IDENTICAL_SIMILARITY = {
    "content_cosine": 1.0,
    "motion_cosine": 1.0,
    "dhash_agreement": 1.0,
    "ssim_mean": 1.0,
    "reduction": {"content": 0.0, "motion": 0.0, "dhash": 0.0},
}


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


def run_detect(path: str, progress_cb=None, should_stop=None, pause=None) -> dict:
    """检测：容器/元数据/SEI + 压缩域（码流层）分析，按阶段汇报进度。"""
    path = _require_file(path)

    def step(percent: int, note: str) -> None:
        if progress_cb:
            progress_cb(percent, note)

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

    check_cancelled()
    check_pause()
    step(5, "读取视频信息")
    probe = ffmpeg.video_info(path)
    check_cancelled()
    check_pause()
    step(20, "扫描容器元数据")
    container_report = container.scan_mp4(path)
    sei_count = container.count_sei(path)
    check_cancelled()
    check_pause()
    step(50, "码流层分析")
    # 容器扫描与 SEI 统计在 bitstream.analyze 内部还会用到：把本遍的结果
    # 传入复用，避免对同一文件重复整文件读取与全片 trace_headers 扫描。
    bitstream = bitstream_analyze.analyze(
        path, max_frames=300, scan=container_report, sei_count=sei_count
    )
    report = {
        "probe": probe,
        "container": container_report,
        "sei_count": sei_count,
        "bitstream": bitstream,
    }
    check_cancelled()
    check_pause()
    step(78, "盲检测抽样")
    try:
        # 跨全片抽样 300 帧逐窗口打分，内存与视频总长解耦。
        frames, _ = ffmpeg.decode_sampled(path, cap=300)
        check_cancelled()
        check_pause()
        step(90, "音频分析")
        blind, confidence = detect.windowed_video_scores_with_confidence(frames)
        structural = detect.structural_scores(frames)
        color_frames, _ = ffmpeg.decode_sampled(
            path, cap=60, grayscale=False, scale_long_edge=detect.STATS_LONG_EDGE
        )
        blind["chroma"] = detect.chroma_blind(color_frames)
        del color_frames
        # 检测只做抽样，取前 120 秒音轨即可，避免长视频整段解码。
        audio = ffmpeg.decode_audio(path, max_seconds=120)
        check_cancelled()
        if audio is not None:
            signal, sample_rate = audio
            blind["echo"] = detect.audio_scores(signal, sample_rate)["echo"]
        report["blind"] = blind
        report["blind_structural"] = structural
        report["blind_confidence"] = confidence
        report["thresholds"] = detect.CALIBRATED_THRESHOLDS
        report["hits"] = detect.hits(blind, structural, confidence)
    except InterruptedError:
        raise
    except Exception:  # noqa: BLE001 - 盲检测失败不影响压缩域报告
        report["blind"] = None
    step(95, "写入报告")
    db.save_report(path, report)
    return report


def _temporal_aligned_vmaf(
    ref: np.ndarray,
    mov: np.ndarray,
    matches: np.ndarray,
    sample_pairs: int = 4,
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


def run_similarity(a: str, b: str) -> dict:
    a, b = _require_file(a), _require_file(b)
    # 相似度报告最多消费 60 帧，全片 float64 解码毫无必要且会在长片上
    # 造成十数 GB 瞬时峰值；改为跨片抽样 float32，内存有界。
    frames_a, _ = ffmpeg.decode_sampled(a, cap=200)
    frames_b, _ = ffmpeg.decode_sampled(b, cap=200)
    return embedding.similarity_report(frames_a, frames_b)


def _scan_entry(entry: Path) -> dict:
    """探测单个导入文件；失败仅记录错误，不中断整批。"""
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
    return item


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
    # ffprobe 各自启动独立进程，线程池并行探测可缩短大批量文件夹导入的耗时。
    workers = min(max(1, len(entries)), 8, os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        files = list(pool.map(_scan_entry, entries))
    return {
        "directory": target.is_dir(),
        "files": files,
        "valid": sum(1 for item in files if item["video"]),
        "invalid": sum(1 for item in files if not item["video"]),
    }


LIBRARY_DIR = Path(
    os.environ.get(
        "CTHULHU_LIBRARY_DIR",
        str(Path(__file__).resolve().parents[2] / "data" / "library"),
    )
)


def probe_video(path: str) -> dict | None:
    """读取视频元数据；失败返回 None（不阻断导入，仅缺失展示信息）。"""
    try:
        return ffmpeg.video_info(path)
    except Exception:  # noqa: BLE001 - 元数据失败不影响素材导入
        return None


def content_hash(path: str) -> str:
    """分块计算文件 MD5，避免大文件一次性读入内存。"""
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class DuplicateImport(Exception):
    """内容完全相同的素材重复登记：由 API 层翻译为 409 提示。"""

    def __init__(self, existing_name: str) -> None:
        super().__init__(existing_name)
        self.existing_name = existing_name


def ingest_upload(
    tmp_path: str,
    *,
    stem: str,
    ext: str,
    size: int,
    content_hash_value: str,
) -> dict:
    """网页导入落地：内容去重 → 冲突命名 → 落盘 → 登记素材库。

    接收已写入临时文件的上传内容，负责文件归属与库记录；HTTP 层只做
    流式落盘、扩展名校验与空文件拦截。
    """
    records = db.list_library()
    record_by_path = {record["path"]: record for record in records}
    candidates = [
        Path(record["path"])
        for record in records
        if (record.get("size") or 0) == size and os.path.isfile(record["path"])
    ]
    LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    candidates += [
        existing
        for existing in LIBRARY_DIR.glob(f"*{ext}")
        if existing.is_file() and existing.stat().st_size == size and existing not in candidates
    ]
    for existing in candidates:
        record = record_by_path.get(str(existing))
        existing_hash = (record.get("meta") or {}).get("content_hash") if record else None
        if not existing_hash:
            existing_hash = content_hash(str(existing))
            if record is not None:
                cached_meta = record.get("meta") or {}
                cached_meta["content_hash"] = existing_hash
                db.update_library_meta(record["path"], cached_meta)
        if existing_hash == content_hash_value:
            os.unlink(tmp_path)
            meta = probe_video(str(existing)) or {}
            meta["content_hash"] = existing_hash
            db.add_library(str(existing), existing.name, size, meta)
            return {
                "path": str(existing),
                "name": existing.name,
                "size": size,
                "video": meta or None,
                "duplicate": True,
            }
    target = LIBRARY_DIR / f"{stem}{ext}"
    counter = 1
    while target.exists():
        target = LIBRARY_DIR / f"{stem}_{counter}{ext}"
        counter += 1
    os.replace(tmp_path, target)
    meta = probe_video(str(target)) or {}
    meta["content_hash"] = content_hash_value
    db.add_library(str(target), target.name, size, meta)
    return {
        "path": str(target),
        "name": target.name,
        "size": size,
        "video": meta or None,
        "duplicate": False,
    }


def register_library_path(path: str) -> dict:
    """登记本地视频到素材库；同路径别名允许刷新，内容相同抛 DuplicateImport。"""
    records = db.list_library()
    real = os.path.realpath(path)
    # 同一路径（含符号链接别名）重复导入：允许刷新时间，不算重复。
    if any(os.path.realpath(record["path"]) == real for record in records):
        content_hash_value = None
    else:
        size = os.path.getsize(path)
        candidates = [
            record
            for record in records
            if (record.get("size") or 0) == size and os.path.isfile(record["path"])
        ]
        if not candidates:
            content_hash_value = None
        else:
            content_hash_value = content_hash(path)
            for record in candidates:
                meta = record.get("meta") or {}
                if meta.get("content_hash") is None:
                    meta["content_hash"] = content_hash(record["path"])
                    db.update_library_meta(record["path"], meta)
                if meta.get("content_hash") == content_hash_value:
                    raise DuplicateImport(record["name"])
    record = {
        "path": path,
        "name": Path(path).name,
        "size": os.path.getsize(path),
    }
    meta = probe_video(path) or {}
    if content_hash_value:
        meta["content_hash"] = content_hash_value
    record["meta"] = meta
    db.add_library(record["path"], record["name"], record["size"], record["meta"])
    return record


_DEFERRED_METRICS_LOCK = threading.Lock()

# 清洗结果里的指标字段统一清单：同步/异步指标遍与后台回写共用同一份键名，
# 避免三处手写字典漂移。
_RESULT_METRIC_KEYS = (
    "similarity_before",
    "similarity_after",
    "psnr_db",
    "ssim",
    "vmaf",
    "vmaf_aligned",
    "quality_metrics_na",
    "export_health",
)


def _result_template(
    path: str,
    output: str,
    opts: DesensitizeOptions,
    state: _DesensitizeState,
) -> dict:
    """清洗结果基础结构：指标字段初始为 None，由指标遍填充。"""
    return {
        "input": path,
        "output": output,
        "transform_strategy": state.strategy.name,
        "preset": opts.preset,
        "frames": state.total_in,
        # 成片帧数：回声清除等同步变速会改变时长，元数据必须如实反映产物，
        # 否则调用方按 frames 算时长会与文件对不上。
        "out_frames": state.total_out,
        "purify_note": purify_note(opts, state),
        "quality_gate_note": _quality_gate_note(opts, state),
        "profile_note": getattr(state, "profile_note", "off"),
        "profile_metrics": getattr(state, "profile_metrics", None),
        **{key: None for key in _RESULT_METRIC_KEYS},
    }


def purify_note(opts: DesensitizeOptions, state) -> str:
    """净化状态说明：关闭 / 已应用 / 依赖缺位跳过 / 中途加载失败回退。"""
    del opts
    note = getattr(state, "purify_note", "off")
    options = getattr(state, "transform_options", None)
    if note in {"applied", "will_download"} and options is not None:
        extras: list[str] = ["潜空间重建"]
        detail = getattr(options, "purify_detail", 0.0)
        temporal = getattr(options, "purify_temporal", 0.0)
        if detail > 0:
            sigma = getattr(options, "purify_detail_sigma", 0.0)
            wide = bool(getattr(options, "purify_detail_wide", False))
            if sigma > 0:
                band = f"σ{sigma:.1f}"
            else:
                band = "σ自动·宽带回注+字幕增强" if wide else "σ自动"
            extras.append(f"detail={detail:.2f}@{band}")
        if temporal > 0:
            extras.append(f"temporal={temporal:.2f}")
        if extras:
            return f"{note} ({', '.join(extras)})"
    return note


def _quality_gate_note(opts: DesensitizeOptions, state) -> str:
    """画质门控状态：门控只作用于策略层之后的攻击层，净化不在门控范围内。"""
    if not opts.quality_protect:
        return "off"
    if getattr(state, "purify_enabled", False):
        return "applied: attack layer only (purify excluded)"
    return "applied"


def _deferred_measure(
    path: str,
    output: str,
    opts: DesensitizeOptions,
    state: _DesensitizeState,
    metrics_gate=None,
) -> None:
    """后台指标遍：任务先完成，指标算完后回写产物记录。

    metrics_gate 为空闲闸门（任务队列忙时返回 False）：闸门未放行前不启动
    计算，避免低配机器上指标遍与下一批编码任务抢核；放行后全局串行执行。
    指标回写带重试，容忍调用方「先返回、再落 variant 记录」的时序差。
    进程退出时线程随之结束，未回写的指标允许丢失（产物本身不受影响）。
    """
    if metrics_gate is not None:
        while not metrics_gate():
            time.sleep(1.0)
    try:
        with _DEFERRED_METRICS_LOCK:
            report = _measure_desensitize(path, output, opts, state, None)
    except Exception:  # noqa: BLE001 - 指标失败不影响已完成的产物
        return
    payload = {key: report.get(key) for key in _RESULT_METRIC_KEYS}
    payload["purify_note"] = report.get("purify_note")
    payload["quality_gate_note"] = report.get("quality_gate_note")
    payload["profile_note"] = report.get("profile_note")
    payload["profile_metrics"] = report.get("profile_metrics")
    payload["quality_gate_note"] = report.get("quality_gate_note")
    for _ in range(10):
        if db.update_variant_metrics(output, payload):
            return
        time.sleep(0.2)


def run_desensitize(
    path: str,
    output: str,
    *,
    compute_metrics: bool = True,
    defer_metrics: bool = False,
    metrics_gate=None,
    progress_cb=None,
    should_stop=None,
    pause=None,
    **options,
) -> dict:
    """内容脱敏：分析遍 + 分块流式处理遍 + 指标遍，内存与视频总长度解耦。

    选项经 DesensitizeOptions 归一化（未知键与旧版显式签名一样直接报错），
    内部分为 准备/编码/指标 三个阶段执行。
    compute_metrics=False 时编码完成即返回，不启动指标遍；供 GUI 任务队列
    使用（画质/相似度指标无业务消费，且指标遍会在低配机上造成内存尖峰）。
    直接 API、CLI 与基准研究保持默认 True，继续输出完整指标。
    defer_metrics=True 时指标遍转入后台线程，任务即刻完成，指标随后回写
    variants 记录；适合任务队列等「完成即反馈」的调用方。
    metrics_gate 配合 defer_metrics 使用：返回 False 时后台指标计算等待，
    供任务队列在忙碌期间错峰计算指标。
    """
    opts = DesensitizeOptions(**options)
    if opts.purify_strength > 0:
        purify.reset_failure()
    path = _require_file(path)
    output = os.path.expanduser(output)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    def check_cancelled() -> None:
        if should_stop and should_stop():
            raise InterruptedError("任务已取消")

    if progress_cb:
        progress_cb(5, "准备处理")
    if not ffmpeg.has_encoder(opts.codec):
        raise ValueError(f"当前 FFmpeg 缺少视频编码器 {opts.codec}，请更换输出编码或安装完整版 FFmpeg")
    process_path = path
    state = prepare_desensitize(process_path, opts, progress_cb, check_cancelled)
    _encode_desensitize(process_path, output, opts, state, progress_cb, should_stop, pause)
    # 净化在分块处理中途失败时，把线程本地的失败原因固化进状态，
    # 这样异步指标线程/结果模板也能拿到同一条降级说明。
    if opts.purify_strength > 0:
        failure = purify.last_failure()
        if failure:
            state.purify_note = f"fallback: {failure}"
    if not compute_metrics:
        return {
            **_result_template(path, output, opts, state),
            # GUI 产物不再自动回填指标：任务完成即产物就绪。
            "similarity_before": IDENTICAL_SIMILARITY,
            "quality_metrics_na": True,
        }
    if defer_metrics:
        threading.Thread(
            target=_deferred_measure,
            args=(path, output, opts, state, metrics_gate),
            daemon=True,
            name="cthulhu-deferred-metrics",
        ).start()
        return {
            **_result_template(path, output, opts, state),
            # 与自身比较恒为 1：异步路径同样直接给出，与同步指标遍口径一致。
            "similarity_before": IDENTICAL_SIMILARITY,
            "metrics_pending": True,
        }
    return _measure_desensitize(path, output, opts, state, progress_cb)

def _measure_desensitize(
    path: str,
    output: str,
    opts: DesensitizeOptions,
    state: _DesensitizeState,
    progress_cb,
) -> dict:
    """指标遍：抽样重解码后计算，避免整片驻留。"""
    if progress_cb:
        progress_cb(97, "计算画质指标")
    # 原片与产物并行抽样解码：指标口径对抽样帧数不敏感，60 帧足够稳定。
    with ThreadPoolExecutor(max_workers=2) as pool:
        (original_sampled, _), (processed_sampled, _) = pool.map(
            lambda src: ffmpeg.decode_sampled(src, cap=60),
            (path, output),
        )
    ref_s, mov_s, matches = metrics.temporal_match(original_sampled, processed_sampled)
    # 几何去同步（旋转）使逐像素画质指标失去对齐口径，数值会误导用户。
    quality_na = opts.rotate > 0
    report = _result_template(path, output, opts, state)
    # 与自身比较恒为 1，直接给出常量，省去一次全量 embedding。
    report["similarity_before"] = IDENTICAL_SIMILARITY
    report["similarity_after"] = embedding.similarity_report(original_sampled, processed_sampled)
    report["psnr_db"] = None if quality_na else round(metrics.matched_psnr(ref_s, mov_s, matches), 2)
    report["ssim"] = None if quality_na else round(metrics.matched_ssim(ref_s, mov_s, matches), 4)
    # 重排/变速后时间轴错位，朴素 VMAF 恒近 0 无参考价值，改用对齐分。
    report["vmaf"] = None
    report["vmaf_aligned"] = (
        None
        if (quality_na or opts.skip_vmaf)
        else _temporal_aligned_vmaf(ref_s, mov_s, matches, fps=state.output_fps)
    )
    report["quality_metrics_na"] = quality_na
    report["export_health"] = _export_health(output)
    return report


def _resize_frames(frames: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    from PIL import Image

    from cthulhu_backend.parallel import map_frames

    width, height = size

    def resize(frame: np.ndarray) -> np.ndarray:
        image = Image.fromarray(frame, mode="RGB")
        return np.asarray(image.resize((width, height), Image.BICUBIC), dtype=np.uint8)

    return map_frames(resize, frames)


def _frames_to_u8(frames: np.ndarray) -> np.ndarray:
    """decode_video 可能返回 [0,1] float；统一转 uint8 再参与平均/编码。"""
    if frames.dtype == np.uint8:
        return frames
    return (np.clip(frames, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def run_collusion(
    paths: list[str],
    output: str,
    *,
    mode: str = "mean",
    max_frames: int = 600,
    progress_cb=None,
    should_stop=None,
    pause=None,
) -> dict:
    """共谋平均：同一内容的多份不同水印副本对齐后平均，冲掉各副本水印。

    报告结论：8/16/32 副本平均 → BA 0.599/0.575/0.542，PSNR 65~67dB、
    SSIM≈1.0；需要同一内容、不同水印的副本，属于研究/授权测试用途。
    """
    if len(paths) < 2:
        raise ValueError("共谋平均至少需要 2 个副本")
    if len(paths) > 32:
        raise ValueError("共谋平均最多支持 32 个副本")
    if mode not in {"mean", "median"}:
        raise ValueError(f"未知共谋模式：{mode}（可选 mean / median）")
    sources = [_require_file(path) for path in paths]
    output = os.path.expanduser(output)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    def check_cancelled() -> None:
        if should_stop and should_stop():
            raise InterruptedError("任务已取消")
        if pause is not None:
            while pause.is_set():
                if should_stop and should_stop():
                    raise InterruptedError("任务已取消")
                time.sleep(0.2)

    decoded: list[np.ndarray] = []
    target_size: tuple[int, int] | None = None
    fps: float | None = None
    for index, source in enumerate(sources):
        check_cancelled()
        if progress_cb:
            progress_cb(
                5 + int(index / len(sources) * 55),
                f"解码副本 {index + 1}/{len(sources)}",
            )
        frames, info = ffmpeg.decode_video(
            source, grayscale=False, out_dtype="uint8", max_frames=max_frames
        )
        frames = _frames_to_u8(frames)
        if len(frames) == 0:
            raise ValueError(f"副本没有可解码帧：{source}")
        width, height = int(info["width"]), int(info["height"])
        if target_size is None:
            target_size = (width, height)
            fps = float(info["fps"])
        elif (width, height) != target_size:
            frames = _resize_frames(frames, target_size)
        decoded.append(frames)

    count = min(len(frames) for frames in decoded)
    if count < 2:
        raise ValueError("副本帧数不足，无法共谋平均")
    if progress_cb:
        progress_cb(65, f"平均 {len(decoded)} 个副本")
    stack = np.stack([frames[:count].astype(np.float32) for frames in decoded], axis=0)
    averaged = np.median(stack, axis=0) if mode == "median" else np.mean(stack, axis=0)
    result_frames = np.clip(averaged, 0, 255).round().astype(np.uint8)

    check_cancelled()
    if progress_cb:
        progress_cb(75, "编码共谋结果")
    with tempfile.TemporaryDirectory(prefix="cthulhu-collusion-") as tmp:
        video_only = os.path.join(tmp, "video.mp4")
        ffmpeg.encode_video(result_frames, video_only, fps=fps or 30.0, crf=18)
        _mux_output(
            sources[0],
            output,
            video_only,
            count,
            None,
            16000,
            DesensitizeOptions(),
            SimpleNamespace(output_fps=fps or 30.0, audio_tempo=None),
        )
    if progress_cb:
        progress_cb(95, "计算画质")
    reference_float = result_frames.astype(np.float32) / 255.0
    quality = [
        {
            "psnr_db": round(
                metrics.psnr(frames[:count].astype(np.float32) / 255.0, reference_float), 2
            ),
            "ssim": round(
                metrics.ssim(frames[:count].astype(np.float32) / 255.0, reference_float), 4
            ),
        }
        for frames in decoded
    ]
    return {
        "paths": sources,
        "output": output,
        "copies": len(sources),
        "frames": count,
        "mode": mode,
        "fps": fps,
        "resolution": list(target_size or (0, 0)),
        "quality": quality,
        "estimated_watermark_reduction": round(1.0 / math.sqrt(len(sources)), 4),
    }


def export_outputs(paths: list[str], export_dir: str) -> dict:
    """把源视频已生成的清洗/修复产物复制到导出目录，并导出产物记录清单。

    产物按清洗优先、修复其次的规则匹配，同一素材存在多个产物时取最新修改的。
    源视频尚未处理时归入 missing，由前端提示用户先执行清洗或修复。
    """
    export_root = Path(os.path.expanduser(export_dir or "~/Documents/Cthulhu"))
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
        # 目标目录正是产物所在目录时无需复制自己。
        if product.resolve() != destination.resolve():
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
        f"{source_path.stem}_修复*.mp4",
        f"{source_path.stem}_repaired*.mp4",
        f"{source_path.stem}_候选*.mp4",
    ):
        for candidate in source_path.parent.glob(pattern):
            name = candidate.name
            # 排除中间产物：分段文件（.segN）、视频轨临时（.video.mp4）、
            # 转码链临时（.chain.mp4 / .final.mp4），避免处理中/异常残留混入列表。
            if candidate.is_file() and not any(
                token in name for token in (".video.mp4", ".seg", ".chain.mp4", ".final.mp4")
            ):
                paths.add(candidate)
    return list(paths)


def _product_kind(path: Path) -> str:
    name = path.name
    if "_修复" in name:
        return "repaired"
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
    """删除产物：优先按产物记录删除；无记录的历史产物才按命名规则放行。"""
    target = Path(path)
    if db.get_variant_by_output(path):
        removed_file = False
        if target.is_file():
            target.unlink()
            removed_file = True
        removed_record = db.delete_variant_by_output(path)
        return removed_file or removed_record
    name = target.name
    if not any(part in name for part in ("_清洗", "_cleaned", "_修复", "_repaired", "_候选")):
        raise ValueError("仅允许删除清洗/修复/候选产物")
    removed_file = False
    if target.is_file():
        target.unlink()
        removed_file = True
    removed_record = db.delete_variant_by_output(str(target))
    return removed_file or removed_record


def list_all_products() -> dict:
    """全局产物清单：跨全部源素材汇总清洗/修复产物，供产物管理页查看占用与清理。"""
    items: dict[str, dict] = {}
    # 优先汇总 variants 记录，覆盖源素材已从素材库移除但产物仍在的历史记录。
    for record in db.list_variants():
        output = record.get("output")
        if not output:
            continue
        path = Path(os.path.expanduser(output))
        key = str(path)
        kind = record.get("kind") or "cleaned"
        if kind == "candidate":
            continue
        exists = path.is_file()
        items[key] = {
            "path": key,
            "name": path.name,
            "kind": kind,
            "size": path.stat().st_size if exists else 0,
            "mtime": path.stat().st_mtime if exists else record.get("created_at") or 0,
            "source": record.get("source") or "",
            "exists": exists,
        }
    # 补齐素材库内尚未写入记录的历史产物。
    for item in db.list_library():
        source = item["path"]
        for candidate in _product_candidates(source):
            kind = _product_kind(candidate)
            if kind == "candidate":
                continue
            key = str(candidate)
            if key in items:
                continue
            exists = candidate.is_file()
            items[key] = {
                "path": key,
                "name": candidate.name,
                "kind": kind,
                "size": candidate.stat().st_size if exists else 0,
                "mtime": candidate.stat().st_mtime if exists else 0,
                "source": source,
                "exists": exists,
            }
    products = sorted(items.values(), key=lambda item: item["mtime"], reverse=True)
    return {
        "products": products,
        "count": len(products),
        "total_size": sum(item["size"] for item in products),
    }


def delete_products(paths: list[str]) -> dict:
    """批量删除产物：先校验全部可删（有记录或符合历史命名），再逐个删除。"""
    targets = [Path(path) for path in paths]
    for target in targets:
        if db.get_variant_by_output(str(target)):
            continue
        if not any(
            part in target.name for part in ("_清洗", "_cleaned", "_修复", "_repaired", "_候选")
        ):
            raise ValueError("仅允许删除清洗/修复/候选产物")
    removed = 0
    for target in targets:
        if delete_output(str(target)):
            removed += 1
    return {"removed": removed}


def record_variant(
    source: str,
    output: str,
    options: dict,
    seed: int,
    template_id: str | None = None,
    metrics: dict | None = None,
    kind: str = "cleaned",
) -> dict:
    """把产物参数与指标写入 variants 数据表，A/B 追溯用。"""
    return db.create_variant(
        source=source,
        output=output,
        options=options,
        seed=seed,
        template_id=template_id,
        metrics=metrics,
        kind=kind,
    )
