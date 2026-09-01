"""GUI 后端 API：检测 / 相似度 / 内容脱敏，任务进度经事件广播推送。"""

from __future__ import annotations

import asyncio
import errno
import hashlib
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
from fastapi import APIRouter, File, HTTPException, Query, Response, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, ValidationError

from cthulhu_backend import db, services
from cthulhu_backend.events import broker
from cthulhu_backend.jobs import job_queue
from cthulhu_backend.media import ffmpeg
from cthulhu_backend.schemas import DesensitizeOptions, DesensitizeRequest, TemplatePayload
from cthulhu_backend.watermark import detect as watermark_detect

router = APIRouter(prefix="/api")

# 封面缩略图磁盘缓存：素材多、反复刷新列表时不重复启动 ffmpeg 抽帧。
_THUMB_CACHE_DIR = Path(
    os.environ.get(
        "CTHULHU_THUMB_CACHE",
        str(Path(__file__).resolve().parents[2] / "data" / "thumb-cache"),
    )
)

# 界面素材封面统一使用 320px 宽；清理时只按此宽度重算期望键。
_THUMB_DEFAULT_WIDTH = 320

_VIDEO_MEDIA_TYPES = {
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".mkv": "video/x-matroska",
    ".avi": "video/x-msvideo",
    ".webm": "video/webm",
    ".flv": "video/x-flv",
    ".ts": "video/mp2t",
    ".m4v": "video/x-m4v",
}


def _library_file(record: dict) -> dict:
    """把素材库记录组装成前端可直接渲染的清单项。"""
    path = record["path"]
    meta = record.get("meta") or {}
    if not meta and os.path.isfile(path):
        meta = services.probe_video(path) or {}
        db.update_library_meta(path, meta)
    missing = not os.path.isfile(path)
    return {
        "path": path,
        "name": record["name"],
        "size": record["size"],
        "video": meta if meta and not missing else None,
        "error": "文件不存在或已被移动" if missing else None,
        "report": db.get_report(path),
    }


def _friendly_detail(exc: Exception) -> str:
    """把底层异常翻译成普通用户可理解的提示。"""
    if isinstance(exc, ValueError):
        return str(exc)
    if isinstance(exc, FileNotFoundError):
        return "找不到源文件，它可能已被移动、重命名或删除"
    if isinstance(exc, PermissionError):
        return "没有访问该文件或目录的权限，请检查文件权限"
    if isinstance(exc, subprocess.CalledProcessError):
        return "视频处理失败，文件可能已损坏或格式不受支持"
    if isinstance(exc, OSError) and exc.errno == errno.ENOSPC:
        return "磁盘空间不足，请清理空间后重试"
    if isinstance(exc, OSError):
        return "读写文件失败，请检查磁盘状态与文件是否被占用"
    return "处理失败，请重试；若持续出现，请反馈给开发者"


def _default_ffmpeg_install_dir() -> str:
    """无环境变量时的 ffmpeg 安装兜底目录，按平台放系统缓存区，
    避免在主目录或应用数据目录留下安装残留。"""
    if sys.platform == "darwin":
        return str(Path.home() / "Library" / "Caches" / "cthulhu-backend" / "ffmpeg")
    if sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return str(Path(local) / "cthulhu-backend" / "Cache" / "ffmpeg")
    return str(Path.home() / ".cache" / "cthulhu-backend" / "ffmpeg")


class PathRequest(BaseModel):
    path: str


class PairRequest(BaseModel):
    a: str
    b: str


class ProductsRequest(BaseModel):
    paths: list[str]


class ScanRequest(BaseModel):
    path: str


class TaskSpec(BaseModel):
    kind: Literal["detect", "desensitize", "repair"]
    path: str
    options: dict = {}


class RegionSpec(BaseModel):
    x: float
    y: float
    w: float
    h: float
    start: float | None = None
    end: float | None = None


class RepairRequest(BaseModel):
    path: str
    output: str
    regions: list[RegionSpec]
    crf: int = Field(23, ge=0, le=51)


class ExportRequest(BaseModel):
    paths: list[str]
    export_dir: str | None = None


class JobRequest(BaseModel):
    name: str
    parallelism: int | None = Field(None, ge=1, le=8)
    tasks: list[TaskSpec]


class PriorityRequest(BaseModel):
    priority: int = Field(0, ge=0, le=1000)


class TemplateCreate(BaseModel):
    name: str
    payload: TemplatePayload


class TemplateUpdate(BaseModel):
    name: str
    payload: TemplatePayload


@router.post("/detect")
async def detect(request: PathRequest) -> dict:
    await broker.publish({"type": "task:start", "task": "detect", "path": request.path})
    try:
        report = await asyncio.to_thread(services.run_detect, request.path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=_friendly_detail(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=_friendly_detail(exc)) from exc
    await broker.publish({"type": "task:done", "task": "detect", "path": request.path})
    return report


@router.post("/similarity")
async def similarity(request: PairRequest) -> dict:
    try:
        return await asyncio.to_thread(services.run_similarity, request.a, request.b)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=_friendly_detail(exc)) from exc


@router.post("/import/scan")
async def import_scan(request: ScanRequest) -> dict:
    try:
        return await asyncio.to_thread(services.scan_import_path, request.path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=_friendly_detail(exc)) from exc


@router.get("/demo/library")
def demo_library() -> dict:
    return services.demo_library()


@router.get("/frame")
def frame(
    path: str = Query(...),
    t: float = Query(0.0, ge=0),
) -> Response:
    """抽取视频指定时间点的帧（PNG），供预览与前后对比渲染。"""
    try:
        content = ffmpeg.extract_frame(path, t)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=_friendly_detail(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=422, detail=_friendly_detail(exc)) from exc
    return Response(content=content, media_type="image/png")


@router.get("/thumb")
def thumb(
    path: str = Query(...),
    width: int = Query(_THUMB_DEFAULT_WIDTH, ge=16, le=1920),
) -> Response:
    """返回等比缩放的首帧缩略图（JPEG），带磁盘缓存，供素材封面使用。"""
    key = _thumb_cache_key(path, width)
    if key is None:
        raise HTTPException(
            status_code=404,
            detail=_friendly_detail(FileNotFoundError(path)),
        )
    cache_file = _THUMB_CACHE_DIR / f"{key}.jpg"
    if cache_file.is_file():
        try:
            return Response(content=cache_file.read_bytes(), media_type="image/jpeg")
        except OSError:
            pass
    try:
        content = ffmpeg.extract_thumbnail(path, width)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=_friendly_detail(exc)) from exc
    try:
        _THUMB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        temp = cache_file.with_suffix(".tmp")
        temp.write_bytes(content)
        temp.replace(cache_file)
    except OSError:
        pass
    return Response(content=content, media_type="image/jpeg")


def _thumb_cache_key(path: str, width: int) -> str | None:
    """按「绝对路径 + 修改时间 + 宽度」计算封面缓存键；源文件缺失时返回 None。"""
    try:
        mtime = os.path.getmtime(path)
    except FileNotFoundError:
        return None
    return hashlib.sha1(f"{os.path.abspath(path)}|{mtime}|{width}".encode()).hexdigest()


def sweep_thumb_cache(grace_seconds: float = 3600.0) -> dict:
    """清理封面缓存：素材库当前期望的条目保留，其余孤儿文件回收。

    封面按「路径 + mtime + 宽度」定址，素材被删除、移动或 mtime 变化后，
    旧键都会成为孤儿。这里以素材库当前记录重算期望键集合，其余 .jpg
    视为孤儿清除；.tmp 为写入中间态，按宽限期保护后清除。其他非缓存
    产生的文件一律不动。删除前按宽限期保护，避免误删刚生成/正在写入的条目。
    """
    if not _THUMB_CACHE_DIR.is_dir():
        return {"kept": 0, "removed": 0, "removed_bytes": 0}
    expected = {
        f"{key}.jpg"
        for record in db.list_library()
        if (key := _thumb_cache_key(record["path"], _THUMB_DEFAULT_WIDTH))
    }
    now = time.time()
    kept = 0
    removed = 0
    removed_bytes = 0
    for entry in _THUMB_CACHE_DIR.iterdir():
        try:
            if not entry.is_file():
                continue
            age = now - entry.stat().st_mtime
        except OSError:
            continue
        if entry.name in expected:
            kept += 1
            continue
        if entry.suffix not in {".jpg", ".tmp"}:
            continue
        if age < grace_seconds:
            continue
        try:
            removed_bytes += entry.stat().st_size
            entry.unlink()
            removed += 1
        except OSError:
            pass
    return {"kept": kept, "removed": removed, "removed_bytes": removed_bytes}


@router.get("/library")
def library_list() -> dict:
    """返回持久化素材库（最新导入在前），并标记内容完全相同的重复副本。"""
    records = db.list_library()
    hashes: dict[str, int] = {}
    for record in records:
        content_hash = (record.get("meta") or {}).get("content_hash")
        if content_hash:
            hashes[content_hash] = hashes.get(content_hash, 0) + 1
    for record in records:
        meta = record.get("meta") or {}
        path = record["path"]
        if meta.get("content_hash") is None and os.path.isfile(path):
            # 仅在存在同大小文件时补算哈希，避免列表加载拖慢。
            same_size = any(
                other["path"] != path
                and (other.get("size") or 0) == (record.get("size") or 0)
                and os.path.isfile(other["path"])
                for other in records
            )
            if same_size:
                meta["content_hash"] = services.content_hash(path)
                db.update_library_meta(path, meta)
                hashes[meta["content_hash"]] = hashes.get(meta["content_hash"], 0) + 1
    files = []
    for record in records:
        item = _library_file(record)
        content_hash = (record.get("meta") or {}).get("content_hash")
        item["duplicate"] = bool(content_hash and hashes.get(content_hash, 0) > 1)
        files.append(item)
    return {"files": files, "valid": sum(1 for item in files if item["video"]), "invalid": 0}


@router.post("/library")
def library_add(request: PathRequest) -> dict:
    """登记一个本地视频到素材库；内容完全相同的文件不重复导入。"""
    path = os.path.abspath(request.path)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="文件不存在或已被移动")
    try:
        record = services.register_library_path(path)
    except services.DuplicateImport as exc:
        raise HTTPException(
            status_code=409,
            detail=f"已存在相同内容的素材：{exc.existing_name}，未重复导入",
        ) from exc
    return _library_file(record)


@router.delete("/library")
def library_remove(path: str = Query(...)) -> dict:
    """从素材库移除记录与对应产物；源视频文件一律保留，由用户手动管理。"""
    removed = db.remove_library(path)
    for product in services._product_candidates(path):
        try:
            product.unlink()
        except OSError:
            pass
        db.delete_variant_by_output(str(product))
    return {"removed": 1 if removed else 0}


@router.delete("/library/all")
def library_clear() -> dict:
    """清空素材库：移除全部素材记录并删除对应产物文件与记录。

    源视频文件一律保留（无论位于何处），由用户手动管理。
    """
    materials = db.list_library()
    removed_products = 0
    removed_bytes = 0
    for item in materials:
        source = item["path"]
        for product in services._product_candidates(source):
            try:
                removed_bytes += product.stat().st_size
            except OSError:
                pass
            try:
                product.unlink()
                removed_products += 1
            except OSError:
                pass
            db.delete_variant_by_output(str(product))
        db.remove_library(source)
    return {
        "removed_materials": len(materials),
        "removed_products": removed_products,
        "removed_bytes": removed_bytes,
    }


@router.get("/media")
def media(path: str = Query(...)) -> FileResponse:
    """流式返回视频文件，支持 HTTP Range，播放器可随时拖动定位。"""
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="文件不存在或已被移动")
    ext = Path(path).suffix.lower()
    media_type = _VIDEO_MEDIA_TYPES.get(ext, "application/octet-stream")
    return FileResponse(
        path,
        media_type=media_type,
        filename=Path(path).name,
        content_disposition_type="inline",
    )


@router.post("/import")
async def import_video(file: Annotated[UploadFile, File()]) -> dict:
    """接收网页端导入的视频：流式落盘，去重与入库交给服务层。"""
    filename = Path(file.filename or "").name
    ext = Path(filename).suffix.lower()
    if ext not in services.VIDEO_EXTENSIONS:
        raise HTTPException(status_code=422, detail="仅支持 MP4 / MOV / MKV / AVI / WEBM / FLV / TS 视频")
    # 清理文件名，防止路径穿越与特殊字符。
    stem = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in Path(filename).stem)
    stem = stem.strip("._") or "素材"
    # 边落盘边计算内容哈希：一次遍历完成写入与去重指纹，避免大文件二次读盘。
    tmp_fd, tmp_path = tempfile.mkstemp(prefix="cthulhu-import-", suffix=ext)
    digest = hashlib.md5()
    size = 0
    with os.fdopen(tmp_fd, "wb") as out:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            digest.update(chunk)
            out.write(chunk)
    if size == 0:
        os.unlink(tmp_path)
        raise HTTPException(status_code=422, detail="导入的文件为空")
    return services.ingest_upload(
        tmp_path,
        stem=stem,
        ext=ext,
        size=size,
        content_hash_value=digest.hexdigest(),
    )


@router.post("/audio/analyze")
def audio_analyze(request: PathRequest) -> dict:
    """音频分析：波形、对数频谱与回声隐藏置信度。"""
    # 波形取音轨开头、频谱与回声置信度均为统计口径，只解码前 120 秒，
    # 避免长视频整轨 float64 解码造成数百 MB 瞬时峰值。
    audio = ffmpeg.decode_audio(request.path, max_seconds=120)
    if audio is None:
        raise HTTPException(status_code=404, detail="该视频没有音轨")
    signal, sample_rate = audio
    step = max(1, sample_rate // 100)
    waveform = [round(float(v), 4) for v in signal[::step][:1000]]
    spectrum = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal), 1.0 / sample_rate)
    bands: list[float] = []
    edges = np.geomspace(max(float(freqs[1]), 20.0), sample_rate / 2, 33)
    for index in range(32):
        mask = (freqs >= edges[index]) & (freqs < edges[index + 1])
        bands.append(float(np.mean(spectrum[mask])))
    peak = max(bands) or 1.0
    return {
        "sample_rate": sample_rate,
        "duration": round(len(signal) / sample_rate, 2),
        "echo_score": round(watermark_detect.audio_scores(signal, sample_rate)["echo"], 4),
        "waveform": waveform,
        "spectrum": [round(band / peak, 4) for band in bands],
    }


@router.post("/desensitize")
async def desensitize(request: DesensitizeRequest) -> dict:
    await broker.publish({"type": "task:start", "task": "desensitize", "path": request.path})
    try:
        report = await asyncio.to_thread(
            services.run_desensitize,
            request.path,
            request.output,
            output_mode=request.output_mode,
            reorder=request.reorder,
            speed=request.speed,
            recrop=request.recrop,
            regrade=request.regrade,
            perturb=request.perturb,
            audio_remix=request.audio_remix,
            sharpness=request.sharpness,
            color_restore=request.color_restore,
            denoise=request.denoise,
            anti_reembed=request.anti_reembed,
            banner=request.banner,
            seed=request.seed,
            codec=request.codec,
            lossless=request.lossless,
            spoof=request.spoof,
            bitrate_kbps=request.bitrate_kbps,
            gop=request.gop,
            resolution=request.resolution,
            fps_out=request.fps_out,
            rotate=request.rotate,
            phash_attack=request.phash_attack,
            phash_epsilon=request.phash_epsilon,
            phash_iters=request.phash_iters,
            multi_hash_attack=request.multi_hash_attack,
            median=request.median,
            noise=request.noise,
            requant=request.requant,
            dct_step=request.dct_step,
            drop_every=request.drop_every,
            jitter=request.jitter,
            perspective=request.perspective,
            warp=request.warp,
            chroma_levels=request.chroma_levels,
            subtract_beta=request.subtract_beta,
            saliency=request.saliency,
            audio_strong=request.audio_strong,
            echo_defeat=request.echo_defeat,
            skip_vmaf=request.skip_vmaf,
            filter_scale=request.filter_scale,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=_friendly_detail(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=_friendly_detail(exc)) from exc
    await broker.publish({"type": "task:done", "task": "desensitize", "path": request.path})
    return report


@router.post("/repair")
async def repair(request: RepairRequest) -> dict:
    try:
        return await asyncio.to_thread(
            services.run_repair,
            request.path,
            request.output,
            [region.model_dump() for region in request.regions],
            request.crf,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=_friendly_detail(exc)) from exc


@router.post("/export")
def export_videos(request: ExportRequest) -> dict:
    """把已处理素材的产物复制到设置中的导出目录。"""
    settings = db.load_settings()
    export_dir = request.export_dir or settings.get("export_dir") or "~/Documents/Cthulhu"
    return services.export_outputs(request.paths, str(export_dir))


@router.get("/outputs")
async def outputs(path: str = Query(...)) -> dict:
    """列出源素材的全部处理产物（清洗/修复/候选），最新在前。"""
    try:
        return services.list_outputs(path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=_friendly_detail(exc)) from exc


@router.get("/outputs/counts")
def outputs_counts() -> dict:
    """素材库各源素材的产物数量汇总。"""
    return services.library_output_counts()


@router.delete("/outputs")
def delete_output(path: str = Query(...)) -> dict:
    """删除指定处理产物（仅限清洗/修复/候选命名）。"""
    try:
        removed = services.delete_output(path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not removed:
        raise HTTPException(status_code=404, detail="产物不存在")
    return {"removed": 1}


@router.get("/products")
def products_list() -> dict:
    """全部处理产物的全局清单，用于产物管理页查看占用与清理。"""
    return services.list_all_products()


@router.post("/products/delete")
def products_delete(request: ProductsRequest) -> dict:
    """批量删除产物文件与记录；仅允许清洗/修复/候选命名，绝不触碰源视频。"""
    try:
        return services.delete_products(request.paths)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/ffmpeg/select")
def select_ffmpeg(request: PathRequest) -> dict:
    """用户手动指定本机已有的 ffmpeg 可执行文件。"""
    path = Path(request.path)
    if not path.is_file() or not os.access(path, os.X_OK):
        raise HTTPException(status_code=400, detail="所选文件不存在或不可执行")
    directory = str(path.parent)
    ffmpeg.set_custom_dir(directory)
    db.save_settings({"ffmpeg_dir": directory})
    return {"ffmpeg_dir": directory, "available": ffmpeg.has_ffmpeg()}


@router.post("/ffmpeg/install", status_code=202)
def install_ffmpeg() -> dict:
    """后台下载静态视频处理引擎，前端轮询状态直到完成。"""
    target = os.environ.get("CTHULHU_FFMPEG_INSTALL_DIR") or _default_ffmpeg_install_dir()
    return services.install_ffmpeg(target)


@router.get("/ffmpeg/install")
def ffmpeg_install_status() -> dict:
    """查询视频处理引擎一键安装进度。"""
    return services.ffmpeg_install_state()


@router.post("/jobs", status_code=202)
async def create_job(request: JobRequest) -> dict:
    # 清洗任务在入队时就校验选项，拼错字段名或越界参数直接 422，
    # 避免把坏任务放进后台慢慢失败。
    for task in request.tasks:
        if task.kind == "desensitize":
            options = {
                key: value
                for key, value in task.options.items()
                if key not in {"output", "transform_strategy", "preset"}
            }
            try:
                DesensitizeOptions.model_validate(options)
            except ValidationError as exc:
                raise HTTPException(
                    status_code=422,
                    detail=f"清洗任务参数无效：{exc.errors()[0]['msg']}",
                ) from exc
    settings = db.load_settings()
    try:
        default_parallelism = int(settings.get("parallelism", 2))
    except (TypeError, ValueError):
        default_parallelism = 2
    default_parallelism = min(4, max(1, default_parallelism))
    job = job_queue.create_job(
        request.name,
        [task.model_dump() for task in request.tasks],
        request.parallelism or default_parallelism,
    )
    # 立即广播排队状态，前端无需轮询即可禁用重复提交按钮。
    await broker.publish({"type": "job:state", "job": job_queue.public_view(job)})
    return job_queue.public_view(job)


@router.get("/jobs")
async def list_jobs() -> list[dict]:
    return [job_queue.public_view(job) for job in job_queue.list()]


@router.get("/jobs/{job_id}")
async def get_job(job_id: str) -> dict:
    job = job_queue.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    return job_queue.public_view(job)


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str) -> dict:
    job = job_queue.cancel(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    return job_queue.public_view(job)


@router.post("/jobs/{job_id}/retry", status_code=202)
async def retry_job(job_id: str) -> dict:
    job = job_queue.retry(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="任务不存在或无失败项")
    await broker.publish({"type": "job:state", "job": job_queue.public_view(job)})
    return job_queue.public_view(job)


@router.post("/jobs/{job_id}/pause")
async def pause_job(job_id: str) -> dict:
    job = job_queue.pause(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    return job_queue.public_view(job)


@router.post("/jobs/{job_id}/resume")
async def resume_job(job_id: str) -> dict:
    job = job_queue.resume(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    return job_queue.public_view(job)


@router.post("/jobs/{job_id}/priority")
async def set_job_priority(job_id: str, request: PriorityRequest) -> dict:
    job = job_queue.set_priority(job_id, request.priority)
    if job is None:
        raise HTTPException(status_code=404, detail="任务不存在或不在排队状态")
    return job_queue.public_view(job)


@router.get("/settings")
def get_settings() -> dict:
    return db.load_settings()


@router.put("/settings")
def put_settings(payload: dict) -> dict:
    db.save_settings(payload)
    return db.load_settings()


@router.get("/templates")
def get_templates() -> list[dict]:
    return db.list_templates()


@router.post("/templates", status_code=201)
def post_template(request: TemplateCreate) -> dict:
    return db.create_template(request.name, request.payload.model_dump(mode="json"))


@router.put("/templates/{template_id}")
def put_template(template_id: str, request: TemplateUpdate) -> dict:
    payload = request.payload.model_dump(mode="json")
    if not db.update_template(template_id, request.name, payload):
        raise HTTPException(status_code=404, detail="模板不存在")
    return {"id": template_id, "name": request.name, "payload": payload}


@router.delete("/templates/{template_id}")
def remove_template(template_id: str) -> dict:
    if not db.delete_template(template_id):
        raise HTTPException(status_code=404, detail="模板不存在")
    return {"ok": True}


@router.get("/variants")
def list_variants(
    source: str | None = Query(None),
) -> list[dict]:
    """产物记录清单：按源文件过滤，返回完整参数与指标快照。"""
    return db.list_variants(source=source)


@router.delete("/jobs")
def remove_jobs(scope: str = Query("finished", pattern="^(finished|all)$")) -> dict:
    removed = job_queue.clear(scope)
    db.delete_jobs(scope)
    return {"removed": removed}
