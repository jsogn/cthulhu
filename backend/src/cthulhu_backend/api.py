"""GUI 后端 API：检测 / 相似度 / 内容脱敏，任务进度经事件广播推送。"""

from __future__ import annotations

import asyncio
import errno
import hashlib
import os
import subprocess
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
from fastapi import APIRouter, File, HTTPException, Query, Response, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

from cthulhu_backend import db, services
from cthulhu_backend.events import broker
from cthulhu_backend.jobs import job_queue
from cthulhu_backend.media import ffmpeg
from cthulhu_backend.watermark import detect as watermark_detect

router = APIRouter(prefix="/api")

# 网页端导入素材的落地目录：浏览器拖入的 File 没有本机路径，
# 上传到此后即可与桌面端一样走「本地路径 → 播放/检测/清洗」的完整链路。
_LIBRARY_DIR = Path(
    os.environ.get(
        "CTHULHU_LIBRARY_DIR",
        str(Path(__file__).resolve().parents[2] / "data" / "library"),
    )
)

# 封面缩略图磁盘缓存：素材多、反复刷新列表时不重复启动 ffmpeg 抽帧。
_THUMB_CACHE_DIR = Path(
    os.environ.get(
        "CTHULHU_THUMB_CACHE",
        str(Path(__file__).resolve().parents[2] / "data" / "thumb-cache"),
    )
)

_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".flv", ".ts", ".m4v"}

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


def _probe_video(path: str) -> dict | None:
    """读取视频元数据；失败返回 None（不阻断导入，仅缺失展示信息）。"""
    try:
        return ffmpeg.video_info(path)
    except Exception:  # noqa: BLE001 - 元数据失败不影响素材导入
        return None


def _library_file(record: dict) -> dict:
    """把素材库记录组装成前端可直接渲染的清单项。"""
    path = record["path"]
    meta = record.get("meta") or {}
    if not meta and os.path.isfile(path):
        meta = _probe_video(path) or {}
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


class PathRequest(BaseModel):
    path: str


class PairRequest(BaseModel):
    a: str
    b: str


class CandidatesRequest(BaseModel):
    path: str
    output_dir: str
    count: int = Field(3, ge=1, le=10)
    options: dict = {}
    base_seed: int | None = None


class ScanRequest(BaseModel):
    path: str


class DesensitizeRequest(BaseModel):
    # 兼容前端的 camelCase 字段名（audioRemix 等）与后端的 snake_case。
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    path: str
    output: str
    reorder: bool = True
    speed: float = Field(1.0, gt=0)
    recrop: float = Field(0.0, ge=0, le=0.2)
    regrade: bool = True
    perturb: float = Field(0.0, ge=0, le=1)
    audio_remix: bool = True
    sharpness: bool = True
    color_restore: bool = True
    denoise: bool = False
    anti_reembed: bool = False
    banner: str = ""
    seed: int = 0
    codec: str = "libx264"
    lossless: bool = False
    spoof: bool = False
    bitrate_kbps: int | None = Field(None, ge=100, le=100000)
    gop: int | None = Field(None, ge=1, le=600)
    resolution: str | None = None
    fps_out: float | None = Field(None, gt=0, le=240)
    rotate: float = Field(0.0, ge=0, le=10)
    phash_attack: bool = False
    phash_epsilon: float = Field(0.03, gt=0, le=1)
    phash_iters: int = Field(120, ge=1, le=1000)
    multi_hash_attack: bool = False
    median: int = Field(0, ge=0, le=9)
    noise: float = Field(0.0, ge=0, le=1)
    requant: int = Field(0, ge=0, le=256)
    dct_step: float = Field(0.0, ge=0, le=256)
    drop_every: int = Field(0, ge=0, le=1000)
    jitter: float = Field(0.0, ge=0, le=1)
    perspective: float = Field(0.0, ge=0, le=1)
    warp: float = Field(0.0, ge=0, le=1)
    mirror: bool = False
    chroma_levels: int = Field(0, ge=0, le=256)
    subtract_beta: float = Field(0.0, ge=0, le=4)
    transcode_chain: bool = False
    saliency: int = Field(0, ge=0, le=4)


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
    payload: dict = {}


class TemplateUpdate(BaseModel):
    name: str
    payload: dict = {}


class AuditCreate(BaseModel):
    name: str
    time: str
    action: str
    params: str = ""
    out: str = ""
    result: str = ""


class AuditUpdate(BaseModel):
    result: str


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
    width: int = Query(320, ge=16, le=1920),
) -> Response:
    """返回等比缩放的首帧缩略图（JPEG），带磁盘缓存，供素材封面使用。"""
    try:
        mtime = os.path.getmtime(path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=_friendly_detail(exc)) from exc
    key = hashlib.sha1(f"{os.path.abspath(path)}|{mtime}|{width}".encode()).hexdigest()
    cache_file = _THUMB_CACHE_DIR / f"{key}.jpg"
    if cache_file.is_file():
        return Response(content=cache_file.read_bytes(), media_type="image/jpeg")
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


@router.get("/library")
def library_list() -> dict:
    """返回持久化素材库（最新导入在前），应用重启后仍可恢复。"""
    files = [_library_file(record) for record in db.list_library()]
    return {"files": files, "valid": sum(1 for item in files if item["video"]), "invalid": 0}


@router.post("/library")
def library_add(request: PathRequest) -> dict:
    """登记一个本地视频到素材库，重启后保持显示。"""
    path = os.path.abspath(request.path)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="文件不存在或已被移动")
    record = {
        "path": path,
        "name": Path(path).name,
        "size": os.path.getsize(path),
        "meta": _probe_video(path) or {},
    }
    db.add_library(record["path"], record["name"], record["size"], record["meta"])
    return _library_file(record)


@router.delete("/library")
def library_remove(path: str = Query(...)) -> dict:
    """从素材库移除记录；网页上传的库内副本同时删除，桌面源文件不受影响。"""
    removed = db.remove_library(path)
    try:
        target = Path(path).resolve()
        target.relative_to(_LIBRARY_DIR.resolve())
        if target.is_file():
            target.unlink()
    except ValueError:
        pass
    return {"removed": 1 if removed else 0}


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


@router.post("/upload")
async def upload_video(file: Annotated[UploadFile, File()]) -> dict:
    """接收网页端导入的视频，写入本地素材库后返回可播放/可处理的路径。"""
    filename = Path(file.filename or "").name
    ext = Path(filename).suffix.lower()
    if ext not in _VIDEO_EXTENSIONS:
        raise HTTPException(status_code=422, detail="仅支持 MP4 / MOV / MKV / AVI / WEBM / FLV / TS 视频")
    # 清理文件名，防止路径穿越与特殊字符，并为同名文件追加序号。
    stem = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in Path(filename).stem)
    stem = stem.strip("._") or "素材"
    _LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    target = _LIBRARY_DIR / f"{stem}{ext}"
    counter = 1
    while target.exists():
        target = _LIBRARY_DIR / f"{stem}_{counter}{ext}"
        counter += 1
    size = 0
    with target.open("wb") as out:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            out.write(chunk)
    if size == 0:
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail="上传的文件为空")
    meta = _probe_video(str(target)) or {}
    db.add_library(str(target), target.name, size, meta)
    return {"path": str(target), "name": target.name, "size": size, "video": meta or None}


@router.post("/audio/analyze")
def audio_analyze(request: PathRequest) -> dict:
    """音频分析：波形、对数频谱与回声隐藏置信度。"""
    audio = ffmpeg.decode_audio(request.path)
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
            mirror=request.mirror,
            chroma_levels=request.chroma_levels,
            subtract_beta=request.subtract_beta,
            transcode_chain=request.transcode_chain,
            saliency=request.saliency,
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
    export_dir = request.export_dir or settings.get("export_dir") or "~/导出/暗水印清洗"
    return services.export_outputs(request.paths, str(export_dir))


@router.post("/candidates")
async def candidates(request: CandidatesRequest) -> dict:
    """同一素材生成多个差异化候选并按低损优选评分排序。"""
    try:
        return await asyncio.to_thread(
            services.generate_candidates,
            request.path,
            request.output_dir,
            request.count,
            request.options,
            base_seed=request.base_seed,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=_friendly_detail(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=_friendly_detail(exc)) from exc


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
    target = os.environ.get("CTHULHU_FFMPEG_INSTALL_DIR") or str(Path.home() / ".cthulhu" / "ffmpeg")
    return services.install_ffmpeg(target)


@router.get("/ffmpeg/install")
def ffmpeg_install_status() -> dict:
    """查询视频处理引擎一键安装进度。"""
    return services.ffmpeg_install_state()


@router.post("/jobs", status_code=202)
async def create_job(request: JobRequest) -> dict:
    settings = db.load_settings()
    try:
        default_parallelism = int(settings.get("parallelism", 2))
    except (TypeError, ValueError):
        default_parallelism = 2
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
    return db.create_template(request.name, request.payload)


@router.put("/templates/{template_id}")
def put_template(template_id: str, request: TemplateUpdate) -> dict:
    if not db.update_template(template_id, request.name, request.payload):
        raise HTTPException(status_code=404, detail="模板不存在")
    return {"id": template_id, "name": request.name, "payload": request.payload}


@router.delete("/templates/{template_id}")
def remove_template(template_id: str) -> dict:
    if not db.delete_template(template_id):
        raise HTTPException(status_code=404, detail="模板不存在")
    return {"ok": True}


@router.get("/variants")
def list_variants(
    source: str | None = Query(None),
    batch: str | None = Query(None),
) -> list[dict]:
    """产物记录清单：按源文件/批次过滤，返回完整参数与指标快照。"""
    return db.list_variants(source=source, batch=batch)


@router.get("/audit")
def get_audit() -> list[dict]:
    return db.list_audit()


@router.post("/audit", status_code=201)
def post_audit(request: AuditCreate) -> dict:
    return db.append_audit(request.model_dump())


@router.put("/audit/{audit_id}")
def put_audit(audit_id: int, request: AuditUpdate) -> dict:
    if not db.update_audit(audit_id, request.result):
        raise HTTPException(status_code=404, detail="审计记录不存在")
    return {"id": audit_id, "result": request.result}


@router.delete("/audit")
def remove_audit() -> dict:
    return {"removed": db.clear_audit()}


@router.delete("/jobs")
def remove_jobs(scope: str = Query("finished", pattern="^(finished|all)$")) -> dict:
    removed = job_queue.clear(scope)
    db.delete_jobs(scope)
    return {"removed": removed}
