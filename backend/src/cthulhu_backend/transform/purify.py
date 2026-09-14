"""潜空间净化原语：TAESD 瓶颈重建，攻击生成式分布外的低频盲水印。

机理（PUBLIC_SCHEME_REPORT.md / research §19）：公开深度水印
（TrustMark/MBRS/WAM/VideoSeal/PixelSeal）写在低频，纯降采样带不走它
（BA≈1.0），真正起作用的是**潜空间瓶颈下的学习式重建**——重建结果由解码器
先验主导，叠加式水印无法被复现。因此用 10MB 的 TAESD（8 倍 AE）替代原先
2.4GB 的 sd-turbo：清除率持平、保真度更高、快 50 倍（research §19.3）。

开箱即用（A 方案）：
- 运行时依赖（torch/diffusers 等）由 `backend[purify]` 安装并进入桌面端打包，
  不需要用户手工装 Python 包；
- 权重随安装包内置（`packaging/models/madebyollin--taesd`，9.3MB），安装后
  零下载；模型目录默认 `CTHULHU_PURIFY_MODEL_DIR` 或打包内置目录，已存在的
  HF 缓存/应用数据目录仅作为开发环境兜底。
- 开发环境没有内置权重时才允许按需下载（`CTHULHU_PURIFY_ALLOW_DOWNLOAD=0`
  可彻底关闭）。

推理用全局锁串行（并发任务共用一个模型时避免线程安全与显存叠加）；取消/暂停
与进度通过线程本地控制面传入（`set_control`），不进入可序列化的变换上下文，
因此不会破坏多进程武器池的 pickle。逐帧推理按小批滚动、MPS 逐批清缓存，
内存与视频长度解耦。
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
from PIL import Image

from cthulhu_backend.transform import temporal
from cthulhu_backend.transform.postprocess import (
    POST_SHARPEN_AMOUNT,
    POST_SHARPEN_SIGMA,
    reinject_detail,
    unsharp_batch,
)

TAESD_MODEL_ID = os.environ.get("CTHULHU_PURIFY_TAESD_MODEL", "madebyollin/taesd")
ALLOW_DOWNLOAD_ENV = "CTHULHU_PURIFY_ALLOW_DOWNLOAD"
MODEL_DIR_ENV = "CTHULHU_PURIFY_MODEL_DIR"
# 细节回注带宽（σ）与后置锐化：水印在低频、字幕/纹理在中频，σ 决定回注多宽。
# σ 是**像素单位**，而水印的尺度随分辨率缩放——1080p 用 1.5 会糊成一片，用
# 2.5 才等价于 512p 上的 1.5（research §19.5）。因此默认按长边自动换算：
#   σ_auto = clip(短边 × 0.0024, 1.2, 3.0)
# 重建本身是低分辨率瓶颈，观感偏软，末尾再做一次轻度锐化把边缘拉回来。
DETAIL_SIGMA_RATIO = 0.0024
DETAIL_SIGMA_MIN = 1.2
DETAIL_SIGMA_MAX = 3.0
# B 档（画质优先）：字幕笔画在 1080p 约 4~6px、512p 约 2~3px，安全带宽（上面的
# 0.0024）救不回它们，必须把带宽推到笔画尺度——代价是水印部分回流（BA 0.58~0.62）。
WIDE_DETAIL_SIGMA_RATIO = 0.006
WIDE_DETAIL_SIGMA_MIN = 2.5
WIDE_DETAIL_SIGMA_MAX = 8.0
# 字幕区域内的带宽倍数：笔画在 1080p 约 4~6px，需要比安全带宽宽数倍才能捞回。
SUBTITLE_SIGMA_FACTOR = 3.0
# 字幕掩膜：阈值越高只留笔画、膨胀越小越不碰背景边缘。实测收紧掩膜（70/1/3.0）
# 并不能减少水印回流（BA 0.7695 vs 0.7383，噪声内），反而少覆盖笔画，因此保留
# 偏宽的设置以保证字幕可辨（research §19.6）。
SUBTITLE_MASK_THRESHOLD = 40
SUBTITLE_MASK_DILATE_ITER = 3
SUBTITLE_MASK_SOFT_SIGMA = 6.0
SUBTITLE_MASK_GAIN = 1.5
DEFAULT_DETAIL_SIGMA = 0.0  # 0 = 自动（按帧长边换算）

# 生成式档（A，扩散 img2img）尚未实现：参数与验收判据在设计文档 §6，
# 落地时按那份形态重新引入权重与档位，不在这里预留未消费的常量。

logger = logging.getLogger(__name__)

_LOAD_LOCK = threading.Lock()
_INFER_LOCK = threading.Lock()
_INSTALL_LOCK = threading.Lock()
_LOCAL = threading.local()
_TAESD = None
_TAESD_FAILED: tuple[str, bool] | None = None

_STATUS_LOCK = threading.RLock()
_STATUS: dict = {
    "state": "unknown",
    "progress": 0.0,
    "error": None,
}


# ---------------------------------------------------------------------------
# 能力/状态探测
# ---------------------------------------------------------------------------


def check_available() -> tuple[bool, str]:
    """依赖探测（只查 spec、不导入重型包），成功返回 (True, "")。"""
    for name in (
        "torch",
        "diffusers",
        "transformers",
        "accelerate",
        "safetensors",
        "huggingface_hub",
        "tqdm",
    ):
        if importlib.util.find_spec(name) is None:
            return False, f"missing dependency: {name}"
    return True, ""


def allow_download() -> bool:
    """是否允许首次下载权重；A 方案默认允许，显式设 0 才关闭。"""
    raw = os.environ.get(ALLOW_DOWNLOAD_ENV)
    if raw is None or not raw.strip():
        return True
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _repo_bundled_model_dir() -> Path | None:
    """开发环境仓库内打包产物目录（packaging/fetch_purify_model.py 生成）。"""
    candidate = Path(__file__).resolve().parents[4] / "packaging" / "models"
    return candidate if candidate.is_dir() else None


def frozen_bundled_model_dir() -> Path | None:
    """PyInstaller one-folder：datas 解包到 sys._MEIPASS/models。"""
    if not getattr(sys, "frozen", False):
        return None
    base = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    candidate = base / "models"
    return candidate if candidate.is_dir() else None


def bundled_model_dir() -> Path | None:
    """安装包内置模型目录；开发环境回退到仓库 packaging/models。"""
    for candidate in (frozen_bundled_model_dir(), _repo_bundled_model_dir()):
        if candidate is not None:
            return candidate
    return None


def model_dir() -> Path:
    """模型目录：环境变量 → 内置模型目录 → 平台应用数据目录（开发兜底）。"""
    raw = os.environ.get(MODEL_DIR_ENV)
    if raw:
        return Path(raw).expanduser()
    bundled = bundled_model_dir()
    if bundled is not None and bundled_taesd_path() is not None:
        return bundled
    return _platform_model_dir()


def _platform_model_dir() -> Path:
    """可写的平台应用数据目录（内置目录只读时的下载兜底）。"""
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "Cthulhu" / "models"
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home())
        return base / "Cthulhu" / "models"
    base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    return base / "cthulhu" / "models"


def taesd_dir_name() -> str:
    return TAESD_MODEL_ID.replace("/", "--")


def local_taesd_path() -> Path:
    return model_dir() / taesd_dir_name()


def taesd_ready(path: Path | None) -> bool:
    """TAESD 目录是否可直接加载（diffusers 单体 AE 只需 config + 权重）。"""
    if path is None or not path.is_dir():
        return False
    config = path / "config.json"
    weights = list(path.glob("*.safetensors"))
    return (
        config.is_file()
        and config.stat().st_size > 0
        and any(item.stat().st_size > 0 for item in weights)
    )


def bundled_taesd_path() -> Path | None:
    """安装包内置的 TAESD 目录（packaging/models/madebyollin--taesd）。"""
    base = bundled_model_dir()
    if base is None:
        return None
    candidate = base / taesd_dir_name()
    return candidate if taesd_ready(candidate) else None


def _hf_taesd_snapshot() -> Path | None:
    try:
        from huggingface_hub import scan_cache_dir
    except ImportError:
        return None
    try:
        cache = scan_cache_dir()
    except Exception:  # noqa: BLE001 - 缓存目录不可读按未命中处理
        return None
    for repo in cache.repos:
        if repo.repo_id != TAESD_MODEL_ID:
            continue
        for revision in repo.revisions:
            path = Path(revision.snapshot_path)
            if taesd_ready(path):
                return path
    return None


def taesd_cached() -> bool:
    """潜空间引擎的权重是否就绪（内置 → 模型目录 → 应用数据目录 → HF 缓存）。"""
    return _taesd_lookup() is not None


def _platform_taesd_path() -> Path:
    return _platform_model_dir() / taesd_dir_name()


def _taesd_lookup() -> Path | None:
    """按优先级返回可用目录：内置 → 模型目录 → 应用数据目录 → HF 缓存。"""
    bundled = bundled_taesd_path()
    if bundled is not None:
        return bundled
    for candidate in (local_taesd_path(), _platform_taesd_path()):
        if taesd_ready(candidate):
            return candidate
    return _hf_taesd_snapshot()


DEVICE_ENV = "CTHULHU_PURIFY_DEVICE"
DEVICE_SETTING = "purify_device"
_DEVICES = ("cpu", "mps", "cuda", "auto")


def device_preference() -> str:
    """推理设备偏好：环境变量 > 设置项 > auto（有 GPU 就用，实测快约 3.6 倍）。"""
    override = (os.environ.get(DEVICE_ENV) or "").strip().lower()
    if override in _DEVICES:
        return override
    try:
        from cthulhu_backend import db

        stored = str(db.load_settings().get(DEVICE_SETTING) or "").strip().lower()
    except Exception:  # noqa: BLE001 - 设置读不到时退回默认
        stored = ""
    return stored if stored in _DEVICES else "auto"


def set_device_preference(device: str) -> None:
    """落盘设备偏好并卸载已加载的模型，下一次推理按新设备重建。

    现场反馈（2026-09-14）：MPS 路径会在长片清洗中途触发
    `objc: Cannot form weak reference to instance of class MPSGraph` 直接 SIGABRT，
    引擎被守护进程拉起后又崩，于是整批任务反复中断。稳定性优先，异常退出后
    自动落到 CPU 续跑。
    """
    from cthulhu_backend import db

    db.save_settings({DEVICE_SETTING: device})
    reset_model()


def reset_model() -> None:
    """丢弃已加载的模型缓存，下次推理重新按当前偏好加载。"""
    global _TAESD, _TAESD_FAILED
    with _LOAD_LOCK:
        _TAESD = None
        _TAESD_FAILED = None


def _device_and_dtype() -> tuple[str, object]:
    import torch

    preference = device_preference()
    available = {"cpu": True, "mps": torch.backends.mps.is_available(), "cuda": torch.cuda.is_available()}
    if preference == "cpu":
        return "cpu", torch.float32
    if preference in {"mps", "cuda"}:
        # 显式指定但本机没有该设备时退回 CPU，而不是让任务直接失败。
        return (preference, torch.float16) if available[preference] else ("cpu", torch.float32)
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    if torch.cuda.is_available():
        return "cuda", torch.float16
    return "cpu", torch.float32


def preflight() -> tuple[bool, str]:
    """任务启动前的可用性检查，返回 (可用, 原因)。

    原因 "ready" 表示模型已就绪；"will_download" 表示首次使用会下载权重，
    管线据此在结果里给出明确提示。
    """
    ok, reason = check_available()
    if not ok:
        return False, reason
    # 只需 10MB 的 TAESD：内置即就绪，缺失时按需下载。
    if taesd_cached():
        return True, "ready"
    if not allow_download():
        return False, "TAESD 未安装且已禁用下载"
    return True, "will_download"


def _set_status(**fields) -> None:
    with _STATUS_LOCK:
        _STATUS.update(fields)


def model_status() -> dict:
    """给 API/UI 的状态：依赖、10MB TAESD 权重、设备与下载策略。"""
    ok, reason = check_available()
    with _STATUS_LOCK:
        status = dict(_STATUS)
    cached = taesd_cached() if ok else False
    state = status["state"]
    if not ok:
        state = "unavailable"
    elif status["state"] == "downloading":
        pass
    elif cached:
        state = "ready"
        status["progress"] = 1.0
        status["error"] = None
    elif status["state"] != "error":
        state = "missing"
    device = "unavailable"
    if ok:
        try:
            device = _device_and_dtype()[0]
        except Exception:  # noqa: BLE001 - 设备探测失败不影响状态返回
            device = "unknown"
    return {
        "available": ok,
        "reason": reason,
        "model_id": TAESD_MODEL_ID,
        "model_cached": cached,
        "bundled": bundled_taesd_path() is not None,
        "allow_download": allow_download(),
        "device": device,
        "state": state,
        "progress": round(float(status["progress"]), 4),
        "error": status["error"],
        "model_dir": str(model_dir()),
        "latent": {
            "model_id": TAESD_MODEL_ID,
            "cached": cached,
            "bundled": bundled_taesd_path() is not None,
            "path": str(bundled_taesd_path() or local_taesd_path()),
        },
    }


# ---------------------------------------------------------------------------
# 取消/暂停/进度控制面（线程本地，不进入可序列化状态）
# ---------------------------------------------------------------------------


def set_control(
    *,
    should_stop: Callable[[], bool] | None = None,
    pause=None,
    progress: Callable[[float, str], None] | None = None,
) -> None:
    _LOCAL.control = {"should_stop": should_stop, "pause": pause, "progress": progress}


def clear_control() -> None:
    _LOCAL.control = None


def _control() -> dict:
    return getattr(_LOCAL, "control", None) or {}


def last_failure() -> str | None:
    """最近一次加载/推理失败原因；用于在任务结果里回写净化降级说明。"""
    return getattr(_LOCAL, "failure", None)


def reset_failure() -> None:
    """任务开始时清空上次失败，避免跨任务污染 purify_note。"""
    _LOCAL.failure = None


def _record_failure(reason: str) -> None:
    _LOCAL.failure = reason


def _check_control(
    should_stop: Callable[[], bool] | None,
    pause,
) -> None:
    """逐帧取消/暂停检查：取消抛 InterruptedError，暂停原地等待。"""
    if should_stop is not None and should_stop():
        raise InterruptedError("任务已取消")
    if pause is None:
        return
    while pause.is_set():
        if should_stop is not None and should_stop():
            raise InterruptedError("任务已取消")
        time.sleep(0.2)


# ---------------------------------------------------------------------------
# 下载与加载
# ---------------------------------------------------------------------------


def install_model(*, wait: bool = False) -> dict:
    """安装/下载 TAESD 权重（10MB）；wait=False 后台执行供 API 轮询。

    内置权重随安装包分发，正常情况下这里只是把内置目录判定为 ready。
    """
    ok, reason = check_available()
    if not ok:
        _set_status(state="unavailable", error=reason)
        return model_status()
    if taesd_cached():
        _set_status(state="ready", progress=1.0, error=None)
        return model_status()
    if not allow_download():
        _set_status(state="missing", error="TAESD 未安装且已禁用下载")
        return model_status()
    with _INSTALL_LOCK:
        with _STATUS_LOCK:
            already = _STATUS["state"] == "downloading"
        if not already:
            _set_status(state="downloading", progress=0.0, error=None)
            if wait:
                _install_worker()
            else:
                threading.Thread(
                    target=_install_worker,
                    name="cthulhu-purify-install",
                    daemon=True,
                ).start()
    return model_status()


def _ensure_taesd() -> str:
    """确保 TAESD 权重可用（10MB 级），返回可交给 from_pretrained 的路径。"""
    found = _taesd_lookup()
    if found is not None:
        return str(found)
    if not allow_download():
        raise RuntimeError(
            f"TAESD 未安装（内置缺失）且已禁用下载；设 {ALLOW_DOWNLOAD_ENV}=1 允许下载"
        )
    from huggingface_hub import snapshot_download

    # 内置目录可能位于只读的安装包内：优先写模型目录，失败再退到应用数据目录。
    target = local_taesd_path()
    try:
        target.mkdir(parents=True, exist_ok=True)
        probe = target / ".write-probe"
        probe.write_bytes(b"")
        probe.unlink()
    except OSError:
        target = _platform_taesd_path()
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        TAESD_MODEL_ID,
        local_dir=str(target),
        max_workers=4,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.md"],
    )
    if not taesd_ready(target):
        raise RuntimeError("TAESD 权重下载完成但校验未通过")
    return str(target)


def _load_taesd():
    """懒加载 TAESD（潜空间引擎）；跨任务复用，失败原因保留。"""
    global _TAESD, _TAESD_FAILED
    if _TAESD is not None:
        return _TAESD
    with _LOAD_LOCK:
        if _TAESD is not None:
            return _TAESD
        if _TAESD_FAILED is not None:
            raise RuntimeError(_TAESD_FAILED[0])
        try:
            from diffusers import AutoencoderTiny

            device, dtype = _device_and_dtype()
            model_ref = _ensure_taesd()
            model = AutoencoderTiny.from_pretrained(
                model_ref,
                torch_dtype=dtype,
                local_files_only=True,
            )
            model = model.to(device).eval()
        except Exception as exc:
            _TAESD_FAILED = (str(exc), allow_download())
            raise
        _TAESD = model
        _TAESD_FAILED = None
        return _TAESD


def _install_worker() -> None:
    try:
        _ensure_taesd()
        _set_status(state="ready", progress=1.0, error=None)
    except Exception as exc:  # noqa: BLE001 - 状态里保留失败原因，供 API/UI 展示
        logger.warning("净化模型安装失败：%s", exc)
        _set_status(state="error", error=str(exc))


def _clear_cache() -> None:
    """MPS 上逐批回收驱动缓存，避免长片任务内存持续爬升。"""
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception as exc:  # noqa: BLE001 - 缓存回收失败不影响任务
        logger.debug("MPS 缓存清理失败：%s", exc)


def _to_u8(frame: np.ndarray, is_u8: bool) -> np.ndarray:
    if is_u8:
        return frame
    return (np.clip(frame, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def _subtitle_weight(batch: np.ndarray) -> np.ndarray | None:
    """字幕掩膜：逐帧笔画检测 + 时序持久性（只有跨帧不动的笔画才留下）。

    烧录字幕细、高对比、跨帧位置不变；画面内容在动，因此对多帧的笔画掩膜取
    逐像素最小即可滤掉运动边缘。没有字幕的视频返回 None，这一步等于不生效。
    """
    try:
        import cv2
    except ImportError:
        return None
    masks: list[np.ndarray] = []
    for frame in batch:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
        _, mask = cv2.threshold(grad, SUBTITLE_MASK_THRESHOLD, 255, cv2.THRESH_BINARY)
        masks.append(mask)
    keep = np.minimum.reduce(masks)
    if not keep.any():
        return None
    keep = cv2.dilate(
        keep, np.ones((3, 3), np.uint8), iterations=SUBTITLE_MASK_DILATE_ITER
    )
    soft = cv2.GaussianBlur(
        keep.astype(np.float32) / 255.0, (0, 0), SUBTITLE_MASK_SOFT_SIGMA
    )
    return np.clip(soft * SUBTITLE_MASK_GAIN, 0.0, 1.0)


def auto_detail_sigma(
    width: int, height: int, requested: float, wide: bool = False
) -> float:
    """把 σ 折算成当前分辨率下的等效值。

    σ 是像素单位，而水印的尺度随分辨率等比缩放：固定 1.5 在 512p 上刚好，
    在 1080p 上只回注了极细的纹理，画面会糊成一片（research §19.5）。
    requested<=0 时按**短边**自动换算（水印尺度随画面短边缩放）：
    512p→1.2、1080p→2.6、上限 3.0（实测 3.5 起部分方案水印回流）。

    wide=True 是 B 档「画质优先」：带宽推到字幕笔画尺度（1080p≈6.5、512p≈3.1），
    字幕可读、人脸仍偏软，代价是水印部分回流。
    """
    if requested > 0:
        return float(requested)
    short = min(width, height)
    if wide:
        scaled = short * WIDE_DETAIL_SIGMA_RATIO
        return float(min(WIDE_DETAIL_SIGMA_MAX, max(WIDE_DETAIL_SIGMA_MIN, scaled)))
    scaled = short * DETAIL_SIGMA_RATIO
    return float(min(DETAIL_SIGMA_MAX, max(DETAIL_SIGMA_MIN, scaled)))


def _latent_size(width: int, height: int, max_edge: int) -> tuple[int, int]:
    """潜空间引擎的目标尺寸：长边限制 + 8 的倍数（VAE/TAESD 下采样要求）。"""
    if max_edge <= 0:
        return width, height
    longest = max(width, height)
    if longest <= max_edge:
        return width, height
    scale = max_edge / longest
    return (
        max(8, round(width * scale / 8.0) * 8),
        max(8, round(height * scale / 8.0) * 8),
    )


def _resize_batch(frames: np.ndarray, size: tuple[int, int], *, down: bool) -> np.ndarray:
    """整批缩放：cv2 可用时走 cv2（快一个量级），否则回退 PIL。"""
    width, height = size
    if frames.shape[2] == width and frames.shape[1] == height:
        return frames
    try:
        import cv2
    except ImportError:
        cv2 = None
    if cv2 is not None:
        interp = cv2.INTER_AREA if down else cv2.INTER_CUBIC
        return np.stack(
            [cv2.resize(frame, (width, height), interpolation=interp) for frame in frames]
        )
    resample = Image.LANCZOS if down else Image.BICUBIC
    return np.stack(
        [np.asarray(Image.fromarray(frame).resize((width, height), resample)) for frame in frames]
    )


def _purify_latent(
    frames: np.ndarray,
    *,
    max_edge: int,
    batch: int,
    detail: float,
    detail_sigma: float,
    detail_wide: bool = False,
    should_stop,
    pause,
    progress,
    seed: int,
) -> np.ndarray:
    """潜空间瓶颈净化：降采样 → TAESD 编码/解码 → 放大 → 高频回注。

    机理（研究 §19）：公开深度水印写在低频，纯缩放带不走它（BA≈1.0）；
    真正起作用的是 8 倍潜空间瓶颈下的学习式重建——重建结果由解码器先验
    主导，叠加式水印无法被复现。相比原扩散方案省掉了 UNet 与重 VAE，
    720p 实测端到端 0.027s/帧（约 0.8× 视频时长）。
    """
    model = _load_taesd()
    import torch

    gray = frames.ndim == 3
    total = len(frames)
    batch = max(1, int(batch))
    height, width = frames.shape[1], frames.shape[2]
    target_w, target_h = _latent_size(width, height, max_edge)
    resized = (target_w, target_h) != (width, height)
    is_u8 = frames.dtype == np.uint8
    device = model.device if hasattr(model, "device") else "cpu"
    dtype = next(model.parameters()).dtype
    sigma_eff = auto_detail_sigma(width, height, detail_sigma, wide=detail_wide)

    out = np.empty((total, height, width) if gray else (total, height, width, 3), dtype=np.uint8)
    for start in range(0, total, batch):
        _check_control(should_stop, pause)
        stop = min(start + batch, total)
        prepared = np.stack(
            [
                _to_u8(frames[index], is_u8)
                if not gray
                else np.repeat(_to_u8(frames[index], is_u8)[..., None], 3, axis=-1)
                for index in range(start, stop)
            ]
        )
        small = _resize_batch(prepared, (target_w, target_h), down=True) if resized else prepared
        tensor = (
            torch.from_numpy(np.ascontiguousarray(small))
            .permute(0, 3, 1, 2)
            .to(device=device, dtype=dtype)
            / 255.0
        )
        with _INFER_LOCK, torch.inference_mode():
            latents = model.encode(tensor).latents
            decoded = model.decode(latents).sample
        decoded_u8 = (
            (decoded.clamp(0.0, 1.0).permute(0, 2, 3, 1).float().cpu().numpy() * 255.0)
            .round()
            .astype(np.uint8)
        )
        restored = (
            _resize_batch(decoded_u8, (width, height), down=False) if resized else decoded_u8
        )
        # 画质优先档（detail_wide）：在字幕区域内把回注带宽放宽到笔画尺度。
        sub_weight = _subtitle_weight(prepared) if (detail_wide and detail > 0) else None
        wide_sigma = sigma_eff * SUBTITLE_SIGMA_FACTOR if sub_weight is not None else None
        for offset, index in enumerate(range(start, stop)):
            arr = restored[offset].astype(np.float32) / 255.0
            if gray:
                arr = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
            if detail > 0:
                arr = reinject_detail(
                    arr,
                    frames[index],
                    strength=detail,
                    sigma=sigma_eff,
                    is_u8=is_u8,
                    gray=gray,
                    wide_sigma=wide_sigma,
                    weight=sub_weight,
                )
            scaled = np.multiply(arr, 255.0, out=arr if arr.flags.writeable else None)
            np.rint(scaled, out=scaled)
            out[index] = scaled.astype(np.uint8)
            if progress is not None:
                progress((index + 1) / total, "潜空间净化")
        _clear_cache()
    return unsharp_batch(out, POST_SHARPEN_AMOUNT, POST_SHARPEN_SIGMA)


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


def purify_frames(
    frames: np.ndarray,
    *,
    strength: float = 0.25,
    seed: int = 0,
    max_edge: int = 0,
    batch: int = 1,
    detail: float = 0.0,
    detail_sigma: float = DEFAULT_DETAIL_SIGMA,
    detail_wide: bool = False,
    temporal_strength: float = 0.0,
    temporal_mode: str = "luma",
    should_stop: Callable[[], bool] | None = None,
    pause=None,
    progress: Callable[[float, str], None] | None = None,
) -> np.ndarray:
    """逐帧净化；依赖/模型缺失或推理失败时返回原帧（任务回退继续）。

    走 TAESD 潜空间瓶颈重建：max_edge>0 时先降采样再编码/解码，最后放回原
    尺寸；detail>0 时回注原帧高频（σ=detail_sigma 决定带宽，见 §19.3），
    把攻击强度与画质损失解耦；temporal_strength>0 时先用黑盒时序估计削弱
    跨帧稳定低频分量。strength 现在只作为开关与旧参数兼容位（≤0 直接返回
    原帧）。两条增强路径都只改变净化层，不触碰白盒 decoder/权重边界。
    """
    input_frames = frames
    if strength <= 0 or len(frames) == 0:
        return input_frames
    control = _control()
    if should_stop is None:
        should_stop = control.get("should_stop")
    if pause is None:
        pause = control.get("pause")
    if progress is None:
        progress = control.get("progress")
    ok, reason = check_available()
    if not ok:
        _record_failure(reason)
        return input_frames
    # 只用 10MB 的 TAESD：内置即就绪，缺失且禁用下载时才回退原帧。
    if _TAESD is None and not allow_download() and not taesd_cached():
        _record_failure("TAESD 未安装且已禁用下载")
        return input_frames
    work_frames = frames
    if temporal_strength > 0 and len(frames) >= 4:
        try:
            # 水印是低频分量，估计在 720p/16 帧以内即可；避免 4K 长片在
            # 重建之外再叠一份全分辨率时序中值的内存峰值。
            estimate = temporal.estimate(
                frames,
                mode=temporal_mode,
                max_frames=16,
                max_edge=720,
            )
            work_frames = temporal.subtract(
                frames, estimate, temporal_strength, mode=temporal_mode
            )
        except Exception as exc:  # noqa: BLE001 - 时序估计失败不阻断净化
            logger.warning("时序残差估计失败，跳过该增强：%s", exc)
            work_frames = frames
    try:
        return _purify_latent(
            work_frames,
            max_edge=max_edge,
            batch=batch,
            detail=detail,
            detail_sigma=detail_sigma,
            detail_wide=detail_wide,
            should_stop=should_stop,
            pause=pause,
            progress=progress,
            seed=seed,
        )
    except InterruptedError:
        raise
    except Exception as exc:  # noqa: BLE001 - 净化失败回退原帧，任务不中断
        _record_failure(str(exc))
        return input_frames
