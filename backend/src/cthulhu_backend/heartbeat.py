"""引擎心跳：独立线程写盘，供 Electron 主进程区分「引擎在忙」与「引擎已死」。

为什么不靠 /api/health：清洗一条视频时事件循环可能被长任务、内存压力拖住几十秒，
HTTP 探针必然失败，但引擎其实在正常干活。误判的代价是把几十分钟的任务掐掉重跑，
所以守护进程需要一条与事件循环解耦的信号——心跳线程只做「写个小 JSON」，
numpy/torch 的大块运算会释放 GIL，因此循环卡住时它照样能反映真实状态。

未配置 `CTHULHU_ENGINE_HEARTBEAT` 时完全不启动（测试与纯后端运行不受影响）。
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

HEARTBEAT_ENV = "CTHULHU_ENGINE_HEARTBEAT"
RESTART_CAUSE_ENV = "CTHULHU_RESTART_CAUSE"
RESTART_ABNORMAL_ENV = "CTHULHU_RESTART_ABNORMAL"
_INTERVAL_SECONDS = 5.0

_lock = threading.Lock()
_active_tasks = 0
_progress_at = time.time()
_thread: threading.Thread | None = None


def refresh(active_tasks: int) -> None:
    """更新在跑任务数；归零时把「最近进展」顺延，空闲不算停滞。"""
    global _active_tasks, _progress_at
    with _lock:
        _active_tasks = max(0, int(active_tasks))
        if _active_tasks == 0:
            _progress_at = time.time()


def note_progress() -> None:
    """任务有实质进展（分块完成、阶段切换）时调用。"""
    global _progress_at
    with _lock:
        _progress_at = time.time()


def snapshot() -> dict:
    with _lock:
        return {
            "pid": os.getpid(),
            "alive_at": time.time(),
            "active_tasks": _active_tasks,
            "progress_at": _progress_at,
        }


def restart_cause() -> str:
    """本次启动的「上次重启原因」，由 Electron 主进程注入；没有则为空串。"""
    return (os.environ.get(RESTART_CAUSE_ENV) or "").strip()


def restart_abnormal() -> bool:
    """上次是不是「异常退出」（信号或非零退出码），而不是被守护进程主动重启。

    异常退出多半来自原生崩溃：现场实测 MPS 推理会在长片清洗中途 SIGABRT，
    任务队列据此把推理设备降级到 CPU，避免整批任务在同一个坑里反复中断；
    用户主动退出 App 属于正常退出，不触发降级。
    """
    return (os.environ.get(RESTART_ABNORMAL_ENV) or "").strip() == "1"


def start() -> None:
    """启动写盘线程（幂等）；未配置路径时什么都不做。"""
    global _thread
    raw = os.environ.get(HEARTBEAT_ENV)
    if not raw:
        return
    if _thread is not None and _thread.is_alive():
        return
    _thread = threading.Thread(
        target=_loop,
        args=(Path(raw),),
        daemon=True,
        name="engine-heartbeat",
    )
    _thread.start()


def _loop(path: Path) -> None:
    # 先写临时文件再原子替换，读方不会读到写了一半的 JSON。
    tmp = path.with_name(f"{path.name}.tmp")
    while True:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(snapshot()), encoding="utf-8")
            tmp.replace(path)
        except OSError:
            # 心跳写不进去只影响守护判断，绝不能让线程把引擎带崩。
            pass
        time.sleep(_INTERVAL_SECONDS)
