"""暗水印清洗台后端入口。

开发模式：Electron 以 sidecar 方式拉起 uvicorn，Vite 将 /api 与 /ws 代理到本服务。
生产模式：本服务额外托管前端构建产物（ui/dist），Electron 窗口直接加载本服务地址。
"""

from __future__ import annotations

import asyncio
import faulthandler
import os
import platform
import secrets
import threading
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from cthulhu_backend import db
from cthulhu_backend.api import router as api_router
from cthulhu_backend.api import sweep_thumb_cache
from cthulhu_backend.events import broker
from cthulhu_backend.jobs import job_queue
from cthulhu_backend.version import APP_VERSION


def _host_info() -> dict:
    """采集本机基础配置，供界面底部状态栏作为参考信息展示。"""
    info = {
        "os": platform.system(),
        "arch": platform.machine(),
        "cpu_count": os.cpu_count() or 0,
        "memory_gb": None,
    }
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
        info["memory_gb"] = round(page_size * page_count / (1024**3), 1)
    except (AttributeError, OSError, ValueError):
        info["memory_gb"] = None
    return info

# 本地服务鉴权：同一台机器上的任意进程/网页都能访问 127.0.0.1，
# 因此以一次性令牌挡住未授权的本机调用（与用户登录无关）。
_ENV_TOKEN = os.environ.get("CTHULHU_AUTH_TOKEN")
AUTH_TOKEN = _ENV_TOKEN if _ENV_TOKEN else secrets.token_urlsafe(32)
if not _ENV_TOKEN:
    print(f"[auth] 未提供 CTHULHU_AUTH_TOKEN，已生成一次性令牌：{AUTH_TOKEN}")


def _enable_faulthandler() -> None:
    """原生崩溃（段错误/中止）时落一份 Python 栈，便于定位引擎掉线原因。"""
    path = os.environ.get("CTHULHU_CRASH_LOG")
    if path:
        try:
            handle = open(path, "a", buffering=1, encoding="utf-8")  # noqa: SIM115 - 进程级句柄，随退出关闭
            handle.write(f"\n===== cthulhu-backend {APP_VERSION} pid={os.getpid()} 启动 =====\n")
            faulthandler.enable(file=handle)
            return
        except OSError as exc:
            print(f"[crash-log] 无法打开崩溃日志，回退 stderr：{exc}")
    faulthandler.enable()


_enable_faulthandler()


def _ws_send_timeout() -> float:
    try:
        value = float(os.environ.get("CTHULHU_WS_SEND_TIMEOUT", "10"))
    except ValueError:
        value = 10.0
    return max(1.0, value)


_WS_SEND_TIMEOUT = _ws_send_timeout()


async def _close_ws(ws: WebSocket, code: int = 1011, reason: str = "") -> None:
    """尽力关闭 WebSocket，连接状态未知时忽略一切关闭失败。"""
    try:
        await asyncio.wait_for(ws.close(code=code, reason=reason), timeout=2)
    except (TimeoutError, RuntimeError, WebSocketDisconnect):
        pass


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init_db()
    try:
        swept = sweep_thumb_cache()
        if swept["removed"]:
            print(f"[thumb-cache] 启动清理：移除 {swept['removed']} 个孤儿封面")
    except Exception as exc:  # noqa: BLE001 - 缓存清理失败不阻塞启动
        print(f"[thumb-cache] 启动清理失败（可忽略）：{exc}")
    saved_ffmpeg_dir = db.load_settings().get("ffmpeg_dir")
    if saved_ffmpeg_dir:
        from cthulhu_backend.media import ffmpeg

        ffmpeg.set_custom_dir(str(saved_ffmpeg_dir))
    try:
        from cthulhu_backend import services

        services.ensure_demo_library()
    except Exception as exc:  # noqa: BLE001 - ffmpeg 缺失时不阻塞启动
        print(f"[demo-library] 初始化失败（可忽略）：{exc}")
    job_queue.start()
    job_queue.restore()
    thumb_sweep_stop = threading.Event()
    threading.Thread(
        target=_periodic_thumb_sweep,
        args=(thumb_sweep_stop,),
        daemon=True,
        name="thumb-cache-sweeper",
    ).start()
    yield
    thumb_sweep_stop.set()
    await job_queue.stop()


def _periodic_thumb_sweep(stop_event: threading.Event) -> None:
    """低频兜底清理封面缓存孤儿；占用极小，无需实时。"""
    try:
        interval = float(os.environ.get("CTHULHU_THUMB_SWEEP_INTERVAL", "14400"))
    except ValueError:
        interval = 14400.0
    while not stop_event.wait(interval):
        try:
            swept = sweep_thumb_cache()
            if swept["removed"]:
                print(f"[thumb-cache] 定时清理：移除 {swept['removed']} 个孤儿封面")
        except Exception as exc:  # noqa: BLE001 - 清理失败不影响服务运行
            print(f"[thumb-cache] 定时清理失败（可忽略）：{exc}")


app = FastAPI(title="暗水印清洗台后端", version=APP_VERSION, lifespan=lifespan)


@app.middleware("http")
async def auth_middleware(request, call_next):
    """除健康检查外，所有 /api 请求都要求携带匹配令牌。"""
    if request.url.path == "/api/health" or not request.url.path.startswith("/api"):
        return await call_next(request)
    token = request.headers.get("x-cthulhu-token") or request.query_params.get("token")
    if token != AUTH_TOKEN:
        return JSONResponse(status_code=401, content={"detail": "无效的访问令牌"})
    return await call_next(request)


# 开发模式前端运行在 Vite（5173），允许其跨源调用本地 API。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health() -> dict:
    """健康检查，Electron 主进程据此判断后端是否就绪。"""
    from cthulhu_backend.media import ffmpeg

    return {
        "ok": True,
        "version": APP_VERSION,
        "service": "cthulhu-backend",
        "ffmpeg": ffmpeg.has_ffmpeg(),
        "host": _host_info(),
    }


@app.websocket("/ws/events")
async def events(ws: WebSocket) -> None:
    """任务进度与状态事件通道（心跳 + 任务事件广播）。"""
    if ws.query_params.get("token") != AUTH_TOKEN:
        await ws.close(code=4401, reason="无效的访问令牌")
        return
    # 先订阅再 accept：避免连接建立与订阅之间漏掉已广播的任务事件。
    queue = broker.subscribe()
    try:
        await ws.accept()
        await ws.send_json({"type": "hello", "version": APP_VERSION})
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=5)
            except TimeoutError:
                event = {"type": "heartbeat"}
            try:
                await asyncio.wait_for(ws.send_json(event), timeout=_WS_SEND_TIMEOUT)
            except TimeoutError:
                # 客户端长时间不消费（休眠/半开连接）：断开并释放资源，
                # 避免一个卡死的连接拖住事件循环协程或无限堆积事件。
                await _close_ws(ws, 1011, "发送超时")
                return
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        broker.unsubscribe(queue)


app.include_router(api_router)


# 生产模式：若前端已构建，则由后端托管静态资源。
_DIST = os.environ.get(
    "CTHULHU_STATIC_DIR",
    os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "ui", "dist")),
)
if os.path.isdir(_DIST):
    app.mount("/", StaticFiles(directory=_DIST, html=True), name="ui")


def main() -> None:
    """console script 入口。"""
    port = int(os.environ.get("CTHULHU_PORT", "57173"))
    uvicorn.run("cthulhu_backend.main:app", host="127.0.0.1", port=port)
