"""暗水印清洗台后端入口。

开发模式：Electron 以 sidecar 方式拉起 uvicorn，Vite 将 /api 与 /ws 代理到本服务。
生产模式：本服务额外托管前端构建产物（ui/dist），Electron 窗口直接加载本服务地址。
"""

from __future__ import annotations

import asyncio
import os
import platform
import secrets
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from cthulhu_backend import db
from cthulhu_backend.api import router as api_router
from cthulhu_backend.events import broker
from cthulhu_backend.jobs import job_queue

APP_VERSION = "0.1.0"


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


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init_db()
    saved_ffmpeg_dir = db.load_settings().get("ffmpeg_dir")
    if saved_ffmpeg_dir:
        from cthulhu_backend.media import ffmpeg

        ffmpeg.set_custom_dir(str(saved_ffmpeg_dir))
    try:
        from cthulhu_backend import services

        services.ensure_demo_library()
    except Exception as exc:  # noqa: BLE001 - ffmpeg 缺失时不阻塞启动
        print(f"[demo-library] 初始化失败（可忽略）：{exc}")
    job_queue.restore()
    job_queue.start()
    yield
    await job_queue.stop()


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
    await ws.accept()
    await ws.send_json({"type": "hello", "version": APP_VERSION})
    queue = broker.subscribe()
    try:
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=5)
            except TimeoutError:
                event = {"type": "heartbeat"}
            await ws.send_json(event)
    except WebSocketDisconnect:
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
