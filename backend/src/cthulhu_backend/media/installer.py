"""FFmpeg/ffprobe 静态二进制一键下载与安装（后台线程 + 状态机）。"""

from __future__ import annotations

import os
import platform
import shutil
import threading
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path

from cthulhu_backend.media import ffmpeg

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


def install_ffmpeg(target_dir: str, *, on_installed: Callable[[str], None] | None = None) -> dict:
    """在后台线程下载静态 ffmpeg/ffprobe 并安装到目标目录。

    安装器不依赖持久层：装好后把目标目录回调给调用方（由其决定写设置），
    这样单测与嵌入式用法都不必拉起数据库。
    """
    if _ffmpeg_install_state.get("status") == "downloading":
        return ffmpeg_install_state()
    _ffmpeg_install_state.update({"status": "downloading", "detail": ""})
    threading.Thread(
        target=_install_worker, args=(target_dir, on_installed), daemon=True
    ).start()
    return ffmpeg_install_state()


def _install_worker(target_dir: str, on_installed: Callable[[str], None] | None) -> None:
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
        if on_installed is not None:
            on_installed(str(target))
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
