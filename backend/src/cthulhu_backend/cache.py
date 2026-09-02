"""有界分析结果缓存：只缓存派生的轻量结果，不缓存原始帧或音频。

缓存对象是「颜色统计」与「镜头边界」这类几 KB 的小结果，而不是抽样帧
数组（长视频一帧抽样的 float32 可达数 GB）。从设计上保证磁盘占用可预期：
目录总量有硬上限，超限按 mtime 做 LRU 清理，进程退出后缓存仍在但受上限
约束，不会随素材数量无界增长。

键基于文件身份（真实路径 + 大小 + mtime_ns + 首尾 128KB 的快速哈希），
读取时重新校验，文件被替换/修改后自动失效，不需要完整文件 MD5。
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np

from cthulhu_backend import db

# 缓存条目格式版本：detect_cuts / channel_stats 等派生算法语义变化时必须
# 递增，否则旧机器上的缓存会静默沿用旧算法结果。
SCHEMA_VERSION = 1
_QUICK_HASH_BYTES = 64 * 1024

_DEFAULT_MAX_BYTES = 16 * 1024 * 1024


def _quick_hash(path: str) -> str:
    """首尾各 64KB 的快速指纹：校验文件身份用，避免完整 MD5 读全片。"""
    digest = hashlib.md5()
    size = os.path.getsize(path)
    with open(path, "rb") as handle:
        digest.update(handle.read(_QUICK_HASH_BYTES))
        if size > _QUICK_HASH_BYTES:
            handle.seek(-_QUICK_HASH_BYTES, os.SEEK_END)
            digest.update(handle.read(_QUICK_HASH_BYTES))
    return digest.hexdigest()


def _file_identity(path: str) -> dict[str, Any]:
    real = os.path.realpath(path)
    stat = os.stat(real)
    return {
        "path": real,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "quick_hash": _quick_hash(real),
    }


class AnalysisCache:
    """按文件身份保存/读取颜色统计与镜头边界，磁盘总量有上限。"""

    def __init__(self, directory: Path, max_bytes: int | None = None) -> None:
        self._dir = Path(directory)
        env_bytes = os.environ.get("CTHULHU_CACHE_MAX_BYTES")
        if env_bytes:
            try:
                max_bytes = max(1024, int(env_bytes))
            except ValueError:
                pass
        self._max_bytes = int(max_bytes or _DEFAULT_MAX_BYTES)
        self._lock = threading.Lock()

    def _entry_path(self, kind: str, path: str) -> Path:
        # 文件名只由 kind + 真实路径决定：同一素材无论怎么变化都落在同一
        # 个条目上，失效校验（身份字段）失败时才能删到正确的旧文件。
        raw = f"{kind}|{path}"
        key = hashlib.md5(raw.encode()).hexdigest()[:24]
        return self._dir / f"{key}.json"

    def _load(self, kind: str, path: str) -> Any | None:
        try:
            current = _file_identity(path)
            entry_path = self._entry_path(kind, current["path"])
            if not entry_path.exists():
                return None
            with open(entry_path, encoding="utf-8") as handle:
                entry = json.load(handle)
            # 缓存文件被外部改坏（合法 JSON 但结构不对）时删除条目并退化为
            # 未命中，绝不能把解析异常漏给清洗主流程。
            if not isinstance(entry, dict):
                entry_path.unlink(missing_ok=True)
                return None
            if entry.get("version") != SCHEMA_VERSION or entry.get("kind") != kind:
                entry_path.unlink(missing_ok=True)
                return None
            stored = entry.get("identity") or {}
            if not isinstance(stored, dict):
                entry_path.unlink(missing_ok=True)
                return None
            if any(stored.get(k) != current[k] for k in ("path", "size", "mtime_ns", "quick_hash")):
                entry_path.unlink(missing_ok=True)
                return None
            os.utime(entry_path, None)
            return entry["data"]
        except (OSError, ValueError, KeyError):
            return None

    def _save(self, kind: str, path: str, data: Any) -> None:
        try:
            identity = _file_identity(path)
            entry = {
                "version": SCHEMA_VERSION,
                "kind": kind,
                "identity": identity,
                "data": data,
            }
            self._dir.mkdir(parents=True, exist_ok=True)
            entry_path = self._entry_path(kind, identity["path"])
            tmp_path = entry_path.with_suffix(".tmp")
            with open(tmp_path, "w", encoding="utf-8") as handle:
                json.dump(entry, handle, ensure_ascii=False)
            os.replace(tmp_path, entry_path)
            self._trim()
        except OSError:
            pass

    def _trim(self) -> None:
        """目录总量超限时按 mtime 从旧到新删除，直到回到上限内。"""
        try:
            entries = [p for p in self._dir.iterdir() if p.is_file()]
            # 进程崩溃遗留的临时文件不再参与统计，直接清掉。
            for stale in [p for p in entries if p.suffix == ".tmp"]:
                stale.unlink(missing_ok=True)
            json_files = [p for p in entries if p.suffix == ".json"]
            total = sum(p.stat().st_size for p in json_files)
            if total <= self._max_bytes:
                return
            for entry in sorted(json_files, key=lambda p: p.stat().st_mtime):
                if total <= self._max_bytes:
                    break
                size = entry.stat().st_size
                entry.unlink(missing_ok=True)
                total -= size
        except OSError:
            pass

    # ---------- 对外接口：失败一律退化为“未命中”，不影响主流程 ----------

    def get_color_stats(self, path: str) -> tuple[np.ndarray, np.ndarray] | None:
        with self._lock:
            data = self._load("color_stats", path)
        if not isinstance(data, dict):
            return None
        try:
            ref_mean = np.asarray(data["ref_mean"], dtype=np.float32)
            ref_std = np.asarray(data["ref_std"], dtype=np.float32)
        except (KeyError, TypeError, ValueError):
            return None
        return ref_mean, ref_std

    def put_color_stats(self, path: str, ref_mean: np.ndarray, ref_std: np.ndarray) -> None:
        with self._lock:
            self._save(
                "color_stats",
                path,
                {
                    "ref_mean": np.asarray(ref_mean).tolist(),
                    "ref_std": np.asarray(ref_std).tolist(),
                },
            )

    def get_shot_boundaries(self, path: str) -> list[tuple[int, int]] | None:
        with self._lock:
            data = self._load("shot_boundaries", path)
        if not isinstance(data, list):
            return None
        try:
            return [
                (int(item[0]), int(item[1]))
                for item in data
                if len(item) == 2 and int(item[0]) < int(item[1])
            ]
        except (KeyError, TypeError, ValueError):
            return None

    def put_shot_boundaries(self, path: str, boundaries: list[tuple[int, int]]) -> None:
        payload = [[int(a), int(b)] for a, b in boundaries]
        with self._lock:
            self._save("shot_boundaries", path, payload)


def get_analysis_cache() -> AnalysisCache:
    """惰性单例：目录默认落在应用数据目录下，环境变量可覆盖。"""
    directory = Path(
        os.environ.get("CTHULHU_CACHE_DIR", str(Path(db.DB_PATH).parent / "analysis_cache"))
    )
    return AnalysisCache(directory)


analysis_cache = get_analysis_cache()
