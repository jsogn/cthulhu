"""SQLite 持久层：设置、模板、审计日志与任务历史。"""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from pathlib import Path
from typing import Any

DB_PATH = os.environ.get(
    "CTHULHU_DB",
    str(Path(__file__).resolve().parents[2] / "data" / "cthulhu.db"),
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS templates (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    status TEXT NOT NULL,
    parallelism INTEGER NOT NULL,
    created_at REAL NOT NULL,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS reports (
    path TEXT PRIMARY KEY,
    payload TEXT NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS library (
    path TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    size INTEGER NOT NULL,
    meta TEXT NOT NULL DEFAULT '{}',
    added_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS variants (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    output TEXT NOT NULL,
    options TEXT NOT NULL,
    seed INTEGER NOT NULL,
    template_id TEXT,
    metrics TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_variants_output ON variants (output);
"""

def _connect() -> sqlite3.Connection:
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db() -> None:
    with _connect() as connection:
        connection.executescript(SCHEMA)
        # 早期实验版 variants 表含 batch/label 列，精简版移除；老库自动迁移。
        columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(variants)").fetchall()
        }
        indexes = {
            row["name"]
            for row in connection.execute("PRAGMA index_list(variants)").fetchall()
        }
        if "idx_variants_batch" in indexes:
            connection.execute("DROP INDEX idx_variants_batch")
        for legacy in ("batch", "label"):
            if legacy in columns:
                connection.execute(f"ALTER TABLE variants DROP COLUMN {legacy}")
        # 处理历史已与任务中心合并，清理旧审计表。
        connection.execute("DROP TABLE IF EXISTS audit")
        # 演示素材默认关闭；清理旧版本写入的演示库残留，避免用户素材库里
        # 出现无法删除的合成演示视频。
        if os.environ.get("CTHULHU_DEMO_LIBRARY") != "1":
            _ensure_library_table(connection)
            connection.execute("DELETE FROM library WHERE path LIKE '%demo-library%'")
        # 平台命名的种子模板从未实现平台专属处理，属误导性残留，统一清除；
        # 模板改为完全由用户创建与管理。
        for legacy_name in ("抖音投流", "快手分发", "跨平台通用"):
            connection.execute("DELETE FROM templates WHERE name = ?", (legacy_name,))
        # 迁移旧版展示参数为真实清洗参数。
        legacy = {"轻度": (25, 15, False), "平衡": (30, 20, True), "深度": (40, 30, True)}
        for row in connection.execute("SELECT id, payload FROM templates").fetchall():
            payload = json.loads(row["payload"])
            # 旧字段 restruct 或新字段 retime 均表示已经迁移过，直接跳过。
            if "restruct" in payload or "retime" in payload:
                continue
            restruct, perturb, denoise = legacy.get(payload.get("level"), (30, 20, True))
            connection.execute(
                "UPDATE templates SET payload = ? WHERE id = ?",
                (
                    json.dumps(
                        {
                            "level": payload.get("level", "平衡"),
                            "restruct": restruct,
                            "perturb": perturb,
                            "denoise": denoise,
                            "audio": True,
                            "codec": "H.264",
                        },
                        ensure_ascii=False,
                    ),
                    row["id"],
                ),
            )


# ---------- 设置 ----------
def load_settings() -> dict[str, Any]:
    try:
        with _connect() as connection:
            rows = connection.execute("SELECT key, value FROM settings").fetchall()
        return {row["key"]: json.loads(row["value"]) for row in rows}
    except sqlite3.OperationalError:
        # 数据库尚未初始化（如测试直接调用服务）时返回空设置。
        return {}


def save_settings(values: dict[str, Any]) -> None:
    with _connect() as connection:
        connection.executemany(
            "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
            [(key, json.dumps(value, ensure_ascii=False)) for key, value in values.items()],
        )


# ---------- 模板 ----------
def list_templates() -> list[dict]:
    with _connect() as connection:
        rows = connection.execute("SELECT id, name, payload, created_at FROM templates ORDER BY created_at").fetchall()
    return [
        {"id": row["id"], "name": row["name"], "payload": json.loads(row["payload"]), "created_at": row["created_at"]}
        for row in rows
    ]


def create_template(name: str, payload: dict) -> dict:
    import time

    template_id = uuid.uuid4().hex[:12]
    with _connect() as connection:
        connection.execute(
            "INSERT INTO templates (id, name, payload, created_at) VALUES (?, ?, ?, ?)",
            (template_id, name, json.dumps(payload, ensure_ascii=False), time.time()),
        )
    return {"id": template_id, "name": name, "payload": payload}


def update_template(template_id: str, name: str, payload: dict) -> bool:
    with _connect() as connection:
        cursor = connection.execute(
            "UPDATE templates SET name = ?, payload = ? WHERE id = ?",
            (name, json.dumps(payload, ensure_ascii=False), template_id),
        )
    return cursor.rowcount > 0


def delete_template(template_id: str) -> bool:
    with _connect() as connection:
        cursor = connection.execute("DELETE FROM templates WHERE id = ?", (template_id,))
    return cursor.rowcount > 0


# ---------- 产物记录（变体） ----------
def create_variant(
    source: str,
    output: str,
    options: dict,
    seed: int,
    template_id: str | None = None,
    metrics: dict | None = None,
) -> dict:
    import time

    variant_id = uuid.uuid4().hex[:12]
    with _connect() as connection:
        connection.execute(
            "INSERT INTO variants (id, source, output, options, seed, template_id, metrics, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                variant_id,
                source,
                output,
                json.dumps(options, ensure_ascii=False),
                int(seed),
                template_id,
                json.dumps(metrics or {}, ensure_ascii=False),
                time.time(),
            ),
        )
    return {
        "id": variant_id,
        "source": source,
        "output": output,
        "options": options,
        "seed": int(seed),
        "template_id": template_id,
        "metrics": metrics or {},
        "created_at": time.time(),
    }


def list_variants(source: str | None = None) -> list[dict]:
    query = "SELECT * FROM variants"
    conditions: list[str] = []
    params: list[Any] = []
    if source:
        conditions.append("source = ?")
        params.append(source)
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY created_at DESC"
    with _connect() as connection:
        rows = connection.execute(query, params).fetchall()
    return [
        {
            "id": row["id"],
            "source": row["source"],
            "output": os.path.expanduser(row["output"]),
            "options": json.loads(row["options"]),
            "seed": row["seed"],
            "template_id": row["template_id"],
            "metrics": json.loads(row["metrics"]),
            "created_at": row["created_at"],
        }
        for row in rows
    ]


def delete_variant_by_output(output: str) -> bool:
    try:
        with _connect() as connection:
            cursor = connection.execute(
                "DELETE FROM variants WHERE output IN (?, ?)",
                (output, os.path.expanduser(output)),
            )
        return cursor.rowcount > 0
    except sqlite3.OperationalError:
        # 表尚未初始化时（如测试直接调用删除），视为无记录。
        return False


# ---------- 任务 ----------
def save_job(job: dict) -> None:
    with _connect() as connection:
        connection.execute(
            "INSERT OR REPLACE INTO jobs (id, name, status, parallelism, created_at, payload) VALUES (?, ?, ?, ?, ?, ?)",
            (
                job["id"],
                job["name"],
                job["status"],
                job["parallelism"],
                job["created_at"],
                json.dumps(job, ensure_ascii=False),
            ),
        )


def load_jobs() -> list[dict]:
    with _connect() as connection:
        rows = connection.execute("SELECT payload FROM jobs ORDER BY created_at DESC").fetchall()
    return [json.loads(row["payload"]) for row in rows]


def delete_jobs(scope: str = "all") -> int:
    with _connect() as connection:
        if scope == "all":
            cursor = connection.execute("DELETE FROM jobs")
        else:
            cursor = connection.execute(
                "DELETE FROM jobs WHERE status IN ('done', 'failed', 'canceled')"
            )
    return cursor.rowcount


# ---------- 检测报告缓存 ----------
def save_report(path: str, report: dict) -> None:
    import time

    with _connect() as connection:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS reports (path TEXT PRIMARY KEY, payload TEXT NOT NULL, updated_at REAL NOT NULL)"
        )
        connection.execute(
            "INSERT OR REPLACE INTO reports (path, payload, updated_at) VALUES (?, ?, ?)",
            (path, json.dumps(report, ensure_ascii=False), time.time()),
        )


def get_report(path: str) -> dict | None:
    with _connect() as connection:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS reports (path TEXT PRIMARY KEY, payload TEXT NOT NULL, updated_at REAL NOT NULL)"
        )
        row = connection.execute("SELECT payload FROM reports WHERE path = ?", (path,)).fetchone()
    return json.loads(row["payload"]) if row else None


# ---------- 素材库 ----------
def _ensure_library_table(connection: sqlite3.Connection) -> None:
    connection.execute(
        "CREATE TABLE IF NOT EXISTS library ("
        "path TEXT PRIMARY KEY, name TEXT NOT NULL, size INTEGER NOT NULL, "
        "meta TEXT NOT NULL DEFAULT '{}', added_at REAL NOT NULL)"
    )


def add_library(path: str, name: str, size: int, meta: dict | None = None) -> None:
    """登记素材到持久化素材库；重复导入会刷新时间，排到最前。"""
    import time

    with _connect() as connection:
        _ensure_library_table(connection)
        connection.execute(
            "INSERT OR REPLACE INTO library (path, name, size, meta, added_at) VALUES (?, ?, ?, ?, ?)",
            (path, name, size, json.dumps(meta or {}, ensure_ascii=False), time.time()),
        )


def list_library() -> list[dict]:
    """按最新导入在前返回素材清单（同批导入按先后倒序）。"""
    with _connect() as connection:
        _ensure_library_table(connection)
        rows = connection.execute(
            "SELECT rowid, path, name, size, meta FROM library ORDER BY added_at DESC, rowid DESC"
        ).fetchall()
    return [
        {
            "path": row["path"],
            "name": row["name"],
            "size": row["size"],
            "meta": json.loads(row["meta"]),
        }
        for row in rows
    ]


def update_library_meta(path: str, meta: dict) -> None:
    with _connect() as connection:
        _ensure_library_table(connection)
        connection.execute(
            "UPDATE library SET meta = ? WHERE path = ?",
            (json.dumps(meta, ensure_ascii=False), path),
        )


def remove_library(path: str) -> bool:
    with _connect() as connection:
        _ensure_library_table(connection)
        cursor = connection.execute("DELETE FROM library WHERE path = ?", (path,))
    return cursor.rowcount > 0
