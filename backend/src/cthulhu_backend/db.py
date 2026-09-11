"""SQLite 持久层：设置、模板、审计日志与任务历史。"""

from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from cthulhu_backend.schemas import TemplatePayload
from cthulhu_backend.tiers import tier_purify

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
    created_at REAL NOT NULL,
    builtin INTEGER NOT NULL DEFAULT 0
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
    kind TEXT NOT NULL DEFAULT 'cleaned',
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_variants_output ON variants (output);
"""

def _preset_payload(**overrides) -> dict:
    """以 TemplatePayload 构造预置模板，保证种子数据与写入契约一致。"""
    return TemplatePayload(**overrides).model_dump(mode="json")


# 预置内容签名：改动 PRESET_TEMPLATES 内容必须同时升 PRESET_VERSION 并更新这里，
# 否则 test_preset_content_signature_matches_version 会失败（审计 R3 的两次真实故障）。
PRESET_SIGNATURES = {34: "39d4e9a2cc3cf935", 35: "b35382a1d4e64629"}


# v35：模板 payload 随参数精简收敛（移除 hsvJitter 等已删字段）后重新播种。
# v34 起内置四个「大平台深度水印」预设（画质优先/平衡/强力/深度清剿），
# 全部启用画面重建；旧库里的经典武器模板与旧字段会在下次启动时被替换。
PRESET_VERSION = 35


PRESET_TEMPLATES = [
    # 四个默认模板都以「大平台深度（暗）水印」为目标：全部打开画面重建，
    # 区别只在清晰度 / 清除强度的取舍（research §19.1~19.6）。
    # 清晰度字段（长边/批大小/细节回填）来自 tiers.py，预设只写自己的强度与武器。
    {
        "name": "画质优先（推荐）",
        "tier": "quality",
        "payload": _preset_payload(
            **tier_purify("quality"),
            audioRemix=True,
            echoDefeat=True,
            audioStrong=True,
            embeddingAttack="both",
            embeddingStrength=0.0,
            embeddingVariant="v2",
            embeddingAggressive=False,
            autoProfile=True,
            sharpness=True,
            colorRestore=True,
            denoise=True,
        ),
    },
    {
        "name": "平衡去水印",
        "tier": "balanced",
        "payload": _preset_payload(
            **tier_purify("balanced"),
            audioRemix=True,
            echoDefeat=True,
            audioStrong=True,
            embeddingAttack="both",
            embeddingStrength=0.0,
            embeddingVariant="v2",
            embeddingAggressive=False,
            autoProfile=True,
            sharpness=True,
            colorRestore=True,
            denoise=True,
        ),
    },
    {
        "name": "强力去水印",
        "tier": "strong",
        "payload": _preset_payload(
            **tier_purify("strong"),
            audioRemix=True,
            echoDefeat=True,
            audioStrong=True,
            embeddingAttack="both",
            embeddingStrength=0.6,
            embeddingVariant="v2",
            embeddingAggressive=True,
            autoProfile=False,
            sharpness=True,
            colorRestore=True,
            denoise=True,
        ),
    },
    {
        "name": "深度清剿（最狠）",
        "tier": "max",
        "payload": _preset_payload(
            **tier_purify("max"),
            audioRemix=True,
            echoDefeat=True,
            audioStrong=True,
            embeddingAttack="both",
            embeddingStrength=0.6,
            embeddingVariant="v2",
            embeddingAggressive=True,
            autoProfile=False,
            requant=64,
            noise=0.003,
            temporalSub=0.6,
            nativeTemporal=True,
            fftPhase=0.5,
            dwtDetail=0.8,
            hashAttack=True,
            hashEpsilon=0.05,
            sharpness=True,
            colorRestore=True,
            denoise=True,
            qualityProtect=True,
            psnrTarget=30.0,
            ssimTarget=0.88,
        ),
    },
]


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    """打开一个短期连接：WAL + 写等待，正常退出自动提交，无论成败都显式关闭。

    旧实现只提交不关闭，连接对象参与引用环，文件描述符与页缓存要到
    周期性 GC 才回收，高流量下会持续累积。
    """
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    # WAL 让读不阻塞写，busy_timeout 吸收并发写竞争，避免 "database is locked"。
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA busy_timeout=5000")
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


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
        # 产物类型改为记录字段承载，不再依赖文件名推断；老库补列并按历史命名回填。
        if "kind" not in columns:
            connection.execute(
                "ALTER TABLE variants ADD COLUMN kind TEXT NOT NULL DEFAULT 'cleaned'"
            )
            connection.execute(
                "UPDATE variants SET kind = 'repaired' "
                "WHERE output LIKE '%_修复%' OR output LIKE '%_repaired%'"
            )
            connection.execute("UPDATE variants SET kind = 'candidate' WHERE output LIKE '%_候选%'")
        # 处理历史已与任务中心合并，清理旧审计表。
        connection.execute("DROP TABLE IF EXISTS audit")
        # 演示素材默认关闭；清理旧版本写入的演示库残留，避免用户素材库里
        # 出现无法删除的合成演示视频。
        if os.environ.get("CTHULHU_DEMO_LIBRARY") != "1":
            _ensure_library_table(connection)
            connection.execute("DELETE FROM library WHERE path LIKE '%demo-library%'")
            # 按真实路径归并历史重复导入（保留最近一次），修复旧版本重复记录。
            seen: dict[str, str] = {}
            for row in connection.execute(
                "SELECT path FROM library ORDER BY added_at DESC"
            ).fetchall():
                try:
                    key = os.path.realpath(row["path"])
                except OSError:
                    key = row["path"]
                if key in seen:
                    connection.execute("DELETE FROM library WHERE path = ?", (row["path"],))
                else:
                    seen[key] = row["path"]
        # 平台命名的旧版种子模板从未实现平台专属处理，属误导性残留，统一清除。
        for legacy_name in ("抖音投流", "快手分发", "跨平台通用"):
            connection.execute("DELETE FROM templates WHERE name = ?", (legacy_name,))
        # 迁移旧版展示参数为真实清洗参数。
        legacy = {"轻度": (25, 15, False), "平衡": (30, 20, True), "深度": (40, 30, True)}
        for row in connection.execute("SELECT id, payload FROM templates").fetchall():
            payload = json.loads(row["payload"])
            # 旧字段 restruct / 新字段 retime / 新版完整参数 audioRemix
            # 均表示已经是真实清洗参数，直接跳过。
            if (
                "restruct" in payload
                or "retime" in payload
                or "audioRemix" in payload
            ):
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
        # 预置默认模板：首次初始化（或升级旧库）时写入一次，用户可自由删除；
        # 按版本迁移：版本变化时重建内置预置（保留用户自建模板），版本一致
        # 不重写，用户删除内置预置后不会自动复活。
        template_columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(templates)").fetchall()
        }
        if "builtin" not in template_columns:
            connection.execute(
                "ALTER TABLE templates ADD COLUMN builtin INTEGER NOT NULL DEFAULT 0"
            )
        known_names = tuple(
            preset["name"] for preset in PRESET_TEMPLATES
        ) + ("快速 · 轻度", "标准 · 均衡", "强力 · 重对抗", "全兵器 · 研究")
        placeholders = ",".join("?" for _ in known_names)
        connection.execute(
            f"UPDATE templates SET builtin = 1 WHERE name IN ({placeholders})",
            known_names,
        )
        settings_rows = connection.execute("SELECT key, value FROM settings").fetchall()
        settings = {row["key"]: json.loads(row["value"]) for row in settings_rows}
        preset_version = int(
            settings.get(
                "preset_version",
                1 if settings.get("preset_templates_seeded") == 1 else 0,
            )
        )
        if preset_version < PRESET_VERSION:
            connection.execute("DELETE FROM templates WHERE builtin = 1")
            for preset in PRESET_TEMPLATES:
                connection.execute(
                    "INSERT INTO templates (id, name, payload, created_at, builtin) "
                    "VALUES (?, ?, ?, ?, 1)",
                    (
                        uuid.uuid4().hex[:12],
                        preset["name"],
                        json.dumps(preset["payload"], ensure_ascii=False),
                        time.time(),
                    ),
                )
            connection.execute(
                "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
                ("preset_version", json.dumps(PRESET_VERSION)),
            )
        connection.execute("DELETE FROM settings WHERE key = 'preset_templates_seeded'")
        # 启动迁移直接写设置表，重置进程内缓存避免读到旧值。
        global _settings_cache
        _settings_cache = None


# ---------- 设置 ----------
_settings_cache: dict[str, Any] | None = None


def load_settings() -> dict[str, Any]:
    """读取全部设置；进程内缓存，写入时失效，避免热路径反复开库读盘。"""
    global _settings_cache
    if _settings_cache is not None:
        return _settings_cache
    try:
        with _connect() as connection:
            rows = connection.execute("SELECT key, value FROM settings").fetchall()
        _settings_cache = {row["key"]: json.loads(row["value"]) for row in rows}
    except sqlite3.OperationalError:
        # 数据库尚未初始化（如测试直接调用服务）时返回空设置。
        return {}
    return _settings_cache


def save_settings(values: dict[str, Any]) -> None:
    global _settings_cache
    with _connect() as connection:
        connection.executemany(
            "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
            [(key, json.dumps(value, ensure_ascii=False)) for key, value in values.items()],
        )
    _settings_cache = None


# ---------- 模板 ----------
def list_templates() -> list[dict]:
    with _connect() as connection:
        # 正序排列：最早创建的在前，预置模板保持「快速/标准/强力/全兵器」顺序。
        rows = connection.execute(
            "SELECT id, name, payload, created_at FROM templates "
            "ORDER BY builtin DESC, created_at ASC, rowid ASC"
        ).fetchall()
    return [
        {"id": row["id"], "name": row["name"], "payload": json.loads(row["payload"]), "created_at": row["created_at"]}
        for row in rows
    ]


def create_template(name: str, payload: dict) -> dict:
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
    kind: str = "cleaned",
) -> dict:
    import time

    variant_id = uuid.uuid4().hex[:12]
    with _connect() as connection:
        connection.execute(
            "INSERT INTO variants (id, source, output, options, seed, template_id, metrics, kind, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                variant_id,
                source,
                output,
                json.dumps(options, ensure_ascii=False),
                int(seed),
                template_id,
                json.dumps(metrics or {}, ensure_ascii=False),
                kind,
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
        "kind": kind,
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
            "kind": row["kind"],
            "created_at": row["created_at"],
        }
        for row in rows
    ]


def _variant_record(row: sqlite3.Row) -> dict:
    """把 variants 行转换为对外字典（路径统一展开 ~）。"""
    return {
        "id": row["id"],
        "source": row["source"],
        "output": os.path.expanduser(row["output"]),
        "options": json.loads(row["options"]),
        "seed": row["seed"],
        "template_id": row["template_id"],
        "metrics": json.loads(row["metrics"]),
        "kind": row["kind"],
        "created_at": row["created_at"],
    }


def _resolve_variant_outputs(connection: sqlite3.Connection, output: str) -> list[str]:
    """返回与给定输出路径等价的 variants.output 拼写列表。

    覆盖三种口径：调用方原样传入、~ 展开、以及库里经 realpath 归并的同文件
    记录。update/delete 共用此解析；get 保留索引直查快路径，不走全表扫描。
    """
    matched = [output]
    expanded = os.path.expanduser(output)
    if expanded != output:
        matched.append(expanded)
    target = os.path.realpath(expanded)
    for row in connection.execute("SELECT output FROM variants").fetchall():
        candidate = row["output"]
        if candidate not in matched and os.path.realpath(os.path.expanduser(candidate)) == target:
            matched.append(candidate)
    return matched


def get_variant_by_output(output: str) -> dict | None:
    """按输出路径查找产物记录，兼容 ~ 与绝对路径两种写法。

    先走 output 索引直查常见拼写（原样 / ~ 展开），未命中再回退全表
    realpath 归并，软链接与路径别名下语义不变。
    """
    try:
        with _connect() as connection:
            candidates = {output, os.path.expanduser(output)}
            for candidate in candidates:
                row = connection.execute(
                    "SELECT id, source, output, options, seed, template_id, metrics, "
                    "kind, created_at FROM variants WHERE output = ?",
                    (candidate,),
                ).fetchone()
                if row is not None:
                    return _variant_record(row)
            target = os.path.realpath(os.path.expanduser(output))
            for row in connection.execute(
                "SELECT id, source, output, options, seed, template_id, metrics, kind, created_at "
                "FROM variants"
            ).fetchall():
                if os.path.realpath(os.path.expanduser(row["output"])) == target:
                    return _variant_record(row)
    except sqlite3.OperationalError:
        return None
    return None


def update_variant_metrics(output: str, metrics: dict) -> bool:
    """按输出路径回写产物指标（异步指标遍完成后调用），命中返回 True。

    路径匹配口径与 get_variant_by_output 一致（原样 / ~ 展开 / realpath）。
    """
    payload = json.dumps(metrics or {}, ensure_ascii=False)
    try:
        with _connect() as connection:
            for candidate in _resolve_variant_outputs(connection, output):
                cursor = connection.execute(
                    "UPDATE variants SET metrics = ? WHERE output = ?",
                    (payload, candidate),
                )
                if cursor.rowcount:
                    return True
    except sqlite3.OperationalError:
        return False
    return False


def delete_variant_by_output(output: str) -> bool:
    """按输出路径删除产物记录，兼容 ~ 与绝对路径两种写法。"""
    try:
        with _connect() as connection:
            matched = _resolve_variant_outputs(connection, output)
            placeholders = ", ".join("?" for _ in matched)
            cursor = connection.execute(
                f"DELETE FROM variants WHERE output IN ({placeholders})",
                matched,
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
    """登记素材到持久化素材库；重复导入（含符号链接/大小写别名）会刷新时间，排到最前。"""
    import time

    real = os.path.realpath(path)

    with _connect() as connection:
        _ensure_library_table(connection)
        # 同一文件可能以不同路径字符串出现（/tmp 与 /private/tmp、iCloud
        # 别名、大小写变体），按解析后的真实路径归并，复用原登记路径。
        for row in connection.execute("SELECT path FROM library").fetchall():
            try:
                if os.path.realpath(row["path"]) == real:
                    path = row["path"]
                    break
            except OSError:
                continue
        connection.execute(
            "INSERT OR REPLACE INTO library (path, name, size, meta, added_at) VALUES (?, ?, ?, ?, ?)",
            (path, name, size, json.dumps(meta or {}, ensure_ascii=False), time.time()),
        )
        # 清除同一文件的历史别名记录（符号链接/大小写变体造成的重复行）。
        for row in connection.execute("SELECT path FROM library").fetchall():
            if row["path"] == path:
                continue
            try:
                if os.path.realpath(row["path"]) == real:
                    connection.execute("DELETE FROM library WHERE path = ?", (row["path"],))
            except OSError:
                continue


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
