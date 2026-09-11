"""SQLite 持久层与 API 测试。"""

from __future__ import annotations

import gc
import sqlite3
import time

import pytest
from fastapi.testclient import TestClient

from cthulhu_backend import samples
from cthulhu_backend.main import app
from cthulhu_backend.media import ffmpeg

needs_ffmpeg = pytest.mark.skipif(not ffmpeg.has_ffmpeg(), reason="需要 ffmpeg")


@pytest.fixture(scope="module")
def client():
    with TestClient(app, headers={"X-CTHULHU-Token": "test-token"}) as test_client:
        yield test_client


def test_settings_roundtrip(client):
    payload = {"export_dir": "/tmp/out", "parallelism": "3", "gpu": "自动（带显存保护）"}
    response = client.put("/api/settings", json=payload)
    assert response.status_code == 200
    assert client.get("/api/settings").json()["export_dir"] == "/tmp/out"


def test_templates_empty_by_default_and_crud(client):
    templates = client.get("/api/templates").json()
    names = {item["name"] for item in templates}
    assert not names.intersection({"抖音投流", "快手分发", "跨平台通用"})

    created = client.post(
        "/api/templates",
        json={"name": "自定义模板", "payload": {"level": "平衡", "crf": 24}},
    )
    assert created.status_code == 201
    template_id = created.json()["id"]
    assert any(item["id"] == template_id for item in client.get("/api/templates").json())

    assert client.delete(f"/api/templates/{template_id}").status_code == 200
    assert client.delete(f"/api/templates/{template_id}").status_code == 404


@needs_ffmpeg
def test_job_persisted_after_completion(client, tmp_path):
    path = tmp_path / "v.mp4"
    ffmpeg.encode_video(samples.make_video_frames(6, 160, 120, seed=80), str(path), fps=30)
    job = client.post(
        "/api/jobs",
        json={"name": "持久化批次", "parallelism": 1, "tasks": [{"kind": "detect", "path": str(path)}]},
    ).json()

    deadline = time.time() + 60
    status = None
    while time.time() < deadline:
        status = client.get(f"/api/jobs/{job['id']}").json()["status"]
        if status in {"done", "failed"}:
            break
        time.sleep(0.1)
    assert status == "done"
    assert any(item["id"] == job["id"] for item in client.get("/api/jobs").json())


@needs_ffmpeg
def test_demo_library_and_report_cache(client, monkeypatch, tmp_path):
    from cthulhu_backend import db, services

    monkeypatch.setattr(services, "DEMO_LIBRARY_DIR", tmp_path)
    monkeypatch.setattr(services, "ENABLE_DEMO", True)
    assert services.ensure_demo_library() == 8

    files = [item for item in client.get("/api/demo/library").json()["files"] if item["video"]]
    assert len(files) == 8

    (tmp_path / "junk_repaired.mp4").write_bytes(b"not a video")
    canonical = client.get("/api/demo/library").json()["files"]
    assert len(canonical) == 8 and all(item["name"] != "junk_repaired.mp4" for item in canonical)

    first = files[0]
    client.post("/api/detect", json={"path": first["path"]})
    assert db.get_report(first["path"]) is not None

    again = client.get("/api/demo/library").json()["files"]
    assert all(item["report"] is None for item in again)


def test_demo_library_disabled_by_default(monkeypatch):
    """演示素材默认关闭，避免正式使用时素材库混入合成演示视频。"""
    from cthulhu_backend import services

    monkeypatch.setattr(services, "ENABLE_DEMO", False)
    assert services.ensure_demo_library() == 0
    assert services.demo_library() == {"directory": True, "files": []}


@needs_ffmpeg
def test_library_register_and_remove(client, tmp_path):
    """素材登记后可跨请求恢复；删除记录不动桌面源文件。"""
    path = tmp_path / "持久化素材.mp4"
    ffmpeg.encode_video(samples.make_video_frames(6, 160, 120, seed=1), str(path), fps=30)

    registered = client.post("/api/library", json={"path": str(path)})
    assert registered.status_code == 200
    assert registered.json()["video"]["width"] == 160

    files = client.get("/api/library").json()["files"]
    assert any(item["path"] == str(path) and item["video"] for item in files)

    removed = client.delete("/api/library", params={"path": str(path)})
    assert removed.json()["removed"] == 1
    assert path.exists()


def _live_connections() -> int:
    """进程里仍存活的 sqlite3 连接对象数（用于观察连接是否被及时释放）。"""
    return sum(1 for obj in gc.get_objects() if isinstance(obj, sqlite3.Connection))


def test_db_operations_do_not_keep_connections_alive(monkeypatch, tmp_path):
    """公开写操作必须即开即关：连接不得留在引用环里等 GC（历史 bug：fd 与页缓存累积）。"""
    from cthulhu_backend import db

    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "conn.db"))
    db.init_db()
    gc.collect()
    baseline = _live_connections()
    for index in range(20):
        db.save_settings({f"k{index}": index})
    gc.collect()
    assert _live_connections() - baseline <= 2


def test_clear_finished_jobs(client):
    created = client.post(
        "/api/jobs",
        json={"name": "待清理", "parallelism": 1, "tasks": [{"kind": "detect", "path": "/no.mp4"}]},
    ).json()
    deadline = time.time() + 60
    while time.time() < deadline:
        if client.get(f"/api/jobs/{created['id']}").json()["status"] in {"done", "failed"}:
            break
        time.sleep(0.1)
    response = client.delete("/api/jobs", params={"scope": "finished"})
    assert response.status_code == 200 and response.json()["removed"] >= 1
    assert all(item["id"] != created["id"] for item in client.get("/api/jobs").json())


def test_jobs_newest_first(client):
    """任务清单应最新在前。"""
    older = client.post(
        "/api/jobs",
        json={"name": "旧任务", "parallelism": 1, "tasks": [{"kind": "detect", "path": "/no/old.mp4"}]},
    ).json()
    newer = client.post(
        "/api/jobs",
        json={"name": "新任务", "parallelism": 1, "tasks": [{"kind": "detect", "path": "/no/new.mp4"}]},
    ).json()
    jobs = client.get("/api/jobs").json()
    ids = [item["id"] for item in jobs if item["id"] in {older["id"], newer["id"]}]
    assert ids == [newer["id"], older["id"]]
