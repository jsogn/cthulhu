"""任务队列与调度测试。"""

from __future__ import annotations

import asyncio
import time
import uuid

import pytest
from fastapi.testclient import TestClient

from cthulhu_backend import db, jobs, planning, samples, services
from cthulhu_backend.jobs import JobQueue
from cthulhu_backend.main import app
from cthulhu_backend.media import ffmpeg

needs_ffmpeg = pytest.mark.skipif(not ffmpeg.has_ffmpeg(), reason="需要 ffmpeg/ffprobe")

@pytest.fixture(scope="module")
def client():
    # 后台任务调度依赖常驻事件循环，须用上下文管理器保持 portal 存活。
    with TestClient(app, headers={"X-CTHULHU-Token": "test-token"}) as test_client:
        yield test_client


def _video(tmp_path, seed=50):
    path = tmp_path / "v.mp4"
    ffmpeg.encode_video(samples.make_video_frames(8, 160, 120, seed=seed), str(path), fps=30)
    return path


def _wait(client: TestClient, job_id: str, timeout: float = 60.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in {"done", "failed", "canceled"}:
            return job
        time.sleep(0.1)
    raise TimeoutError("任务未在期限内结束")


def test_restore_marks_interrupted_job_failed(client):
    """进程中断的未完成任务标记为失败，不静默从头重跑。"""
    job_id = uuid.uuid4().hex[:12]
    db.save_job({
        "id": job_id,
        "name": "中断测试",
        "status": "running",
        "parallelism": 2,
        "created_at": time.time(),
        "tasks": [{
            "id": "t1",
            "kind": "detect",
            "path": "/tmp/x.mp4",
            "options": {},
            "status": "running",
            "percent": 30,
            "result": None,
            "error": None,
        }],
    })
    fresh = JobQueue()
    fresh.restore()
    restored = fresh._jobs[job_id]
    assert restored["status"] == "failed"
    assert restored["tasks"][0]["status"] == "failed"
    assert "进程重启中断" in restored["tasks"][0]["error"]
    assert restored["tasks"][0]["progress_note"] == "进程重启中断"
    db.delete_jobs("all")


def test_restore_does_not_auto_restart_interrupted_job():
    """restore 之后 start 不会把中断任务重新跑一遍。"""
    job_id = uuid.uuid4().hex[:12]
    db.save_job({
        "id": job_id,
        "name": "重启续跑",
        "status": "running",
        "parallelism": 1,
        "created_at": time.time(),
        "tasks": [{
            "id": "t1",
            "kind": "detect",
            "path": "/no/such/file.mp4",
            "options": {},
            "status": "running",
            "percent": 30,
            "result": None,
            "error": None,
        }],
    })
    queue = JobQueue()

    async def drive():
        # 与 lifespan 相同的顺序：先恢复、再启动调度。
        queue.restore()
        queue.start()
        try:
            await asyncio.sleep(0.2)
            return queue.get(job_id)
        finally:
            await queue.stop()

    try:
        job = asyncio.run(drive())
    finally:
        db.delete_jobs("all")
    assert job["status"] == "failed"
    assert job["tasks"][0]["status"] == "failed"
    assert "进程重启中断" in job["tasks"][0]["error"]
    assert job["tasks"][0]["error"], "失败子任务缺少错误信息"


@needs_ffmpeg
def test_desensitize_resolves_strategy_and_preset_from_settings(tmp_path):
    """未在任务选项中显式指定时，策略与编码档从设置项解析。"""
    video = _video(tmp_path, seed=51)
    output = tmp_path / "out.mp4"
    db.save_settings({"transform_strategy": "fast", "preset": "veryfast"})
    try:
        result = jobs._run_desensitize(str(video), {"output": str(output)})
    finally:
        db.save_settings({"transform_strategy": "fast", "preset": "medium"})
    assert result["transform_strategy"] == "fast"
    assert result["preset"] == "veryfast"


@needs_ffmpeg
def test_desensitize_job_disables_metric_pass(monkeypatch, tmp_path):
    """任务队列清洗以 compute_metrics=False 调用，产物完成即返回。"""
    video = _video(tmp_path, seed=52)
    output = tmp_path / "out.mp4"
    captured: dict = {}

    def fake_run(path: str, out: str, **kwargs):
        captured.update(kwargs)
        return {
            "output": str(out),
            "transform_strategy": kwargs.get("transform_strategy", "fast"),
            "preset": kwargs.get("preset", "veryfast"),
            "frames": 8,
        }

    monkeypatch.setattr(services, "run_desensitize", fake_run)
    monkeypatch.setattr(services, "record_variant", lambda *args, **kwargs: {})
    result = jobs._run_desensitize(str(video), {"output": str(output)})
    assert result["output"] == str(output)
    assert captured.get("compute_metrics") is False
    assert captured.get("defer_metrics") is None, "任务队列不再启用延迟指标线程"


def test_pause_resume_lifecycle():
    job_id = uuid.uuid4().hex[:12]
    job = {
        "id": job_id,
        "name": "暂停测试",
        "status": "queued",
        "parallelism": 1,
        "created_at": time.time(),
        "tasks": [{
            "id": "t1",
            "kind": "detect",
            "path": "/tmp/x.mp4",
            "options": {},
            "status": "queued",
            "percent": 0,
            "result": None,
            "error": None,
        }],
    }
    db.save_job(job)
    queue = JobQueue()
    queue._jobs[job_id] = job
    queue.pause(job_id)
    assert queue._jobs[job_id]["status"] == "paused"
    queue.resume(job_id)
    assert queue._jobs[job_id]["status"] == "queued"
    db.delete_jobs("all")


def test_set_priority_requeues_job():
    job_id = uuid.uuid4().hex[:12]
    job = {
        "id": job_id,
        "name": "优先级测试",
        "status": "queued",
        "parallelism": 1,
        "priority": 0,
        "created_at": time.time(),
        "tasks": [],
    }
    db.save_job(job)
    queue = JobQueue()
    queue._jobs[job_id] = job
    updated = queue.set_priority(job_id, 5)
    assert updated is not None and updated["priority"] == 5
    db.delete_jobs("all")


@needs_ffmpeg
def test_desensitize_streams_video_in_chunks(monkeypatch, tmp_path):
    """长视频按分块流式处理，不再整片驻留内存，结果与总长度无关。"""
    frames = samples.make_video_frames(40, 160, 120, seed=3)
    source = tmp_path / "src.mp4"
    output = tmp_path / "out.mp4"
    ffmpeg.encode_video(frames, str(source), fps=30)
    # 强制小块，验证多块流式路径。
    monkeypatch.setattr(planning, "memory_budget_bytes", lambda: 16 * 1024**2)
    report = services.run_desensitize(
        str(source),
        str(output),
        reorder=False,
        speed=1.0,
        regrade=False,
        audio_remix=False,
        sharpness=False,
        color_restore=False,
        seed=1,
    )
    assert output.exists()
    decoded, _info = ffmpeg.decode_video(str(output))
    assert len(decoded) == 40
    assert report["frames"] == 40


@needs_ffmpeg
def test_detect_job_completes(client, tmp_path):
    path = _video(tmp_path)
    response = client.post(
        "/api/jobs",
        json={"name": "检测批次", "parallelism": 2, "tasks": [{"kind": "detect", "path": str(path)}]},
    )
    assert response.status_code == 202
    job = _wait(client, response.json()["id"])
    assert job["status"] == "done"
    assert job["tasks"][0]["status"] == "done"
    assert job["tasks"][0]["result"]["bitstream"]["score"] >= 0


@needs_ffmpeg
def test_failed_task_is_isolated(client, tmp_path):
    good = _video(tmp_path)
    response = client.post(
        "/api/jobs",
        json={
            "name": "混合批次",
            "parallelism": 2,
            "tasks": [
                {"kind": "detect", "path": str(good)},
                {"kind": "detect", "path": "/no/such/file.mp4"},
            ],
        },
    )
    job = _wait(client, response.json()["id"])
    assert job["status"] == "done"  # 单条失败不影响整体
    assert {task["status"] for task in job["tasks"]} == {"done", "failed"}


def test_cancel_returns_job(client):
    job = client.post(
        "/api/jobs",
        json={"name": "待取消", "parallelism": 1, "tasks": [{"kind": "detect", "path": "/x.mp4"}]},
    ).json()
    response = client.post(f"/api/jobs/{job['id']}/cancel")
    assert response.status_code == 200
    assert response.json()["id"] == job["id"]


def test_retry_failed_job(client):
    created = client.post(
        "/api/jobs",
        json={"name": "失败批次", "parallelism": 1, "tasks": [{"kind": "detect", "path": "/no.mp4"}]},
    ).json()
    _wait(client, created["id"])
    response = client.post(f"/api/jobs/{created['id']}/retry")
    assert response.status_code == 202
    retried = response.json()
    # 原地重试：保留原任务，不再新开一条，也不追加名字后缀。
    assert retried["id"] == created["id"]
    assert retried["name"] == "失败批次"
    assert len(retried["tasks"]) == 1
    assert retried["tasks"][0]["status"] == "queued"
