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


def _task(task_id: str, status: str, path: str = "/tmp/x.mp4", **extra) -> dict:
    """构造一条持久化过的子任务记录。"""
    return {
        "id": task_id,
        "kind": "detect",
        "path": path,
        "options": {},
        "status": status,
        "percent": 30 if status == "running" else 0,
        "progress_note": None,
        "elapsed": None,
        "result": None,
        "error": None,
        **extra,
    }


def _save_job(job_id: str, tasks: list[dict], status: str = "running") -> None:
    db.save_job({
        "id": job_id,
        "name": "重启续跑",
        "status": status,
        "parallelism": 1,
        "created_at": time.time(),
        "tasks": tasks,
    })


@pytest.fixture
def fast_runners(monkeypatch):
    """用假 runner 顶替真实检测/清洗，让续跑语义在毫秒级可验证。"""
    calls: list[str] = []

    def fake_runner(path, options, progress=None, stop=None, pause=None):
        calls.append(path)
        return {"path": path}

    monkeypatch.setitem(jobs.RUNNERS, "detect", fake_runner)
    return calls


async def _wait_job(queue: JobQueue, job_id: str, timeout: float = 10.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = queue.get(job_id)
        if job and job["status"] in {"done", "failed", "canceled"}:
            return job
        await asyncio.sleep(0.02)
    raise TimeoutError("任务未在期限内结束")


def test_restore_requeues_interrupted_task_and_keeps_done(fast_runners):
    """进程重启后只重跑被打断的子任务，已完成项保持完成。"""
    job_id = uuid.uuid4().hex[:12]
    _save_job(job_id, [_task("t1", "done", "/tmp/a.mp4"), _task("t2", "running", "/tmp/b.mp4")])
    try:
        queue = JobQueue()
        queue.restore()
        restored = queue.get(job_id)
    finally:
        db.delete_jobs("all")
    assert restored["status"] == "queued"
    assert [task["status"] for task in restored["tasks"]] == ["done", "queued"]
    interrupted = restored["tasks"][1]
    assert interrupted["percent"] == 0
    assert interrupted["error"] is None
    assert interrupted["resume_count"] == 1
    assert "自动续跑" in interrupted["progress_note"]


def test_restore_keeps_untouched_queued_task_clean():
    """从没开始的排队任务不算中断，重启后直接继续且不留下重跑痕迹。"""
    job_id = uuid.uuid4().hex[:12]
    _save_job(job_id, [_task("t1", "queued", "/tmp/a.mp4")])
    try:
        queue = JobQueue()
        queue.restore()
        task = queue.get(job_id)["tasks"][0]
    finally:
        db.delete_jobs("all")
    assert task["status"] == "queued"
    assert "resume_count" not in task
    assert task["progress_note"] is None


def test_restore_continues_interrupted_job_to_done(fast_runners):
    """restore 之后调度器把中断的任务接着跑完，不再整批报废。"""
    job_id = uuid.uuid4().hex[:12]
    _save_job(job_id, [_task("t1", "running", "/tmp/a.mp4"), _task("t2", "queued", "/tmp/b.mp4")])
    queue = JobQueue()

    async def drive():
        # 与 lifespan 相同的顺序：先启动调度、再恢复未完成任务。
        queue.start()
        queue.restore()
        try:
            return await _wait_job(queue, job_id)
        finally:
            await queue.stop()

    try:
        job = asyncio.run(drive())
    finally:
        db.delete_jobs("all")
    assert job["status"] == "done"
    assert [task["status"] for task in job["tasks"]] == ["done", "done"]
    assert sorted(fast_runners) == ["/tmp/a.mp4", "/tmp/b.mp4"]


def test_restore_stops_auto_resume_after_limit():
    """反复中断的素材不再自动重跑，标记失败交回手动重试。"""
    job_id = uuid.uuid4().hex[:12]
    _save_job(job_id, [_task("t1", "running", resume_count=jobs.MAX_AUTO_RESUME)])
    try:
        queue = JobQueue()
        queue.restore()
        job = queue.get(job_id)
        dispatched = not queue._inbox.empty()
    finally:
        db.delete_jobs("all")
    assert job["status"] == "failed"
    assert job["tasks"][0]["status"] == "failed"
    assert "手动重试" in job["tasks"][0]["error"]
    assert not dispatched, "达到续跑上限的任务不应再自动重跑"


def test_restore_keeps_paused_job_paused_and_resumable(fast_runners):
    """重启后暂停态保留，但中断的子任务退回队列，点「继续」能真的跑起来。"""
    job_id = uuid.uuid4().hex[:12]
    _save_job(job_id, [_task("t1", "running", "/tmp/a.mp4")], status="paused")
    queue = JobQueue()

    async def drive():
        queue.start()
        queue.restore()
        try:
            paused = queue.get(job_id)
            assert paused["status"] == "paused"
            assert paused["tasks"][0]["status"] == "queued"
            queue.resume(job_id)
            return await _wait_job(queue, job_id)
        finally:
            await queue.stop()

    try:
        job = asyncio.run(drive())
    finally:
        db.delete_jobs("all")
    assert job["status"] == "done"
    assert fast_runners == ["/tmp/a.mp4"]


def test_retry_clears_resume_budget():
    """手动重试重新给一份自动续跑预算，不被此前的重启次数卡住。"""
    job_id = uuid.uuid4().hex[:12]
    _save_job(
        job_id,
        [_task("t1", "failed", error="进程重启中断", resume_count=jobs.MAX_AUTO_RESUME)],
        status="failed",
    )
    try:
        queue = JobQueue()
        queue.restore()
        job = queue.retry(job_id)
    finally:
        db.delete_jobs("all")
    assert job is not None
    assert job["tasks"][0]["status"] == "queued"
    assert job["tasks"][0]["error"] is None
    assert "resume_count" not in job["tasks"][0]


def test_restore_enters_safe_mode_after_repeated_interruptions():
    """同一个任务被打断两次以上时，本次重启降为单任务续跑。"""
    job_id = uuid.uuid4().hex[:12]
    _save_job(job_id, [_task("t1", "running", resume_count=jobs.SAFE_MODE_AFTER - 1)])
    try:
        queue = JobQueue()
        queue.restore()
        job = queue.get(job_id)
        capacity = queue._slot_capacity()
    finally:
        db.delete_jobs("all")
    assert job["tasks"][0]["resume_count"] == jobs.SAFE_MODE_AFTER
    assert queue.safe_mode is True
    assert capacity == 1, "反复被中断说明机器扛不住当前并发，应降为单任务"
    assert "本次降为单任务续跑" in job["tasks"][0]["progress_note"]


def test_restore_single_interruption_keeps_normal_concurrency():
    """只被打断过一次的批次不降并发，避免白丢吞吐。"""
    job_id = uuid.uuid4().hex[:12]
    _save_job(job_id, [_task("t1", "running")])
    try:
        queue = JobQueue()
        queue.restore()
        task = queue.get(job_id)["tasks"][0]
    finally:
        db.delete_jobs("all")
    assert task["resume_count"] == 1
    assert queue.safe_mode is False
    assert "单任务续跑" not in task["progress_note"]


def test_restore_note_keeps_engine_restart_cause(monkeypatch):
    """续跑说明里带上守护进程给的重启原因，现场反馈才能一眼看懂。"""
    monkeypatch.setenv("CTHULHU_RESTART_CAUSE", "引擎无响应 200 秒且任务无进展")
    job_id = uuid.uuid4().hex[:12]
    _save_job(job_id, [_task("t1", "running", "/tmp/a.mp4")])
    try:
        queue = JobQueue()
        queue.restore()
        task = queue.get(job_id)["tasks"][0]
    finally:
        db.delete_jobs("all")
    assert "进程重启，已自动续跑" in task["progress_note"]
    assert "引擎无响应 200 秒且任务无进展" in task["progress_note"]


def test_restore_degrades_purify_device_after_abnormal_exit(monkeypatch):
    """上次是原生崩溃（异常退出）时，净化推理降级到 CPU 再续跑。"""
    monkeypatch.setenv("CTHULHU_RESTART_ABNORMAL", "1")
    monkeypatch.setattr(jobs.purify, "device_preference", lambda: "auto")
    degraded: list[str] = []
    monkeypatch.setattr(jobs.purify, "set_device_preference", degraded.append)
    job_id = uuid.uuid4().hex[:12]
    _save_job(job_id, [_task("t1", "running", "/tmp/a.mp4")])
    try:
        queue = JobQueue()
        queue.restore()
    finally:
        db.delete_jobs("all")
    assert degraded == ["cpu"], "异常退出后应把推理设备降到 CPU"
    assert queue.safe_mode is False, "首次崩溃只降设备，不降并发"
    assert "推理已降级 CPU" in queue.get(job_id)["tasks"][0]["progress_note"]


def test_restore_keeps_device_when_nothing_unfinished(monkeypatch):
    """已经跑完的任务不该因为一次异常退出就改设备偏好。"""
    monkeypatch.setenv("CTHULHU_RESTART_ABNORMAL", "1")
    degraded: list[str] = []
    monkeypatch.setattr(jobs.purify, "set_device_preference", degraded.append)
    job_id = uuid.uuid4().hex[:12]
    _save_job(job_id, [_task("t1", "done", "/tmp/a.mp4")], status="done")
    try:
        queue = JobQueue()
        queue.restore()
    finally:
        db.delete_jobs("all")
    assert degraded == []
    assert queue.get(job_id)["status"] == "done"


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
