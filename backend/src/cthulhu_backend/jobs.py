"""任务队列与调度：检测/清洗任务的排队、并发执行、进度事件、取消与失败隔离。"""

from __future__ import annotations

import asyncio
import os
import threading
import time
import uuid
from typing import Any

from cthulhu_backend import db, services
from cthulhu_backend.evaluate import dedup_harness
from cthulhu_backend.events import broker
from cthulhu_backend.transform import strategies
from cthulhu_backend.version import APP_VERSION


def _run_detect(path: str, options: dict, progress=None, stop=None, pause=None) -> dict:
    """暗水印检测：按阶段汇报进度，并支持在阶段边界及时响应取消。"""
    return services.run_detect(path, progress_cb=progress, should_stop=stop)


def _run_desensitize(path: str, options: dict, progress=None, stop=None, pause=None) -> dict:
    # 变换策略与编码档：任务选项优先，其次设置项，最后取模块默认。
    # 默认策略为 fast（经检测基准 A/B 验证与 thorough 信号等价，约快 2 倍）。
    # 结果中的 transform_strategy 由 services.run_desensitize 写回，随任务持久化。
    strategy_name = (
        options.get("transform_strategy")
        or db.load_settings().get("transform_strategy")
        or strategies.DEFAULT_STRATEGY
    )
    preset_name = options.get("preset") or db.load_settings().get("preset") or "veryfast"
    params = {
        key: value
        for key, value in options.items()
        if key not in {"output", "transform_strategy", "preset"}
    }
    # 编码加速策略：设置里的 GPU 选项控制 H.265 是否走 VideoToolbox 硬编；
    # 实测 H.264 硬编更慢且更大，因此始终软件编码。
    gpu_setting = db.load_settings().get("gpu", "仅 CPU")
    hardware = not gpu_setting.startswith("仅 CPU") and params.get("codec") == "libx265"
    snapshot = dict(params)
    snapshot["_preset"] = preset_name
    snapshot["_transform_strategy"] = strategy_name
    snapshot["_gpu"] = gpu_setting
    snapshot["_hardware"] = hardware
    snapshot["_app_version"] = APP_VERSION
    result = services.run_desensitize(
        path,
        options["output"],
        progress_cb=progress,
        should_stop=stop,
        hardware=hardware,
        transform_strategy=strategy_name,
        preset=preset_name,
        **params,
    )
    # 自动复检：对清洗产物跑盲检测，量化残留风险供界面反馈。
    try:
        if stop and stop():
            raise InterruptedError("任务已取消")
        post = services.run_blind(options["output"])
        blind = post or {}
        result["residual"] = {key: blind.get(key) for key in ("ss", "qim", "lsb", "echo")}
    except InterruptedError:
        raise
    except Exception:  # noqa: BLE001 - 复检失败不影响任务成功状态
        result["residual"] = None
    # 判重代理基准：量化清洗产物相对原片的逃逸效果，供界面校验区展示。
    try:
        dedup = dedup_harness.compare(path, options["output"])
        result["dedup"] = {
            "duplicate_risk": dedup["duplicate_risk"],
            "risk_level": dedup["risk_level"],
            "distances": dedup["distances"],
        }
    except Exception:  # noqa: BLE001 - 判重打分失败不影响任务成功
        result["dedup"] = None
    try:
        metrics = {
            "duplicate_risk": (result.get("dedup") or {}).get("duplicate_risk"),
            "psnr_db": result.get("psnr_db"),
            "ssim": result.get("ssim"),
            "stability_ratio": result.get("stability_ratio"),
        }
        services.record_variant(
            path,
            result.get("output") or os.path.expanduser(options["output"]),
            snapshot,
            seed=params.get("seed", 0),
            metrics=metrics,
        )
    except Exception:  # noqa: BLE001, S110 - 记录失败不影响任务结果
        pass
    return result


def _run_repair(path: str, options: dict, progress=None, stop=None, pause=None) -> dict:
    result = services.run_repair(
        path,
        options["output"],
        options.get("regions", []),
        options.get("crf", 23),
        progress_cb=progress,
        stop=stop,
        pause=pause,
    )
    try:
        services.record_variant(
            path,
            result.get("output") or os.path.expanduser(options["output"]),
            {
                "regions": options.get("regions", []),
                "crf": options.get("crf", 23),
                "_app_version": APP_VERSION,
            },
            seed=0,
        )
    except Exception:  # noqa: BLE001, S110 - 记录失败不影响任务结果
        pass
    return result


RUNNERS = {
    "detect": _run_detect,
    "desensitize": _run_desensitize,
    "repair": _run_repair,
}


class JobQueue:
    """进程内任务队列：单消费者分发 + 每任务信号量控制并行。"""

    def __init__(self, default_parallelism: int | None = None) -> None:
        if default_parallelism is None:
            default_parallelism = min(4, max(2, os.cpu_count() or 2))
        self.default_parallelism = default_parallelism
        self._jobs: dict[str, dict[str, Any]] = {}
        self._inbox: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._cancelled: set[str] = set()
        self._paused: set[str] = set()
        self._task_stop: dict[str, threading.Event] = {}
        self._task_pause: dict[str, threading.Event] = {}
        self._seq = 0
        self._worker: asyncio.Task | None = None

    def start(self) -> None:
        # worker 缺失或已随旧事件循环结束（如测试的多 portal 场景）时，重建队列与任务。
        if self._worker is None or self._worker.done():
            self._inbox = asyncio.PriorityQueue()
            self._worker = asyncio.create_task(self._run_worker())

    def _enqueue(self, job: dict) -> None:
        self._seq += 1
        job["enqueue_seq"] = self._seq
        self._inbox.put_nowait((job.get("priority", 0), self._seq, job["id"]))

    async def stop(self) -> None:
        if self._worker:
            try:
                self._worker.cancel()
                await self._worker
            except (asyncio.CancelledError, RuntimeError):
                pass
            self._worker = None

    def create_job(self, name: str, tasks: list[dict], parallelism: int | None = None) -> dict:
        now = time.time()
        job = {
            "id": uuid.uuid4().hex[:12],
            "name": name,
            "parallelism": parallelism or self.default_parallelism,
            "status": "queued",
            "priority": 0,
            "created_at": now,
            "tasks": [
                {
                    "id": uuid.uuid4().hex[:8],
                    "kind": task["kind"],
                    "path": task["path"],
                    "options": task.get("options", {}),
                    "status": "queued",
                    "percent": 0,
                    "progress_note": None,
                    "elapsed": None,
                    "result": None,
                    "error": None,
                }
                for task in tasks
            ],
        }
        self._jobs[job["id"]] = job
        db.save_job(self.public_view(job))
        self._enqueue(job)
        self.start()
        return job

    def get(self, job_id: str) -> dict | None:
        return self._jobs.get(job_id)

    def list(self) -> list[dict]:
        """返回任务清单，最新创建在前。"""
        return sorted(
            self._jobs.values(),
            key=lambda job: job.get("created_at", 0),
            reverse=True,
        )

    def cancel(self, job_id: str) -> dict | None:
        job = self._jobs.get(job_id)
        if not job:
            return None
        if job["status"] == "queued":
            job["status"] = "canceled"
            self._cancelled.add(job_id)
            db.save_job(self.public_view(job))
        elif job["status"] == "running":
            self._cancelled.add(job_id)
            for task in job["tasks"]:
                event = self._task_stop.get(task["id"])
                if event is not None:
                    event.set()
        return job

    def pause(self, job_id: str) -> dict | None:
        """暂停任务：运行中的任务停止当前视频处理后不再分发新项。"""
        job = self._jobs.get(job_id)
        if not job:
            return None
        if job["status"] in {"queued", "running", "paused"}:
            self._paused.add(job_id)
            if job["status"] == "queued":
                job["status"] = "paused"
                db.save_job(self.public_view(job))
            else:
                for task in job["tasks"]:
                    event = self._task_pause.get(task["id"])
                    if event is not None:
                        event.set()
        return job

    def resume(self, job_id: str) -> dict | None:
        job = self._jobs.get(job_id)
        if not job:
            return None
        if job_id in self._paused:
            self._paused.discard(job_id)
        for task in job["tasks"]:
            event = self._task_pause.get(task["id"])
            if event is not None:
                event.clear()
        if job["status"] == "paused":
            job["status"] = "queued"
            db.save_job(self.public_view(job))
            self._enqueue(job)
        return job

    def set_priority(self, job_id: str, priority: int) -> dict | None:
        """调整排队任务的优先级，数字越小越先执行。"""
        job = self._jobs.get(job_id)
        if not job or job["status"] != "queued":
            return None
        job["priority"] = priority
        self._enqueue(job)
        db.save_job(self.public_view(job))
        return job

    def retry(self, job_id: str) -> dict | None:
        job = self._jobs.get(job_id)
        if not job:
            return None
        if not any(task["status"] == "failed" for task in job["tasks"]):
            return None
        # 原地重试：失败子任务回到队列，保留原任务记录，不再新开一条任务。
        for task in job["tasks"]:
            if task["status"] == "failed":
                task["status"] = "queued"
                task["percent"] = 0
                task["progress_note"] = None
                task["elapsed"] = None
                task["error"] = None
                task["result"] = None
                task.pop("started_at", None)
        job["status"] = "queued"
        job.pop("enqueue_seq", None)
        db.save_job(self.public_view(job))
        self._enqueue(job)
        return job

    def restore(self) -> None:
        """启动时载入持久化任务，未完成任务自动重新入队续跑。"""
        for job in db.load_jobs():
            if job["status"] in {"queued", "running"}:
                for task in job["tasks"]:
                    if task["status"] in {"queued", "running"}:
                        task["status"] = "queued"
                        task["percent"] = 0
                        task["error"] = None
                job["status"] = "queued"
                if "续跑" not in job["name"]:
                    job["name"] = f"{job['name']} · 续跑"
                db.save_job(job)
                self._jobs[job["id"]] = job
                self._enqueue(job)
            else:
                self._jobs[job["id"]] = job

    def clear(self, scope: str = "all") -> int:
        """从内存移除任务并返回移除数量。"""
        if scope == "all":
            removed = len(self._jobs)
            self._jobs.clear()
            return removed
        finished = [
            job_id
            for job_id, job in self._jobs.items()
            if job["status"] in {"done", "failed", "canceled"}
        ]
        for job_id in finished:
            self._jobs.pop(job_id, None)
        return len(finished)

    async def _run_worker(self) -> None:
        while True:
            _, seq, job_id = await self._inbox.get()
            job = self._jobs.get(job_id)
            if job is None or job.get("enqueue_seq") != seq:
                continue
            await self._process(job_id)

    async def _process(self, job_id: str) -> None:
        job = self._jobs[job_id]
        if job["status"] != "queued":
            return
        if job_id in self._cancelled:
            job["status"] = "canceled"
            await self._publish_job(job)
            db.save_job(self.public_view(job))
            return
        if job_id in self._paused:
            return
        job["status"] = "running"
        await self._publish_job(job)
        db.save_job(self.public_view(job))

        semaphore = asyncio.Semaphore(job["parallelism"])
        loop = asyncio.get_running_loop()

        async def run_task(task: dict) -> None:
            stop = threading.Event()
            pause = threading.Event()
            self._task_stop[task["id"]] = stop
            self._task_pause[task["id"]] = pause
            try:
                async with semaphore:
                    if job_id in self._paused:
                        return
                    if job_id in self._cancelled or task["status"] != "queued":
                        return
                    task["status"] = "running"
                    task["started_at"] = time.time()
                    await self._publish_task(job, task)

                    def on_progress(percent: int, note: str) -> None:
                        task["percent"] = percent
                        task["progress_note"] = note
                        task["elapsed"] = round(
                            time.time() - task.get("started_at", time.time()), 1
                        )
                        asyncio.run_coroutine_threadsafe(self._publish_task(job, task), loop)

                    try:
                        runner = RUNNERS[task["kind"]]
                        result = await asyncio.to_thread(
                            runner, task["path"], task["options"], on_progress, stop.is_set, pause
                        )
                        task["result"] = result
                        task["status"] = "done"
                        task["percent"] = 100
                        task["progress_note"] = "完成"
                    except Exception as exc:  # noqa: BLE001 - 任务级失败隔离
                        if stop.is_set():
                            task["status"] = "canceled"
                            task["error"] = None
                            task["progress_note"] = "已取消"
                        else:
                            task["status"] = "failed"
                            task["error"] = str(exc)
                    task["elapsed"] = round(time.time() - task.get("started_at", time.time()), 1)
                    await self._publish_task(job, task)
            finally:
                self._task_stop.pop(task["id"], None)
                self._task_pause.pop(task["id"], None)

        await asyncio.gather(*(run_task(task) for task in job["tasks"]))
        if job_id in self._cancelled:
            job["status"] = "canceled"
        elif job_id in self._paused:
            job["status"] = "paused"
        else:
            job["status"] = "failed" if job["tasks"] and all(t["status"] == "failed" for t in job["tasks"]) else "done"
        db.save_job(self.public_view(job))
        await self._publish_job(job)

    async def _publish_job(self, job: dict) -> None:
        await broker.publish({"type": "job:state", "job": self.public_view(job)})

    async def _publish_task(self, job: dict, task: dict) -> None:
        await broker.publish(
            {
                "type": "task:state",
                "job_id": job["id"],
                "task_id": task["id"],
                "task": {
                    k: task.get(k)
                    for k in (
                        "id", "kind", "path", "options", "status", "percent",
                        "progress_note", "elapsed", "error", "result",
                    )
                },
            }
        )

    @staticmethod
    def public_view(job: dict) -> dict:
        return {
            "id": job["id"],
            "name": job["name"],
            "parallelism": job["parallelism"],
            "status": job["status"],
            "created_at": job["created_at"],
            "tasks": [
                {
                    k: task.get(k)
                    for k in (
                        "id", "kind", "path", "options", "status", "percent",
                        "progress_note", "elapsed", "error", "result",
                    )
                }
                for task in job["tasks"]
            ],
        }


job_queue = JobQueue()
