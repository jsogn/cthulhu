"""任务队列与调度：检测/清洗任务的排队、并发执行、进度事件、取消与失败隔离。"""

from __future__ import annotations

import asyncio
import os
import threading
import time
import uuid
from typing import Any

from cthulhu_backend import db, services
from cthulhu_backend.events import broker
from cthulhu_backend.transform import strategies
from cthulhu_backend.version import APP_VERSION


def _run_detect(path: str, options: dict, progress=None, stop=None, pause=None) -> dict:
    """暗水印检测：按阶段汇报进度，并支持在阶段边界及时响应取消。"""
    return services.run_detect(path, progress_cb=progress, should_stop=stop, pause=pause)


def _run_desensitize(
    path: str,
    options: dict,
    progress=None,
    stop=None,
    pause=None,
    metrics_gate=None,
) -> dict:
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
        # GUI 产物不需要画质/相似度指标回填；关闭后任务完成即产物就绪，
        # 不再触发指标遍的内存尖峰。runner 仍保留 metrics_gate 形参以兼容
        # 调度器注入（由 services.defer_metrics=True 的直连调用方使用）。
        compute_metrics=False,
        progress_cb=progress,
        should_stop=stop,
        hardware=hardware,
        transform_strategy=strategy_name,
        preset=preset_name,
        pause=pause,
        **params,
    )
    # 产物记录只保存参数快照，画质/相似度指标不再自动回填。
    try:
        services.record_variant(
            path,
            result.get("output") or os.path.expanduser(options["output"]),
            snapshot,
            seed=params.get("seed", 0),
        )
    except Exception:  # noqa: BLE001, S110 - 记录失败不影响任务结果
        pass
    return result


RUNNERS = {
    "detect": _run_detect,
    "desensitize": _run_desensitize,
}

# 自动续跑上限：同一子任务累计被进程中断这么多次后不再自动重跑，避免
# 「一跑就崩」的素材把引擎拖进「崩溃 → 重启 → 重跑」的死循环。
MAX_AUTO_RESUME = 3


def _resume_note(count_interruptions: bool) -> str:
    """重启续跑的可见说明，让用户知道这条任务是重启后自动接回来的。"""
    return "进程重启，已自动续跑" if count_interruptions else "重启后待继续"


class JobQueue:
    """进程内任务队列：单消费者分发 + 全局并发闸门统一调度。

    所有任务（检测/清洗/修复、单条/批量）共用同一并发额度，按系统资源
    （设置里的「同时处理任务数」）统一调度，避免批量任务互相抢占资源。
    """

    def __init__(self, default_parallelism: int | None = None) -> None:
        if default_parallelism is None:
            default_parallelism = 2
        self.default_parallelism = default_parallelism
        self._jobs: dict[str, dict[str, Any]] = {}
        self._inbox: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._gate: asyncio.Condition | None = None
        self._active_slots = 0
        self._cancelled: set[str] = set()
        self._paused: set[str] = set()
        self._task_stop: dict[str, threading.Event] = {}
        self._task_pause: dict[str, threading.Event] = {}
        self._seq = 0
        self._worker: asyncio.Task | None = None
        self._proc_tasks: set[asyncio.Task] = set()

    def start(self) -> None:
        # worker 缺失或已随旧事件循环结束（如测试的多 portal 场景）时，重建队列与任务。
        if self._worker is None or self._worker.done():
            # 重建队列前先把尚未消费的排队项搬过去，避免已入队任务被静默丢弃。
            pending: list[tuple[int, int, str]] = []
            while not self._inbox.empty():
                try:
                    pending.append(self._inbox.get_nowait())
                except asyncio.QueueEmpty:
                    break
            self._inbox = asyncio.PriorityQueue()
            for item in pending:
                self._inbox.put_nowait(item)
            self._gate = asyncio.Condition()
            self._active_slots = 0
            self._worker = asyncio.create_task(self._run_worker())

    def _slot_capacity(self) -> int:
        """全局并发额度：读取设置（1~4），未配置时取默认值。"""
        try:
            value = int(db.load_settings().get("parallelism", self.default_parallelism))
        except (TypeError, ValueError):
            value = self.default_parallelism
        return min(4, max(1, value))

    def is_idle(self) -> bool:
        """当前是否有任务占用并发槽：空闲时后台指标遍才允许启动。"""
        return self._active_slots == 0

    async def _acquire_slot(self) -> None:
        gate = self._gate
        if gate is None:
            return
        async with gate:
            while self._active_slots >= self._slot_capacity():
                await gate.wait()
            self._active_slots += 1

    async def _release_slot(self) -> None:
        gate = self._gate
        if gate is None:
            return
        async with gate:
            self._active_slots = max(0, self._active_slots - 1)
            gate.notify_all()

    def _enqueue(self, job: dict) -> None:
        self._seq += 1
        job["enqueue_seq"] = self._seq
        self._inbox.put_nowait((job.get("priority", 0), self._seq, job["id"]))

    async def stop(self) -> None:
        # 先取消在途的任务处理协程，再停调度循环；避免退出时遗留
        # 无法收口的 _process 协程继续占着并发闸门或写数据库。
        pending = [task for task in self._proc_tasks if not task.done()]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
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
        """暂停任务：运行中的任务在下一个分块边界停住，排队项不再分发。"""
        job = self._jobs.get(job_id)
        if not job:
            return None
        if job["status"] in {"queued", "running", "paused"}:
            self._paused.add(job_id)
            if job["status"] in {"queued", "running"}:
                job["status"] = "paused"
                db.save_job(self.public_view(job))
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
        # 仍有运行中的任务说明处理协程还活着，只需解除阻塞、恢复运行态；
        # 完全排队的任务才需要重新入队分发。
        if any(task.get("status") == "running" for task in job["tasks"]):
            job["status"] = "running"
            db.save_job(self.public_view(job))
        elif job["status"] == "paused":
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
                # 手动重试重新给一份自动续跑预算，不被此前的重启次数卡住。
                task.pop("resume_count", None)
        job["status"] = "queued"
        job.pop("enqueue_seq", None)
        db.save_job(self.public_view(job))
        self._enqueue(job)
        return job

    def restore(self) -> None:
        """启动时载入持久化任务：未完成的自动续跑，只有反复中断才交回人工。

        引擎重启（用户退出 / 引擎崩溃 / 看门狗强制重启）不再让整批任务报废：
        排队中的任务本来就没开始，直接回到队列；运行中的任务整条重跑（管线
        没有分段断点），并累计中断次数，达到上限后标记失败等待手动重试。
        暂停中的任务保持暂停，只把中断的子任务退回队列，等用户点「继续」。
        """
        for job in db.load_jobs():
            if job["status"] == "paused":
                # 暂停是用户意图，重启后继续暂停；但中断的子任务必须退回队列，
                # 否则「继续」只会把状态改成运行态、实际没有任何任务在跑。
                self._requeue_interrupted_tasks(job, count_interruptions=False)
                self._jobs[job["id"]] = job
                db.save_job(self.public_view(job))
                continue
            if job["status"] not in {"queued", "running"}:
                self._jobs[job["id"]] = job
                continue
            self._requeue_interrupted_tasks(job, count_interruptions=True)
            if any(task["status"] == "queued" for task in job["tasks"]):
                job["status"] = "queued"
                self._jobs[job["id"]] = job
                db.save_job(self.public_view(job))
                self._enqueue(job)
            else:
                # 未完成项都撞上了续跑上限：整单交给人工，不再自动重跑。
                job["status"] = self._derive_job_status(job["tasks"])
                self._jobs[job["id"]] = job
                db.save_job(self.public_view(job))

    def _requeue_interrupted_tasks(self, job: dict, *, count_interruptions: bool) -> None:
        """把中断的子任务退回队列，交回调度器重新分发。

        运行中的任务没有分段断点，只能整条重跑；因此这里累计中断次数，
        超过 MAX_AUTO_RESUME 次仍被中断的任务标记失败并提示手动重试。
        """
        for task in job["tasks"]:
            if task["status"] not in {"queued", "running"}:
                continue
            interrupted = task["status"] == "running"
            if interrupted and count_interruptions:
                task["resume_count"] = int(task.get("resume_count") or 0) + 1
            if int(task.get("resume_count") or 0) > MAX_AUTO_RESUME:
                task["status"] = "failed"
                task["percent"] = 0
                task["progress_note"] = "进程重启中断"
                task["error"] = (
                    f"进程重启中断，已自动续跑 {MAX_AUTO_RESUME} 次仍未完成；请手动重试"
                )
                continue
            task["status"] = "queued"
            task["percent"] = 0
            task["elapsed"] = None
            task["result"] = None
            task["error"] = None
            task.pop("started_at", None)
            # 只有真正被打断过（或已经续跑过）的任务才需要重启说明；
            # 本来就在排队的任务没丢任何进度，保持干净。
            was_interrupted = interrupted or int(task.get("resume_count") or 0) > 0
            task["progress_note"] = (
                _resume_note(count_interruptions) if was_interrupted else None
            )

    def clear(self, scope: str = "all") -> int:
        """从内存移除任务并返回移除数量。"""
        if scope == "all":
            removed = len(self._jobs)
            self._jobs.clear()
            # 同步清理取消/暂停标记，避免任务已移除后标记集合持续累积。
            self._cancelled.clear()
            self._paused.clear()
            return removed
        finished = [
            job_id
            for job_id, job in self._jobs.items()
            if job["status"] in {"done", "failed", "canceled"}
        ]
        for job_id in finished:
            self._jobs.pop(job_id, None)
            self._cancelled.discard(job_id)
            self._paused.discard(job_id)
        return len(finished)

    async def _run_worker(self) -> None:
        while True:
            try:
                _, seq, job_id = await self._inbox.get()
                job = self._jobs.get(job_id)
                if job is None or job.get("enqueue_seq") != seq:
                    continue
                # 并发处理多个 job：真正的并行度由全局闸门统一控制。
                proc_task = asyncio.create_task(self._process(job_id))
                self._proc_tasks.add(proc_task)
                proc_task.add_done_callback(self._proc_tasks.discard)
            except asyncio.CancelledError:
                raise
            except RuntimeError:
                # 事件循环已关闭，调度循环随之终止。
                raise
            except Exception as exc:  # noqa: BLE001 - 单次分发失败不能杀死调度循环
                print(f"[jobs] 任务分发异常（已跳过，继续调度）：{exc}")

    async def _process(self, job_id: str) -> None:
        try:
            await self._process_inner(job_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - 调度/持久化异常不允许任务卡死在运行态
            await self._fail_job(job_id, f"调度异常：{exc}")

    async def _process_inner(self, job_id: str) -> None:
        job = self._jobs[job_id]
        if job["status"] != "queued":
            return
        if job_id in self._cancelled:
            job["status"] = "canceled"
            await self._publish_job(job)
            db.save_job(self.public_view(job))
            self._cancelled.discard(job_id)
            return
        if job_id in self._paused:
            return
        job["status"] = "running"
        await self._publish_job(job)
        db.save_job(self.public_view(job))

        loop = asyncio.get_running_loop()

        async def run_task(task: dict) -> None:
            stop = threading.Event()
            pause = threading.Event()
            self._task_stop[task["id"]] = stop
            self._task_pause[task["id"]] = pause
            try:
                await self._acquire_slot()
                try:
                    if job_id in self._paused:
                        return
                    if job_id in self._cancelled or task["status"] != "queued":
                        return
                    task["status"] = "running"
                    task["started_at"] = time.time()
                    # 子任务状态变化即落库：进程重启后靠这份快照跳过已完成项、
                    # 只重跑真正被打断的那几条，而不是整批从头再来。
                    self._persist(job)
                    await self._publish_task(job, task)

                    def on_progress(percent: int, note: str) -> None:
                        task["percent"] = percent
                        task["progress_note"] = note
                        task["elapsed"] = round(
                            time.time() - task.get("started_at", time.time()), 1
                        )
                        try:
                            asyncio.run_coroutine_threadsafe(
                                self._publish_task(job, task), loop
                            )
                        except RuntimeError:
                            # 事件循环已关闭（进程退出中），丢弃该进度事件即可。
                            pass

                    try:
                        runner = RUNNERS[task["kind"]]
                        runner_args = [task["path"], task["options"], on_progress, stop.is_set, pause]
                        if task["kind"] == "desensitize":
                            # 空闲闸门由调度器注入，避免 runner 反向依赖全局单例。
                            runner_args.append(self.is_idle)
                        result = await asyncio.to_thread(runner, *runner_args)
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
                    self._persist(job)
                    await self._publish_task(job, task)
                finally:
                    await self._release_slot()
            finally:
                self._task_stop.pop(task["id"], None)
                self._task_pause.pop(task["id"], None)

        await asyncio.gather(*(run_task(task) for task in job["tasks"]))
        if job_id in self._cancelled:
            job["status"] = "canceled"
        elif job_id in self._paused:
            job["status"] = "paused"
        else:
            job["status"] = self._derive_job_status(job["tasks"])
        db.save_job(self.public_view(job))
        await self._publish_job(job)
        # 任务到达终态后清理标记，防止长期运行中集合随取消/暂停操作无界增长。
        self._cancelled.discard(job_id)
        if job["status"] != "paused":
            self._paused.discard(job_id)

    async def _fail_job(self, job_id: str, error: str) -> None:
        """把中途异常的任务标记为失败并广播，避免任务永远停留在「执行中」。"""
        job = self._jobs.get(job_id)
        if job is None or job["status"] not in {"queued", "running"}:
            return
        for task in job["tasks"]:
            if task["status"] in {"queued", "running"}:
                task["status"] = "failed"
                task["error"] = error
                task["progress_note"] = "处理中断"
        # 与正常收尾路径共用同一终态规则：单条失败不影响整单，全部失败才算失败。
        job["status"] = self._derive_job_status(job["tasks"])
        self._persist(job)
        await self._publish_job(job)
        self._cancelled.discard(job_id)
        self._paused.discard(job_id)

    def _persist(self, job: dict) -> None:
        """落库任务快照；写库失败只告警，不影响正在跑的批次。"""
        try:
            db.save_job(self.public_view(job))
        except Exception as exc:  # noqa: BLE001 - 持久化失败不掩盖原始错误
            print(f"[jobs] 任务状态持久化失败（可忽略）：{exc}")

    @staticmethod
    def _derive_job_status(tasks: list[dict]) -> str:
        """统一的任务单终态规则：全部子任务失败才算整单失败。"""
        if tasks and all(task["status"] == "failed" for task in tasks):
            return "failed"
        return "done"

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
                        "progress_note", "elapsed", "error", "result", "resume_count",
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
                        "progress_note", "elapsed", "error", "result", "resume_count",
                    )
                }
                for task in job["tasks"]
            ],
        }


job_queue = JobQueue()
