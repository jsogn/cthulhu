"""进程内事件广播：任务进度经 WebSocket 推送给前端。"""

from __future__ import annotations

import asyncio


class EventBroker:
    def __init__(self) -> None:
        self._queues: set[asyncio.Queue] = set()

    def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        self._queues.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self._queues.discard(queue)

    async def publish(self, event: dict) -> None:
        for queue in list(self._queues):
            queue.put_nowait(event)


broker = EventBroker()
