"""进程内事件广播：任务进度经 WebSocket 推送给前端。"""

from __future__ import annotations

import asyncio

# 每个订阅者最多保留最近若干条事件。任务事件都是完整状态快照，
# 慢客户端消费不过来时丢弃最旧的事件、保留最新状态，避免内存无界增长。
QUEUE_MAXSIZE = 16


class EventBroker:
    def __init__(self) -> None:
        self._queues: set[asyncio.Queue] = set()

    def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_MAXSIZE)
        self._queues.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self._queues.discard(queue)

    async def publish(self, event: dict) -> None:
        for queue in list(self._queues):
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            queue.put_nowait(event)


broker = EventBroker()
