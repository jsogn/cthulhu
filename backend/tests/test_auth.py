"""本地 API 鉴权测试：无令牌一律拒绝，健康检查保持开放。"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from cthulhu_backend.main import app


@pytest.fixture()
def client():
    # 用上下文管理器触发 lifespan（初始化 SQLite），鉴权测试才能触达业务路由。
    with TestClient(app) as test_client:
        yield test_client


def test_api_rejects_missing_or_wrong_token(client):
    assert client.get("/api/settings").status_code == 401
    assert client.get("/api/settings", headers={"X-CTHULHU-Token": "wrong-token"}).status_code == 401


def test_api_accepts_header_and_query_token(client):
    assert client.get("/api/settings", headers={"X-CTHULHU-Token": "test-token"}).status_code == 200
    assert client.get("/api/settings?token=test-token").status_code == 200


def test_health_is_open(client):
    assert client.get("/api/health").status_code == 200


def test_websocket_rejects_missing_token(client):
    with pytest.raises(WebSocketDisconnect), client.websocket_connect("/ws/events") as ws:
        ws.receive_json()
