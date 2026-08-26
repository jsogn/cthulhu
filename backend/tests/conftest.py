"""测试会话：隔离 SQLite 数据文件，避免污染真实 data/。"""

from __future__ import annotations

import os
import tempfile

_test_root = tempfile.mkdtemp(prefix="cthulhu-test-")
os.environ["CTHULHU_DB"] = os.path.join(_test_root, "test.db")
os.environ["CTHULHU_THUMB_CACHE"] = os.path.join(_test_root, "thumb-cache")
os.environ["CTHULHU_AUTH_TOKEN"] = "test-token"


def pytest_configure(config):
    """测试会话初始化隔离数据库 schema，避免直接调用服务层时缺表。"""
    from cthulhu_backend import db

    db.init_db()
