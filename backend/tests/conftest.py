"""测试会话：隔离 SQLite 数据文件，避免污染真实 data/。"""

from __future__ import annotations

import os
import tempfile

os.environ["CTHULHU_DB"] = os.path.join(tempfile.mkdtemp(prefix="cthulhu-test-"), "test.db")
os.environ["CTHULHU_AUTH_TOKEN"] = "test-token"
