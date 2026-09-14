"""应用版本号：产物快照与健康接口共用。

唯一真值：根/ui/electron 的 package.json 与 backend/pyproject.toml 里的 version
由 `backend/scripts/sync_version.py` 从这里派生，`backend/tests/test_version.py`
每次测试校验一致性。
"""

APP_VERSION = "0.6.3"
