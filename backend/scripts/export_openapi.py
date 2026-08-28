"""离线导出 OpenAPI 文档，供前端生成请求类型并做 CI 漂移检查。"""

from __future__ import annotations

import json
import os
import pathlib

os.environ.setdefault("CTHULHU_AUTH_TOKEN", "openapi-export")

from cthulhu_backend.main import app  # noqa: E402

TARGET = pathlib.Path(__file__).resolve().parents[1] / "openapi.json"


def main() -> None:
    spec = app.openapi()
    TARGET.write_text(
        json.dumps(spec, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"已写入 {TARGET}")


if __name__ == "__main__":
    main()
