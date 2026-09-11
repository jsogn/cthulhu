"""分层回归：生产侧不得反向依赖评估层。

审计 R5：`transform` / `similarity` 曾引用 `evaluate.metrics`，与
`evaluate → fingerprint → similarity` 形成包级环，只能靠函数内延迟导入破环。
指标原语下沉到中立层 `quality` 后方向恢复单向；本测试防止回退——包括把导入
藏进函数体的写法（历史上正是这么破环的）。
"""

from __future__ import annotations

import ast
from pathlib import Path

BACKEND_SRC = Path(__file__).resolve().parents[1] / "src" / "cthulhu_backend"
# 生产/领域侧包：允许依赖 quality/watermark/media 等下层，不得依赖 evaluate。
LOWER_LAYER_PACKAGES = ("transform", "similarity", "fingerprint", "watermark")


def _imports_package(path: Path, package: str) -> list[int]:
    """返回该文件里 import `cthulhu_backend.<package>` 的行号（含函数内延迟导入）。"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    lines: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == f"cthulhu_backend.{package}":
            lines.append(node.lineno)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0:2] == ["cthulhu_backend", package]:
                    lines.append(node.lineno)
    return lines


def test_lower_layers_do_not_import_evaluate() -> None:
    offenders: list[str] = []
    for package in LOWER_LAYER_PACKAGES:
        for path in sorted((BACKEND_SRC / package).rglob("*.py")):
            for line in _imports_package(path, "evaluate"):
                offenders.append(f"{path.relative_to(BACKEND_SRC)}:{line}")
    assert not offenders, (
        "生产侧不得依赖评估层 evaluate（指标原语请用中立层 quality）："
        + ", ".join(offenders)
    )
