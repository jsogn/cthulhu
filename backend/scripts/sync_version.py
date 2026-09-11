"""版本号同步：`cthulhu_backend/version.py` 是唯一真值，其余位置由本脚本派生。

用法：
    uv run --project backend python backend/scripts/sync_version.py           # 写入
    uv run --project backend python backend/scripts/sync_version.py --check   # 只校验（测试/CI）

为什么要有它：版本号原先手写在 5 处（version.py、backend/pyproject.toml、
根/ui/electron 三个 package.json），漏改一处就会出现「包名 0.6.1、健康接口还是
0.6.0」这种自相矛盾的产物（审计 R3 的同类问题）。现在以 version.py 为准，
`backend/tests/test_version.py` 会在每次测试时校验一致性。
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VERSION_FILE = ROOT / "backend" / "src" / "cthulhu_backend" / "version.py"
TARGETS = (
    ROOT / "package.json",
    ROOT / "ui" / "package.json",
    ROOT / "electron" / "package.json",
    ROOT / "backend" / "pyproject.toml",
)

_VERSION_PY = re.compile(r'^APP_VERSION\s*=\s*"([^"]+)"', re.MULTILINE)
_JSON_FIELD = re.compile(r'^(\s*"version"\s*:\s*")([^"]+)(")', re.MULTILINE)
_TOML_FIELD = re.compile(r'^(version\s*=\s*")([^"]+)(")', re.MULTILINE)


def read_version() -> str:
    """唯一真值：version.py 里的 APP_VERSION。"""
    match = _VERSION_PY.search(VERSION_FILE.read_text(encoding="utf-8"))
    if match is None:
        raise SystemExit(f"{VERSION_FILE} 里找不到 APP_VERSION")
    return match.group(1)


def _field(path: Path) -> re.Pattern[str]:
    return _TOML_FIELD if path.suffix == ".toml" else _JSON_FIELD


def current_version(path: Path) -> str:
    match = _field(path).search(path.read_text(encoding="utf-8"))
    if match is None:
        raise SystemExit(f"{path} 里找不到 version 字段")
    return match.group(2)


def main() -> None:
    parser = argparse.ArgumentParser(description="同步/校验各清单文件里的版本号")
    parser.add_argument("--check", action="store_true", help="只校验，不写文件")
    args = parser.parse_args()

    version = read_version()
    drift: list[str] = []
    for path in TARGETS:
        pattern = _field(path)
        text = path.read_text(encoding="utf-8")
        found = current_version(path)
        if found == version:
            continue
        rel = path.relative_to(ROOT)
        if args.check:
            drift.append(f"{rel}: {found} != {version}")
            continue
        path.write_text(pattern.sub(rf"\g<1>{version}\g<3>", text, count=1), encoding="utf-8")
        print(f"已更新 {rel}：{found} → {version}")

    if args.check:
        if drift:
            raise SystemExit("版本号不一致（以 version.py 为准）：\n  " + "\n  ".join(drift))
        print(f"版本号一致：{version}")
    elif not drift:
        print(f"版本号已一致：{version}")


if __name__ == "__main__":
    main()
