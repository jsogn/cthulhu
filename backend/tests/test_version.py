"""版本号一致性：`version.py` 是唯一真值，其余清单由脚本派生（审计 R3 同类问题）。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SYNC_SCRIPT = ROOT / "backend" / "scripts" / "sync_version.py"


def test_version_sources_are_in_sync() -> None:
    result = subprocess.run(
        [sys.executable, str(SYNC_SCRIPT), "--check"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        "版本号在多处漂移，请以 version.py 为准执行 "
        f"`python backend/scripts/sync_version.py`：\n{result.stdout}{result.stderr}"
    )
