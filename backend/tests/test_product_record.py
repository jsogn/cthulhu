"""产物参数记录：写入、读取与删除联动。"""

from __future__ import annotations

import json
from pathlib import Path

from cthulhu_backend import services


def test_product_record_roundtrip(tmp_path):
    output = tmp_path / "demo_cleaned_1.mp4"
    output.write_bytes(b"x")
    meta = services.write_product_record(
        str(output),
        "/src/demo.mp4",
        {"speed": 0.955, "seed": 7, "denoise": True},
        {"ssim": 0.8, "duplicate_risk": 0.3},
    )
    data = json.loads(Path(meta).read_text(encoding="utf-8"))
    assert data["source"] == "/src/demo.mp4"
    assert data["options"]["seed"] == 7
    assert data["metrics"]["ssim"] == 0.8


def test_delete_output_removes_record(tmp_path):
    output = tmp_path / "demo_cleaned_1.mp4"
    output.write_bytes(b"x")
    services.write_product_record(str(output), "/src/demo.mp4", {})
    assert services.delete_output(str(output)) is True
    assert not Path(str(output) + ".meta.json").exists()
