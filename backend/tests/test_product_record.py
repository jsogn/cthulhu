"""产物记录（variants 数据表）：写入、查询、删除联动。"""

from __future__ import annotations

from cthulhu_backend import db, services


def test_variant_roundtrip_and_filters():
    db.init_db()
    record = services.record_variant(
        "/src/demo.mp4",
        "/out/demo_cleaned_1.mp4",
        {"speed": 0.955, "seed": 7, "denoise": True},
        seed=7,
        metrics={"duplicate_risk": 0.3},
    )
    rows = db.list_variants(source="/src/demo.mp4")
    assert len(rows) == 1
    assert rows[0]["id"] == record["id"]
    assert rows[0]["options"]["seed"] == 7
    assert rows[0]["metrics"]["duplicate_risk"] == 0.3
    db.delete_variant_by_output("/out/demo_cleaned_1.mp4")


def test_delete_output_removes_variant(tmp_path):
    db.init_db()
    output = tmp_path / "demo_cleaned_1.mp4"
    output.write_bytes(b"x")
    services.record_variant(
        "/src/demo.mp4",
        str(output),
        {"speed": 1.0},
        seed=0,
    )
    assert services.delete_output(str(output)) is True
    assert db.list_variants(source="/src/demo.mp4") == []
