"""预置模板：首次初始化写入一次，删除后不复活。"""

from __future__ import annotations

from cthulhu_backend import db


def test_preset_templates_seeded_once(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "presets.db"))
    db.init_db()

    first = db.list_templates()
    # 模板列表正序：预置模板按定义顺序（快速→标准→强力→全兵器）返回。
    assert [t["name"] for t in first] == [p["name"] for p in db.PRESET_TEMPLATES]

    # 再次初始化不会重复写入，也不会被旧版迁移逻辑改写。
    db.init_db()
    second = db.list_templates()
    assert [t["name"] for t in second] == [t["name"] for t in first]
    assert [t["payload"] for t in second] == [t["payload"] for t in first]

    # 用户删除全部预设后，重新初始化不复活。
    for template in db.list_templates():
        db.delete_template(template["id"])
    db.init_db()
    assert db.list_templates() == []


def test_preset_payloads_cover_all_clean_options() -> None:
    expected = {
        "audioRemix",
        "echoDefeat",
        "antiReembed",
        "anti",
        "regradeOn",
        "recropOn",
        "detailProtectOn",
        "sharpness",
        "colorRestore",
        "denoise",
        "spoof",
        "codec",
        "lossless",
        "resolution",
        "bitrate",
        "gop",
        "fpsOut",
    }
    for preset in db.PRESET_TEMPLATES:
        assert set(preset["payload"]) == expected
