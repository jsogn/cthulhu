"""预置模板：首次初始化写入一次，删除后不复活。"""

from __future__ import annotations

import json
import sqlite3

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
        "rotate",
        "hashAttack",
        "hashEpsilon",
        "hashMode",
        "requant",
        "noise",
        "dctStep",
        "audioStrong",
        "regradeOn",
        "recropOn",
        "temporalSub",
        "fftPhase",
        "fftMag",
        "dwtDetail",
        "warp",
        "perspective",
        "jitter",
        "hsvJitter",
        "nonintRatio",
        "flowDisturb",
        "textureInject",
        "multiscale",
        "complexityTrap",
        "facePerturb",
        "temporalBlur",
        "lpcAttack",
        "copyAttack",
        "nativeTemporal",
        "qualityProtect",
        "psnrTarget",
        "ssimTarget",
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


def test_pure_watermark_preset_excludes_fingerprint_layer_weapons() -> None:
    """「清除推荐」档只保留破坏嵌入水印信号的原语，不含判重指纹/双目标武器。"""
    preset = next(p for p in db.PRESET_TEMPLATES if p["name"] == "清除推荐")
    payload = preset["payload"]
    fingerprint_layer = {
        "rotate",
        "hashAttack",
        "recropOn",
        "regradeOn",
        "hsvJitter",
        "warp",
        "perspective",
        "jitter",
        "nonintRatio",
        "flowDisturb",
        "textureInject",
        "multiscale",
        "complexityTrap",
        "facePerturb",
        "temporalBlur",
        "lpcAttack",
        "copyAttack",
        "antiReembed",
        "spoof",
    }
    for field in fingerprint_layer:
        assert not payload.get(field), f"{field} 是判重指纹/双目标武器，不应出现在清除推荐档"
    watermark_core = {
        "requant",
        "noise",
        "dctStep",
        "temporalSub",
        "nativeTemporal",
        "fftPhase",
        "dwtDetail",
        "denoise",
    }
    for field in watermark_core:
        assert payload.get(field), f"{field} 应开启，保证暗水印破坏力"


def test_weak_weapons_removed_from_presets() -> None:
    """光流退出常规预置；像素重写（伪超分+CLAHE）经实测对基准库无效，全兵器也不带。"""
    strong = next(p for p in db.PRESET_TEMPLATES if p["name"] == "深度清除")
    assert not strong["payload"].get("flowDisturb")
    assert not strong["payload"].get("fftMag")
    assert not strong["payload"].get("textureInject")
    assert not strong["payload"].get("nonintRatio")
    assert not strong["payload"].get("multiscale")
    assert not strong["payload"].get("temporalBlur")
    assert strong["payload"].get("temporalSub") == 1.2
    full = next(p for p in db.PRESET_TEMPLATES if p["name"] == "全部武器（实验）")
    assert full["payload"].get("temporalSub") == 1.2
    assert full["payload"].get("nativeTemporal") is True


def test_balanced_preset_prefers_listening_quality() -> None:
    """LPC 对回声水印收益与听感代价不匹配，常规档按听感优先移除。"""
    balanced = next(p for p in db.PRESET_TEMPLATES if p["name"] == "防重复清除")
    assert not balanced["payload"].get("lpcAttack")
    strong = next(p for p in db.PRESET_TEMPLATES if p["name"] == "深度清除")
    assert not strong["payload"].get("lpcAttack")
    for preset in db.PRESET_TEMPLATES:
        assert preset["payload"].get("denoise"), (
            f"{preset['name']} 应默认开空间降噪（removegrain 是 SS/QIM/DWT 主要破坏者）"
        )


def test_preset_migration_rebuilds_presets_and_keeps_user_templates(monkeypatch, tmp_path) -> None:
    """版本升级时重建内置预置，用户自建模板保留。"""
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "mig.db"))
    db.init_db()
    db.create_template("我的模板", {"audioRemix": False})
    # 模拟旧安装：把预置版本退回 1。
    with sqlite3.connect(str(tmp_path / "mig.db")) as connection:
        connection.execute(
            "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
            ("preset_version", json.dumps(1)),
        )
        connection.commit()
    db.init_db()
    names = [template["name"] for template in db.list_templates()]
    preset_names = [p["name"] for p in db.PRESET_TEMPLATES]
    assert [name for name in names if name in preset_names] == preset_names
    assert "我的模板" in names
