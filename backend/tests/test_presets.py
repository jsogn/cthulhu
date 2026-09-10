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
        "purifyStrength",
        "purifyDetail",
        "purifyDetailSigma",
        "purifyDetailWide",
        "purifyTemporal",
        "purifyMaxEdge",
        "purifyBatch",
        "embeddingAttack",
        "embeddingStrength",
        "embeddingVariant",
        "embeddingAggressive",
        "autoProfile",
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


def test_template_payload_roundtrips_purify_options(monkeypatch, tmp_path) -> None:
    """模板保存/加载必须原样保留净化与嵌入域配置，不能被 extra=ignore 丢弃。"""
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "templates.db"))
    db.init_db()
    payload = {
        "purifyStrength": 0.25,
        "purifySteps": 30,
        "purifyGuidance": 0.0,
        "purifyDetail": 0.6,
        "purifyTemporal": 0.3,
        "embeddingAttack": "chroma",
        "embeddingStrength": 0.4,
        "embeddingVariant": "v2",
        "embeddingAggressive": True,
    }
    created = db.create_template("净化模板", payload)
    loaded = next(t for t in db.list_templates() if t["id"] == created["id"])
    for key, value in payload.items():
        assert loaded["payload"][key] == value


def test_deep_watermark_presets_all_enable_rebuild() -> None:
    """四个内置模板都以大平台深度水印为目标：必须启用画面重建并给出档位。"""
    presets = {p["name"]: p["payload"] for p in db.PRESET_TEMPLATES}
    assert set(presets) == {
        "画质优先（推荐）",
        "平衡去水印",
        "强力去水印",
        "深度清剿（最狠）",
    }
    for name, payload in presets.items():
        assert payload["purifyStrength"] > 0, f"{name} 未启用画面重建"
        assert payload["purifyDetail"] > 0, f"{name} 未开启细节回注"
        assert payload["purifyMaxEdge"] in {128, 192, 256, 512}, name
        assert payload["purifyBatch"] >= 8, name


def test_quality_first_preset_matches_panel_default() -> None:
    """「画质优先（推荐）」必须与面板默认值一致，否则用户会看到两套参数。"""
    payload = next(
        p for p in db.PRESET_TEMPLATES if p["name"] == "画质优先（推荐）"
    )["payload"]
    assert payload["purifyMaxEdge"] == 512
    assert payload["purifyDetailWide"] is True
    assert payload["purifyDetailSigma"] == 0.0
    assert payload["autoProfile"] is True


def test_strong_presets_get_harsher_than_balanced() -> None:
    """强力/深度两档必须比平衡档更狠：更小的长边 + 时序减法 + 嵌入域增强重写。"""
    presets = {p["name"]: p["payload"] for p in db.PRESET_TEMPLATES}
    balanced = presets["平衡去水印"]
    for name in ("强力去水印", "深度清剿（最狠）"):
        payload = presets[name]
        assert payload["purifyMaxEdge"] < balanced["purifyMaxEdge"], name
        assert payload["purifyTemporal"] > 0, f"{name} 未开时序减法"
        assert payload["embeddingStrength"] > 0, f"{name} 未开嵌入域重写"
        assert payload["embeddingAggressive"] is True, name
        assert payload["embeddingAttack"] == "both", name


def test_all_presets_keep_audio_cleanup_and_denoise() -> None:
    """音频三件套与空间降噪是默认安全网，任何模板都不得关闭。"""
    for preset in db.PRESET_TEMPLATES:
        payload = preset["payload"]
        assert payload["audioRemix"] and payload["echoDefeat"], preset["name"]
        assert payload["denoise"], preset["name"]


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
