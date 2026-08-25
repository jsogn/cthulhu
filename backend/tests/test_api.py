"""GUI 后端 API 集成测试。"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from cthulhu_backend import db, samples, services
from cthulhu_backend.main import app
from cthulhu_backend.media import ffmpeg

needs_ffmpeg = pytest.mark.skipif(not ffmpeg.has_ffmpeg(), reason="需要 ffmpeg/ffprobe")

client = TestClient(app, headers={"X-CTHULHU-Token": "test-token"})


def _video(tmp_path, name="v.mp4", seed=20, frames=8):
    path = tmp_path / name
    ffmpeg.encode_video(samples.make_video_frames(frames, 160, 120, seed=seed), str(path), fps=30)
    return path


def test_health():
    assert client.get("/api/health").json()["ok"] is True


def test_detect_missing_file_returns_404():
    response = client.post("/api/detect", json={"path": "/no/such/file.mp4"})
    assert response.status_code == 404


@needs_ffmpeg
def test_detect_includes_blind_scores(tmp_path):
    """检测接口返回盲检测置信度（研究口径启发式）。"""
    path = _video(tmp_path)
    response = client.post("/api/detect", json={"path": str(path)})
    assert response.status_code == 200
    blind = response.json()["blind"]
    assert blind is not None
    for key in ("ss", "qim", "lsb"):
        assert 0 <= blind[key] <= 1


@needs_ffmpeg
def test_audio_analyze_returns_waveform_and_spectrum(tmp_path):
    """音频分析接口返回波形、频谱与回声置信度。"""
    video = samples.make_video_frames(8, 160, 120, seed=21)
    audio = samples.make_audio(1.0, seed=21)
    path = tmp_path / "av.mp4"
    ffmpeg.encode_video_with_audio(video, audio, str(path), fps=30)
    response = client.post("/api/audio/analyze", json={"path": str(path)})
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["waveform"]) > 0
    assert len(payload["spectrum"]) == 32
    assert 0 <= payload["echo_score"] <= 1


@needs_ffmpeg
def test_import_scan_folder(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    _video(tmp_path, "a.mp4", seed=70)
    _video(sub, "b.mp4", seed=71)
    (tmp_path / "c.txt").write_text("忽略")

    response = client.post("/api/import/scan", json={"path": str(tmp_path)})
    assert response.status_code == 200
    payload = response.json()
    assert payload["directory"] is True
    assert payload["valid"] == 2 and payload["invalid"] == 0
    assert {item["name"] for item in payload["files"]} == {"a.mp4", "b.mp4"}


@needs_ffmpeg
def test_detect_reports_bitstream(tmp_path):
    path = _video(tmp_path)
    response = client.post("/api/detect", json={"path": str(path)})
    assert response.status_code == 200
    payload = response.json()
    assert payload["bitstream"]["level"] in {"低", "中", "高"}
    assert 0 <= payload["bitstream"]["score"] <= 100
    assert payload["probe"]["codec"] == "h264"


@needs_ffmpeg
def test_similarity_pair(tmp_path):
    a, b = _video(tmp_path, "a.mp4", seed=21), _video(tmp_path, "b.mp4", seed=22)
    response = client.post("/api/similarity", json={"a": str(a), "b": str(b)})
    assert response.status_code == 200
    report = response.json()
    assert 0.0 <= report["content_cosine"] <= 1.0
    assert "motion_cosine" in report


@needs_ffmpeg
def test_desensitize_end_to_end(tmp_path):
    source = _video(tmp_path, "src.mp4", seed=23, frames=16)
    output = tmp_path / "out.mp4"
    response = client.post(
        "/api/desensitize",
        json={"path": str(source), "output": str(output), "speed": 0.95, "recrop": 0.03},
    )
    assert response.status_code == 200
    payload = response.json()
    assert output.exists()
    assert "similarity_after" in payload
    assert payload["similarity_after"]["content_cosine"] > 0.5


@needs_ffmpeg
def test_desensitize_all_quality_switches(tmp_path):
    source = _video(tmp_path, "q.mp4", seed=24, frames=12)
    output = tmp_path / "qout.mp4"
    response = client.post(
        "/api/desensitize",
        json={
            "path": str(source),
            "output": str(output),
            "sharpness": True,
            "color_restore": True,
            "denoise": True,
            "anti_reembed": True,
        },
    )
    assert response.status_code == 200
    assert output.exists()


@needs_ffmpeg
def test_repair_delogo(tmp_path):
    source = _video(tmp_path, "r.mp4", seed=90)
    output = tmp_path / "rout.mp4"
    response = client.post(
        "/api/repair",
        json={
            "path": str(source),
            "output": str(output),
            "regions": [{"x": 0.2, "y": 0.2, "w": 0.3, "h": 0.2, "start": 0.0, "end": 0.3}],
        },
    )
    assert response.status_code == 200
    assert output.exists()
    assert response.json()["regions"] == 1


@needs_ffmpeg
def test_frame_extraction(tmp_path):
    source = _video(tmp_path, "f.mp4", seed=91)
    response = client.get("/api/frame", params={"path": str(source), "t": 0.1})
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content.startswith(b"\x89PNG")


@needs_ffmpeg
def test_thumbnail_extraction(tmp_path):
    """封面接口返回等比缩放的 JPEG，而非整帧 PNG。"""
    source = _video(tmp_path, "t.mp4", seed=92)
    response = client.get("/api/thumb", params={"path": str(source), "width": 128})
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.content.startswith(b"\xff\xd8")


@needs_ffmpeg
def test_media_streaming_supports_range(tmp_path):
    """媒体接口流式返回原文件并支持 Range 请求（播放器拖动定位）。"""
    source = _video(tmp_path, "m.mp4", seed=93)
    response = client.get("/api/media", params={"path": str(source)})
    assert response.status_code == 200
    assert response.headers["content-type"] == "video/mp4"
    size = int(response.headers["content-length"])
    ranged = client.get(
        "/api/media",
        params={"path": str(source)},
        headers={"Range": f"bytes=0-{min(99, size - 1)}"},
    )
    assert ranged.status_code == 206
    assert ranged.headers["content-range"].startswith("bytes 0-")
    assert len(ranged.content) == min(100, size)


def test_import_rejects_unsupported_extension():
    """导入接口仅接受视频扩展名。"""
    response = client.post(
        "/api/import",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 422


@needs_ffmpeg
def test_import_stores_file_and_returns_path(tmp_path, monkeypatch):
    """导入接口把视频落盘到素材库目录，供播放与检测复用。"""
    source = _video(tmp_path, "up.mp4", seed=94)
    library = tmp_path / "library"
    monkeypatch.setattr("cthulhu_backend.api._LIBRARY_DIR", library)
    with source.open("rb") as handle:
        response = client.post(
            "/api/import",
            files={"file": ("导入/测试 素材.mp4", handle, "video/mp4")},
        )
    assert response.status_code == 200
    payload = response.json()
    assert payload["name"].startswith("测试_素材")
    assert Path(payload["path"]).is_file()
    assert Path(payload["path"]).read_bytes() == source.read_bytes()


@needs_ffmpeg
def test_import_deduplicates_content_and_reuses_cached_hash(tmp_path, monkeypatch):
    """相同内容不重复落盘；第二次导入复用库中缓存的哈希，不再整文件重算。"""
    from cthulhu_backend import api as api_module

    source = _video(tmp_path, "dup.mp4", seed=95)
    library = tmp_path / "library"
    monkeypatch.setattr(api_module, "_LIBRARY_DIR", library)

    def import_named(name: str) -> dict:
        with source.open("rb") as handle:
            response = client.post(
                "/api/import",
                files={"file": (name, handle, "video/mp4")},
            )
        assert response.status_code == 200
        return response.json()

    first = import_named("dup.mp4")
    assert first["duplicate"] is False

    hash_calls: list[str] = []
    original_hash = api_module._content_hash

    def counting_hash(path: str) -> str:
        hash_calls.append(path)
        return original_hash(path)

    monkeypatch.setattr(api_module, "_content_hash", counting_hash)
    second = import_named("dup_copy.mp4")
    assert second["duplicate"] is True
    assert Path(second["path"]).name == "dup.mp4"
    assert not (library / "dup_copy.mp4").exists()
    assert hash_calls == []


@needs_ffmpeg
def test_products_list_and_batch_delete(tmp_path):
    """产物管理：全局清单列出产物，批量删除只清理产物文件与记录。"""
    product = tmp_path / "src_清洗_20260825_10_00_00.mp4"
    product.write_bytes(b"fake-product-bytes")
    db.create_variant(
        source=str(tmp_path / "src.mp4"),
        output=str(product),
        options={"anti": "标准"},
        seed=1,
    )

    listing = client.get("/api/products")
    assert listing.status_code == 200
    payload = listing.json()
    assert payload["count"] == 1
    assert payload["total_size"] == len(b"fake-product-bytes")
    assert payload["products"][0]["name"] == product.name

    removed = client.post("/api/products/delete", json={"paths": [str(product)]})
    assert removed.status_code == 200
    assert removed.json()["removed"] == 1
    assert not product.exists()
    assert client.get("/api/products").json()["count"] == 0


@needs_ffmpeg
def test_products_delete_rejects_non_product(tmp_path):
    """产物删除仅接受产物命名，源视频等普通文件一律拒绝且不受影响。"""
    source = _video(tmp_path, "keep.mp4", seed=91)
    response = client.post("/api/products/delete", json={"paths": [str(source)]})
    assert response.status_code == 400
    assert source.exists()


def test_products_delete_matches_tilde_record(tmp_path, monkeypatch):
    """历史记录以 ~ 路径存储时，按展开后的绝对路径删除同样生效。"""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    output_tilde = "~/x_清洗_y.mp4"
    db.create_variant(
        source=str(tmp_path / "src.mp4"),
        output=output_tilde,
        options={},
        seed=1,
    )
    expanded = os.path.expanduser(output_tilde)
    response = client.post("/api/products/delete", json={"paths": [expanded]})
    assert response.status_code == 200
    assert response.json()["removed"] == 1
    assert client.get("/api/products").json()["count"] == 0


def test_products_delete_by_record_without_marker(tmp_path):
    """产物类型与删除安全以记录为准：改名产物仍可删除，类型取自记录字段。"""
    product = tmp_path / "renamed_output.mp4"
    product.write_bytes(b"renamed-bytes")
    db.create_variant(
        source=str(tmp_path / "src.mp4"),
        output=str(product),
        options={},
        seed=1,
        kind="repaired",
    )
    listing = client.get("/api/products").json()
    assert listing["products"][0]["kind"] == "repaired"
    response = client.post("/api/products/delete", json={"paths": [str(product)]})
    assert response.status_code == 200
    assert response.json()["removed"] == 1
    assert not product.exists()
    assert client.get("/api/products").json()["count"] == 0


@needs_ffmpeg
def test_detect_reports_stage_progress(tmp_path):
    """检测任务按阶段汇报单调递增的进度。"""
    source = _video(tmp_path, "p.mp4", seed=96)
    notes: list[int] = []
    services.run_detect(str(source), progress_cb=lambda percent, _note: notes.append(percent))
    assert notes and notes == sorted(notes) and notes[-1] > 90


@needs_ffmpeg
def test_run_blind_scores(tmp_path):
    """盲检测抽样只返回空间/频域置信度，供清洗产物残留复检。"""
    source = _video(tmp_path, "b.mp4", seed=98)
    blind = services.run_blind(str(source))
    assert blind is not None
    for key in ("ss", "qim", "lsb"):
        assert 0 <= blind[key] <= 1


@needs_ffmpeg
def test_repair_delogo_stops_before_start(tmp_path):
    """取消标志置位时，修复任务在启动 ffmpeg 前立即中断。"""
    source = _video(tmp_path, "c.mp4", seed=97)
    with pytest.raises(InterruptedError):
        ffmpeg.repair_delogo(
            str(source),
            str(tmp_path / "c-out.mp4"),
            [{"x": 0.2, "y": 0.2, "w": 0.3, "h": 0.2}],
            stop=lambda: True,
        )
