"""内容复杂度画像与自动选型接线。"""

from __future__ import annotations

import numpy as np
import pytest

from cthulhu_backend.transform import profile


def test_complexity_monotonic() -> None:
    rng = np.random.default_rng(0)
    flat = np.full((8, 64, 64, 3), 0.5, dtype=np.float32)
    textured = rng.uniform(0.2, 0.8, (8, 64, 64, 3)).astype(np.float32)
    flat_profile = profile.profile_frames(flat)
    textured_profile = profile.profile_frames(textured)
    assert flat_profile.complexity < textured_profile.complexity
    assert flat_profile.suggested_purify_strength <= textured_profile.suggested_purify_strength
    assert 0.0 <= flat_profile.temporal_coherence <= 1.0


def test_scheme_document_matches_engine_maps() -> None:
    """前端「已知来源」下拉与说明必须与引擎映射同源（审计 R2）。"""
    document = {item["id"]: item for item in profile.scheme_document()}
    assert set(document) == {"luma", "chroma"}
    for family_id, item in document.items():
        assert item["attack"] == profile.SCHEME_ATTACK[family_id]
        assert item["max_edge"] == profile.SCHEME_MAX_EDGE[family_id]
        assert item["label"] and item["summary"]
    # 亮度/低频类必须比色度类更狠（research §19.7）。
    assert document["luma"]["max_edge"] < document["chroma"]["max_edge"]


def test_known_scheme_mapping() -> None:
    frames = np.random.default_rng(1).uniform(0, 1, (6, 32, 32, 3)).astype(np.float32)
    assert profile.profile_frames(frames, "videoseal").suggested_attack == "luma"
    assert profile.profile_frames(frames, "wam").suggested_attack == "chroma"
    unknown = profile.profile_frames(frames, "unknown")
    assert unknown.suggested_attack == "both"
    assert "无法可靠识别" in unknown.note


def test_profile_is_deterministic_for_same_input() -> None:
    frames = np.random.default_rng(2).uniform(0, 1, (6, 32, 32, 3)).astype(np.float32)
    assert profile.profile_frames(frames).as_dict() == profile.profile_frames(frames).as_dict()


def test_profile_shots_splits_sampled_frames_by_shot() -> None:
    rng = np.random.default_rng(3)
    flat = np.full((4, 64, 64, 3), 0.5, dtype=np.float32)
    textured = rng.uniform(0.1, 0.9, (4, 64, 64, 3)).astype(np.float32)
    frames = np.concatenate([flat, textured], axis=0)
    profiles = profile.profile_shots(frames, [0], [(0, 4), (4, 8)])
    assert len(profiles) == 2
    assert profiles[0].complexity < profiles[1].complexity


def test_pipeline_applies_per_shot_purify_budget(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """自动画像按镜头下发不同强度，且不超过用户滑块上限。"""
    from cthulhu_backend import samples, services
    from cthulhu_backend.media import ffmpeg
    from cthulhu_backend.transform import purify

    global_profile = profile.Profile(
        complexity=0.5,
        motion=0.02,
        edge_density=0.03,
        suggested_attack="both",
        suggested_purify_strength=0.2,
        suggested_purify_temporal=0.0,
        note="test-global",
    )

    def fake_shots(frames, starts, shot_ranges, **kwargs):
        del frames, starts, kwargs
        return [
            profile.Profile(
                complexity=0.1 + 0.2 * index,
                motion=0.01,
                edge_density=0.02,
                suggested_attack="both",
                suggested_purify_strength=0.1 + 0.1 * index,
                suggested_purify_temporal=0.0,
                note=f"shot-{index}",
            )
            for index in range(len(shot_ranges))
        ]

    monkeypatch.setattr(profile, "profile_frames", lambda frames, scheme="": global_profile)
    monkeypatch.setattr(profile, "profile_shots", fake_shots)
    monkeypatch.setattr(purify, "preflight", lambda: (True, "ready"))
    seen: list[float] = []
    seen_detail: list[tuple[float, float]] = []

    def fake_purify(frames, **kwargs):
        seen.append(float(kwargs.get("strength", 0.0)))
        seen_detail.append(
            (
                float(kwargs.get("detail", 0.0)),
                float(kwargs.get("detail_sigma", 0.0)),
            )
        )
        return frames

    monkeypatch.setattr(purify, "purify_frames", fake_purify)
    frames = samples.make_cut_video(3, 6, 160, 120, seed=0)
    source = tmp_path / "in.mp4"
    output = tmp_path / "out.mp4"
    ffmpeg.encode_video(frames, str(source), fps=30)
    report = services.run_desensitize(
        str(source),
        str(output),
        auto_profile=True,
        purify_strength=0.3,
        purify_detail=1.0,
        purify_detail_sigma=1.5,
        embedding_strength=0.0,
        compute_metrics=False,
        audio_remix=False,
        regrade=False,
        color_restore=False,
        sharpness=False,
    )
    assert report["profile_metrics"]["shot_count"] >= 2
    assert len(report["profile_metrics"]["shots"]) >= 2
    assert len(seen) >= 2
    assert len({round(value, 3) for value in seen}) >= 2
    assert max(seen) <= 0.3 + 1e-6
    # 画质档位参数不被自适应砍掉：detail/σ 原样下发（research §19.2/19.3）。
    assert set(seen_detail) == {(1.0, 1.5)}


def test_pipeline_records_profile(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from cthulhu_backend import samples, services
    from cthulhu_backend.media import ffmpeg
    from cthulhu_backend.transform import purify

    fake = profile.Profile(
        complexity=0.5,
        motion=0.01,
        edge_density=0.02,
        suggested_attack="luma",
        suggested_purify_strength=0.2,
        note="test",
    )
    monkeypatch.setattr(profile, "profile_frames", lambda frames, scheme="": fake)
    monkeypatch.setattr(purify, "preflight", lambda: (False, "test: skip"))
    frames = samples.make_cut_video(1, 6, 160, 120, seed=0)
    source = tmp_path / "in.mp4"
    output = tmp_path / "out.mp4"
    ffmpeg.encode_video(frames, str(source), fps=30)
    report = services.run_desensitize(
        str(source),
        str(output),
        auto_profile=True,
        embedding_attack="auto",
        embedding_strength=0.4,
        purify_strength=0.25,
        compute_metrics=False,
        audio_remix=False,
        regrade=False,
        color_restore=False,
        sharpness=False,
    )
    assert report["profile_note"] == "applied"
    assert report["profile_metrics"]["suggested_attack"] == "luma"
    assert report["purify_note"].startswith("skipped")
