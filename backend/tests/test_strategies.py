"""变换策略骨架测试：thorough 与现状序列一致，注册与回退可用。"""

from __future__ import annotations

import numpy as np
import pytest

from cthulhu_backend import baseline, samples, services
from cthulhu_backend.media import ffmpeg
from cthulhu_backend.transform import strategies
from cthulhu_backend.transform import video as video_transform

needs_ffmpeg = pytest.mark.skipif(not ffmpeg.has_ffmpeg(), reason="需要 ffmpeg/ffprobe")


def make_context(count: int) -> strategies.TransformContext:
    return strategies.TransformContext(
        gammas=np.full(count, 1.05, dtype=np.float32),
        deltas=np.zeros(count, dtype=np.float32),
        banner="",
        seed=0,
        ref_mean=0.5,
        ref_std=0.1,
        mid_rng=np.random.default_rng(0),
    )


def test_thorough_matches_existing_sequence():
    frames = samples.make_video_frames(4, 64, 64, seed=1)
    context = make_context(len(frames))
    options = strategies.TransformOptions(
        recrop=0.0,
        regrade=True,
        color_restore=True,
        sharpness=True,
    )
    actual = strategies.get_strategy("thorough").apply(
        frames.copy(), list(range(len(frames))), context, options
    )
    # 手工复刻现状管线序列，验证封装没有改变行为。
    expected = video_transform.regrade_with_params(
        frames.copy(), context.gammas, context.deltas
    )
    expected = video_transform.color_restore(expected, context.ref_mean, context.ref_std)
    expected = video_transform.sharpen(
        expected, amount=0.25, radius=1.2, threshold=0.01
    )
    np.testing.assert_allclose(actual, expected)


def test_thorough_with_all_off_is_identity():
    frames = samples.make_video_frames(3, 32, 32, seed=2)
    context = make_context(len(frames))
    options = strategies.TransformOptions(
        regrade=False,
        color_restore=False,
        sharpness=False,
    )
    actual = strategies.get_strategy("thorough").apply(
        frames.copy(), list(range(len(frames))), context, options
    )
    np.testing.assert_allclose(actual, frames)


def test_get_strategy_defaults_and_falls_back():
    assert strategies.get_strategy(None).name == "fast"
    assert strategies.get_strategy("不存在的策略").name == "fast"
    assert "thorough" in strategies.STRATEGIES
    assert "fast" in strategies.STRATEGIES


def test_fast_with_all_off_is_identity():
    frames = (samples.make_video_frames(3, 32, 32, seed=4) * 255).round().astype(np.uint8)
    context = make_context(len(frames))
    options = strategies.TransformOptions(
        regrade=False,
        color_restore=False,
        sharpness=False,
    )
    actual = strategies.get_strategy("fast").apply(
        frames.copy(), list(range(len(frames))), context, options
    )
    np.testing.assert_allclose(actual, frames)
    assert actual.dtype == np.uint8


def test_fast_recrop_and_sharpen_keep_shape_and_change_pixels():
    frames = (samples.make_video_frames(4, 96, 64, seed=5) * 255).round().astype(np.uint8)
    context = make_context(len(frames))
    options = strategies.TransformOptions(
        recrop=0.06,
        regrade=False,
        color_restore=False,
        sharpness=True,
    )
    actual = strategies.get_strategy("fast").apply(
        frames.copy(), list(range(len(frames))), context, options
    )
    assert actual.shape == frames.shape
    assert actual.dtype == np.uint8
    assert float(np.mean(np.abs(actual - frames))) > 0.001


def test_fast_regrade_close_to_thorough():
    frames_f = samples.make_video_frames(8, 64, 64, seed=6)
    gammas = np.full(8, 1.05, dtype=np.float32)
    deltas = np.zeros(8, dtype=np.float32)
    expected = video_transform.regrade_with_params(frames_f.copy(), gammas, deltas)
    u8 = (frames_f * 255).round().astype(np.uint8)
    actual = strategies._fast_regrade_u8(u8, gammas, deltas)
    assert actual.dtype == np.uint8
    diff = np.abs(actual.astype(np.float32) / 255.0 - expected)
    assert float(diff.mean()) < 0.02


def test_baseline_compare_roundtrip(tmp_path):
    base_path = tmp_path / "base.json"
    runs_path = tmp_path / "runs.json"
    baseline.write_json(
        {
            "schema_version": 1,
            "files": [{"path": "/tmp/a.mp4", "scores": {"ss": 0.9, "qim": 0.4, "lsb": 0.2, "echo": 0.3}}],
        },
        str(base_path),
    )
    baseline.append_record(
        str(runs_path),
        {
            "strategy": "thorough",
            "input": "/tmp/a.mp4",
            "wall_seconds": 12.3,
            "scores": {"ss": 0.4, "qim": 0.2, "lsb": 0.2, "echo": 0.2},
        },
    )
    report = baseline.compare(str(base_path), str(runs_path))
    assert report["rows"][0]["strategy"] == "thorough"
    assert report["rows"][0]["signal_delta"]["ss"] == -0.5


@needs_ffmpeg
def test_desensitize_records_strategy(tmp_path):
    video = samples.make_cut_video(2, 8, 160, 120, seed=3)
    source = tmp_path / "src.mp4"
    output = tmp_path / "out.mp4"
    ffmpeg.encode_video(video, str(source), fps=30)
    result = services.run_desensitize(
        str(source),
        str(output),
        reorder=False,
        speed=1.0,
        regrade=True,
        audio_remix=False,
        sharpness=False,
        color_restore=False,
    )
    assert result["transform_strategy"] == "fast"
    thorough = services.run_desensitize(
        str(source),
        str(tmp_path / "thorough.mp4"),
        reorder=False,
        speed=1.0,
        regrade=True,
        audio_remix=False,
        sharpness=False,
        color_restore=False,
        transform_strategy="thorough",
    )
    assert thorough["transform_strategy"] == "thorough"
