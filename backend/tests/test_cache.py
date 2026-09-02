"""分析缓存（颜色统计/镜头边界）与异步指标回写的行为回归。"""

from __future__ import annotations

import time

import numpy as np
import pytest

from cthulhu_backend import db, jobs, samples, services
from cthulhu_backend.cache import AnalysisCache
from cthulhu_backend.media import ffmpeg
from cthulhu_backend.schemas import DesensitizeOptions

needs_ffmpeg = pytest.mark.skipif(not ffmpeg.has_ffmpeg(), reason="需要 ffmpeg/ffprobe")


def _opts(**overrides) -> DesensitizeOptions:
    base = {
        "color_restore": True,
        "sharpness": False,
        "audio_remix": False,
        "regrade": False,
        "skip_vmaf": True,
        "seed": 7,
    }
    base.update(overrides)
    return DesensitizeOptions(**base)


def _noop(*args, **kwargs) -> None:
    pass


@needs_ffmpeg
def test_color_stats_reused_without_resample(monkeypatch, tmp_path):
    """仅色彩还原时不再解码 400 帧灰度抽样；同素材二次分析走缓存。"""
    video = tmp_path / "v.mp4"
    ffmpeg.encode_video(samples.make_video_frames(16, 160, 120, seed=3), str(video), fps=30)
    calls: list[tuple[tuple, dict]] = []
    original = services.ffmpeg.decode_sampled

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(services.ffmpeg, "decode_sampled", spy)
    opts = _opts()
    services._prepare_desensitize(str(video), opts, _noop, lambda: False)
    assert len(calls) == 1, "仅色彩还原应只解码 40 帧色彩抽样"
    assert calls[0][1].get("cap") == 40

    services._prepare_desensitize(str(video), opts, _noop, lambda: False)
    assert len(calls) == 1, "二次分析应命中颜色统计缓存，不再解码"


@needs_ffmpeg
def test_shot_boundaries_reused_and_equal(monkeypatch, tmp_path):
    """镜头边界缓存命中后跳过 400 帧抽样，且分段结果与首轮一致。"""
    frames = samples.make_cut_video(4, 8, 160, 120, seed=5)
    video = tmp_path / "cuts.mp4"
    ffmpeg.encode_video(frames, str(video), fps=30)
    calls: list[tuple] = []
    original = services.ffmpeg.decode_sampled

    def spy(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(services.ffmpeg, "decode_sampled", spy)
    opts = _opts(reorder=True)
    first = services._prepare_desensitize(str(video), opts, _noop, lambda: False)
    assert len(calls) == 2, "首轮需要 400 帧镜头抽样 + 40 帧色彩抽样"

    second = services._prepare_desensitize(str(video), opts, _noop, lambda: False)
    assert len(calls) == 2, "二轮镜头边界与颜色统计都应命中缓存"
    assert second.segments == first.segments
    assert second.seg_factors == first.seg_factors
    assert second.seg_out_lens == first.seg_out_lens


def test_cache_entry_invalidated_when_file_changes(tmp_path):
    """文件被修改后缓存自动失效，且残留条目被清理。"""
    target = tmp_path / "media.bin"
    target.write_bytes(b"a" * 1000)
    store = AnalysisCache(tmp_path / "cache")
    store.put_color_stats(str(target), np.array([0.2, 0.3, 0.4]), np.array([0.1] * 3))
    hit = store.get_color_stats(str(target))
    assert hit is not None
    np.testing.assert_allclose(hit[0], [0.2, 0.3, 0.4])

    target.write_bytes(b"b" * 1000)
    assert store.get_color_stats(str(target)) is None
    assert not list((tmp_path / "cache").glob("*.json")), "失效条目应被移除"


def test_cache_total_size_is_bounded_and_lru(tmp_path):
    """目录总量超限时按最旧优先淘汰，总量始终不超过上限。"""
    store = AnalysisCache(tmp_path / "cache", max_bytes=1024)
    for index in range(6):
        target = tmp_path / f"f{index}.bin"
        target.write_bytes(b"x" * 64)
        store.put_color_stats(
            str(target),
            np.full(3, index, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
        )
    total = sum(p.stat().st_size for p in (tmp_path / "cache").glob("*.json"))
    assert total <= 1024
    assert store.get_color_stats(str(tmp_path / "f5.bin")) is not None, "最新条目应保留"
    assert store.get_color_stats(str(tmp_path / "f0.bin")) is None, "最旧条目应被淘汰"


def test_cache_survives_malformed_entries(tmp_path):
    """缓存文件被外部改坏（合法 JSON 但结构不对）时退化为未命中，不外抛异常。"""
    target = tmp_path / "media.bin"
    target.write_bytes(b"a" * 1000)
    store = AnalysisCache(tmp_path / "cache")
    store.put_color_stats(str(target), np.array([0.2, 0.3, 0.4]), np.zeros(3))
    entry = next(iter(store._dir.glob("*.json")))
    entry.write_text("[]", encoding="utf-8")
    assert store.get_color_stats(str(target)) is None
    assert not entry.exists(), "结构损坏的条目应被删除"


def test_get_color_stats_tolerates_missing_keys(monkeypatch, tmp_path):
    """条目数据缺字段时返回 None，而不是把 KeyError 漏给清洗主流程。"""
    store = AnalysisCache(tmp_path / "cache")
    monkeypatch.setattr(store, "_load", lambda kind, path: {"ref_mean": [0.1, 0.2, 0.3]})
    assert store.get_color_stats("/nonexistent") is None


def test_update_variant_metrics(tmp_path):
    """异步指标按输出路径回写，兼容 ~ 展开与 realpath。"""
    output = tmp_path / "out.mp4"
    db.create_variant(
        source=str(tmp_path / "src.mp4"),
        output=str(output),
        options={},
        seed=0,
        metrics=None,
    )
    assert db.update_variant_metrics(str(output), {"psnr_db": 31.2, "ssim": 0.95})
    record = db.get_variant_by_output(str(output))
    assert record is not None
    assert record["metrics"]["psnr_db"] == 31.2
    assert record["metrics"]["ssim"] == 0.95


@needs_ffmpeg
def test_defer_metrics_completes_in_background(tmp_path):
    """defer_metrics=True 立即返回，指标随后回写产物记录。"""
    video = tmp_path / "src.mp4"
    output = tmp_path / "out.mp4"
    ffmpeg.encode_video(samples.make_video_frames(8, 160, 120, seed=9), str(video), fps=30)
    result = services.run_desensitize(
        str(video),
        str(output),
        defer_metrics=True,
        color_restore=False,
        sharpness=False,
        audio_remix=False,
        skip_vmaf=True,
        seed=1,
    )
    assert result["metrics_pending"] is True
    assert result["psnr_db"] is None
    services.record_variant(
        str(video),
        str(output),
        {"seed": 1},
        seed=1,
        metrics=None,
    )
    deadline = time.time() + 30
    while time.time() < deadline:
        record = db.get_variant_by_output(str(output))
        if record and record["metrics"].get("psnr_db") is not None:
            break
        time.sleep(0.2)
    record = db.get_variant_by_output(str(output))
    assert record is not None
    assert record["metrics"].get("psnr_db") is not None


@needs_ffmpeg
def test_deferred_metrics_waits_for_gate(tmp_path):
    """空闲闸门放行前不启动指标计算，放行后才回写。"""
    video = tmp_path / "src.mp4"
    output = tmp_path / "out.mp4"
    ffmpeg.encode_video(samples.make_video_frames(8, 160, 120, seed=9), str(video), fps=30)
    allow = {"ok": False}

    def gate() -> bool:
        return allow["ok"]

    result = services.run_desensitize(
        str(video),
        str(output),
        defer_metrics=True,
        metrics_gate=gate,
        color_restore=False,
        sharpness=False,
        audio_remix=False,
        skip_vmaf=True,
        seed=1,
    )
    assert result["metrics_pending"] is True
    services.record_variant(str(video), str(output), {"seed": 1}, seed=1, metrics=None)
    time.sleep(1.0)
    held = db.get_variant_by_output(str(output))
    assert held is not None and held["metrics"].get("psnr_db") is None, "闸门未放行前不应计算指标"

    allow["ok"] = True
    deadline = time.time() + 20
    while time.time() < deadline:
        record = db.get_variant_by_output(str(output))
        if record and record["metrics"].get("psnr_db") is not None:
            break
        time.sleep(0.2)
    record = db.get_variant_by_output(str(output))
    assert record is not None and record["metrics"].get("psnr_db") is not None


def test_queue_idle_gate():
    """任务队列空闲闸门反映并发槽占用状态。"""
    queue = jobs.JobQueue()
    assert queue.is_idle() is True
    queue._active_slots = 1
    assert queue.is_idle() is False


@needs_ffmpeg
def test_sampling_windows_parallel_equals_serial(monkeypatch, tmp_path):
    """抽样窗口并行与串行逐位一致（帧数组与窗口起始帧号）。"""
    video = tmp_path / "long.mp4"
    frames = samples.make_video_frames(90, 160, 120, seed=11)
    ffmpeg.encode_video(frames, str(video), fps=30)
    monkeypatch.setenv("CTHULHU_SAMPLE_WORKERS", "1")
    serial_frames, _info, serial_starts = ffmpeg.decode_sampled(
        str(video), cap=40, return_starts=True
    )
    monkeypatch.setenv("CTHULHU_SAMPLE_WORKERS", "6")
    parallel_frames, _info2, parallel_starts = ffmpeg.decode_sampled(
        str(video), cap=40, return_starts=True
    )
    np.testing.assert_array_equal(parallel_frames, serial_frames)
    assert parallel_starts == serial_starts


def test_sampling_workers_env_and_memory_caps(monkeypatch):
    """并行度受环境覆盖与 CPU/内存上限共同约束。"""
    from cthulhu_backend.media.ffmpeg import _sampling_workers

    with monkeypatch.context() as mp:
        mp.setenv("CTHULHU_SAMPLE_WORKERS", "9")
        assert _sampling_workers(6, 1024) == 6
        mp.setenv("CTHULHU_SAMPLE_WORKERS", "2")
        assert _sampling_workers(6, 1024) == 2
    workers = _sampling_workers(6, 1024)
    assert 1 <= workers <= 6
