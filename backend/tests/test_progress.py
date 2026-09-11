"""任务进度换算的行为回归（公开模块 cthulhu_backend.progress）。"""

from __future__ import annotations

from cthulhu_backend.progress import PROGRESS_BASE, PROGRESS_MAX, PurifyProgress, purify_progress


def test_purify_progress_note_is_global() -> None:
    """批次内进度要换算成整片帧号与百分比，而不是每批重新从 0 开始。"""
    percent, note = purify_progress(240, 240, 0.5, 5684)
    assert note == "潜空间净化 360/5684"
    assert percent == 8 + int(360 / 5684 * 82)


def test_purify_progress_clamps_fraction_and_total() -> None:
    """进度比例越界时被夹到批内，且总帧号不超过全片。"""
    _, clamped = purify_progress(240, 240, 2.0, 5684)
    assert clamped == "潜空间净化 480/5684"
    percent, oversized = purify_progress(5684, 240, 1.0, 5684)
    assert oversized == "潜空间净化 5684/5684"
    assert percent == PROGRESS_MAX


def test_purify_progress_reporter_emits_download_note_verbatim() -> None:
    """下载阶段的文案（含百分比）由引擎给出，直接透传，不改写成全局帧号。"""
    emitted: list[tuple[int, str]] = []
    reporter = PurifyProgress(lambda percent, note: emitted.append((percent, note)), 1000)
    reporter.set_batch(500, 250)
    reporter(0.0, "下载模型 42%")
    assert emitted == [(PROGRESS_BASE + int(500 / 1000 * 82), "下载模型 42%")]


def test_purify_progress_reporter_maps_batch_to_global_percent() -> None:
    emitted: list[tuple[int, str]] = []
    reporter = PurifyProgress(lambda percent, note: emitted.append((percent, note)), 5684)
    reporter.set_batch(240, 240)
    reporter(0.5, "潜空间净化")
    assert emitted == [(8 + int(360 / 5684 * 82), "潜空间净化 360/5684")]


def test_purify_progress_reporter_is_silent_without_callback() -> None:
    """没有回调或全片帧数为 0 时不得抛错（控制面可能未接进度上报）。"""
    PurifyProgress(None, 100)(0.5, "潜空间净化")
    PurifyProgress(lambda *_: None, 0)(0.5, "潜空间净化")
