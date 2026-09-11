"""压缩域检测模块测试（依赖 ffmpeg）。"""

from __future__ import annotations

import numpy as np
import pytest

from cthulhu_backend import samples
from cthulhu_backend.bitstream import analyze as bitstream_analyze
from cthulhu_backend.bitstream import frames as frames_module
from cthulhu_backend.bitstream import qp as qp_module
from cthulhu_backend.bitstream.stats import autocorr_peak, normalized_entropy
from cthulhu_backend.media import ffmpeg

needs_ffmpeg = pytest.mark.skipif(not ffmpeg.has_ffmpeg(), reason="需要 ffmpeg/ffprobe")


def test_stats_helpers_periodic_autocorr_and_entropy():
    periodic = [1.0, -1.0] * 8
    random = list(np.random.default_rng(0).standard_normal(32))
    assert autocorr_peak(periodic) > 0.9
    assert autocorr_peak(random) < 0.5
    assert 0.0 <= normalized_entropy(np.array([0, 1, 2, 3, 0, 1, 2, 3])) <= 1.0


@needs_ffmpeg
def test_qp_maps_and_allocation(tmp_path):
    frames = samples.make_video_frames(12, 320, 240, seed=11)
    path = tmp_path / "v.mp4"
    ffmpeg.encode_video(frames, str(path), fps=30)

    qp_frames = qp_module.extract_qp_maps(str(path))
    assert qp_frames and len(qp_frames) >= 10
    first_map = np.asarray(qp_frames[0]["rows"])
    assert first_map.ndim == 2 and first_map.size > 0
    assert np.all((first_map >= 0) & (first_map <= 51))

    alloc = frames_module.frame_allocation(str(path))
    assert alloc and len(alloc[0]) >= 10
    assert any(alloc[1])  # 至少一个关键帧


@needs_ffmpeg
def test_analyze_deterministic_and_bounded(tmp_path):
    frames = samples.make_video_frames(12, 320, 240, seed=12)
    path = tmp_path / "w.mp4"
    ffmpeg.encode_video(frames, str(path), fps=30)
    first = bitstream_analyze.analyze(str(path))
    second = bitstream_analyze.analyze(str(path))
    # ffmpeg 的 -debug qp 输出偶发漏行，数值统计存在 1e-4 级抖动；
    # 业务判定字段（分数/等级/命中项）必须稳定。
    assert first["score"] == second["score"]
    assert first["level"] == second["level"]
    assert first["flags"] == second["flags"]
    assert 0 <= first["score"] <= 100
    assert first["level"] in {"低", "中", "高"}
