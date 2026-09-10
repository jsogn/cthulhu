"""共谋平均：多副本对齐平均与画质指标。"""

from __future__ import annotations

import numpy as np

from cthulhu_backend import samples, services
from cthulhu_backend.media import ffmpeg


def test_collusion_average_reduces_static_patterns(tmp_path) -> None:
    gray = samples.make_cut_video(1, 6, 96, 96, seed=0)
    base = np.repeat(gray[..., None], 3, axis=-1)
    height, width = base.shape[1:3]
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    paths = []
    for index in range(3):
        pattern = 0.01 * np.sin(2 * np.pi * (xx + index * 7) / 24) * np.cos(
            2 * np.pi * (yy + index * 5) / 18
        )
        frames = np.clip(base + pattern[None, ..., None], 0, 1)
        target = tmp_path / f"copy-{index}.mp4"
        ffmpeg.encode_video(frames, str(target), fps=30, crf=18)
        paths.append(str(target))
    output = tmp_path / "collusion.mp4"
    result = services.run_collusion(paths, str(output), mode="mean", max_frames=12)
    assert output.is_file()
    assert result["copies"] == 3
    assert result["frames"] == 6
    assert result["estimated_watermark_reduction"] == round(1 / np.sqrt(3), 4)
    assert all(item["psnr_db"] > 25 for item in result["quality"])


def test_collusion_rejects_single_copy(tmp_path) -> None:
    target = tmp_path / "only.mp4"
    ffmpeg.encode_video(samples.make_cut_video(1, 4, 64, 64, seed=1), str(target), fps=30)
    try:
        services.run_collusion([str(target)], str(tmp_path / "out.mp4"))
    except ValueError as exc:
        assert "至少需要 2 个副本" in str(exc)
    else:
        raise AssertionError("单副本应被拒绝")
