"""对真实清洗管线做 cProfile 剖面，定位各武器与基础设施的真实耗时。

用法：
  CTHULHU_BENCH_SRC=<视频路径> uv run --project backend python backend/scripts/prof_pipeline.py
"""

from __future__ import annotations

import cProfile
import os
import pstats
import subprocess
import sys
import tempfile

from cthulhu_backend import services


def main() -> None:
    src = os.environ.get("CTHULHU_BENCH_SRC", "")
    if not src:
        raise SystemExit("请设置 CTHULHU_BENCH_SRC 指向基准视频后重跑")
    tmp = tempfile.mkdtemp(prefix="cthulhu-prof-")
    short = os.path.join(tmp, "short.mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", src, "-frames:v", "120", "-c:v",
         "libx264", "-preset", "veryfast", "-crf", "23", "-an", short],
        check=True,
    )
    out = os.path.join(tmp, "out.mp4")
    opts = {
        "reorder": False, "speed": 1.0, "recrop": 0.0, "perturb": 0.15,
        "regrade": True,
        "audio_remix": False, "echo_defeat": False, "audio_strong": False,
        "sharpness": True, "color_restore": True, "denoise": True,
        "rotate": 0.5, "requant": 64, "noise": 0.003, "dct_step": 12.0,
        "temporal_sub": 0.6, "native_temporal": True,
        "fft_phase": 0.5, "dwt_detail": 0.8, "quality_protect": True,
        "psnr_target": 38.0, "ssim_target": 0.94, "skip_vmaf": True, "seed": 0,
        "hardware": os.environ.get("HARDWARE") == "1",
    }
    profiler = cProfile.Profile()
    profiler.enable()
    services.run_desensitize(short, out, **opts)
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats("tottime")
    stats.print_stats(35)
    return 0


if __name__ == "__main__":
    sys.exit(main())
