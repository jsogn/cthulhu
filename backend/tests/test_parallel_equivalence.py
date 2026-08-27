"""并行评估矩阵与串行结果逐位一致的回归测试。"""

from __future__ import annotations

from cthulhu_backend import samples
from cthulhu_backend.evaluate import harness
from cthulhu_backend.watermark import common, qim, ss

BITS = common.payload_bits(1, 64)


def _variants() -> dict:
    return {
        "ss": {
            "embed": lambda f, b: ss.embed(f, b, seed=0, alpha=0.25),
            "extract": lambda f: ss.extract(f, 64, seed=0),
        },
        "qim": {
            "embed": lambda f, b: qim.embed(f, b, delta=40),
            "extract": lambda f: qim.extract(f, 64, delta=40),
        },
    }


LEVELS = {
    "轻": {"regrade": True, "perturb": 0.15, "seed": 0},
    "重": {"regrade": True, "perturb": 0.3, "denoise": True, "seed": 0},
}


def test_cleanse_matrix_parallel_equals_serial() -> None:
    clean = samples.make_cut_video(2, 8, 320, 240, seed=4)
    serial = harness.run_cleanse_matrix(clean, _variants(), LEVELS, bits=BITS, workers=1)
    parallel = harness.run_cleanse_matrix(clean, _variants(), LEVELS, bits=BITS, workers=3)
    assert parallel == serial


def test_video_harness_parallel_equals_serial() -> None:
    clean = samples.make_video_frames(8, 64, 64, seed=0)
    attacks = ["median", "gaussian", "requant-dct"]
    serial = harness.run_video_harness("ss", clean, BITS, attacks, seed=0, workers=1)
    parallel = harness.run_video_harness("ss", clean, BITS, attacks, seed=0, workers=3)
    assert parallel == serial


def test_detection_harness_parallel_equals_serial() -> None:
    clean = samples.make_video_frames(8, 64, 64, seed=0)
    serial = harness.run_detection_video_harness(
        ["ss", "qim"], clean, BITS, ["median", "gaussian"], workers=1
    )
    parallel = harness.run_detection_video_harness(
        ["ss", "qim"], clean, BITS, ["median", "gaussian"], workers=3
    )
    assert parallel == serial


def test_compressed_baseline_parallel_equals_serial() -> None:
    serial = harness.run_compressed_detection_baseline(
        ["ss", "qim"], 64, 64, 6, 64, [0, 1], ["median", "gaussian"], workers=1
    )
    parallel = harness.run_compressed_detection_baseline(
        ["ss", "qim"], 64, 64, 6, 64, [0, 1], ["median", "gaussian"], workers=3
    )
    assert parallel == serial
