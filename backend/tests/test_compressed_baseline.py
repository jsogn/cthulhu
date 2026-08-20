"""压缩域差分基线测试：锁定水印方案在真实编码链路上的生存力结论。"""

from __future__ import annotations

from cthulhu_backend.evaluate import harness


def _baseline(methods: list[str], seeds: list[int] | None = None) -> dict:
    return harness.run_compressed_detection_baseline(
        methods,
        width=320,
        height=240,
        frames_n=12,
        payload_bits=64,
        seeds=seeds or [0, 1],
        attacks=[],
        crf=23,
    )


def test_compressed_ss_survives_coding():
    """空域扩频弱水印在单次 H.264 编码后仍保留可检差分。"""
    report = _baseline(["ss"])
    assert report["methods"]["ss"]["diff_mean"] > 0.3


def test_compressed_qim_collapses_under_coding():
    """DCT-QIM 格结构被 H.264 量化摧毁，水印分数反而低于干净基线。"""
    report = _baseline(["qim"])
    assert report["methods"]["qim"]["diff_mean"] < -0.1


def test_compressed_lsb_has_no_survival():
    """LSB 在压缩链路无生存力：干净与含水印样本检测分数一致（均误报 1.0）。"""
    report = _baseline(["lsb"])
    entry = report["methods"]["lsb"]
    assert abs(entry["diff_mean"]) < 0.01
