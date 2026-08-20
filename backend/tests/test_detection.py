"""盲检测置信度与检测评估闭环测试。

锁定两类事实：可检方案（QIM/SS/echo）的水印样本置信度应高于干净样本；
低负载 LSB 覆盖型嵌入在数学上不可盲检，检测器保持中性而非误报。
"""

from __future__ import annotations

from cthulhu_backend import samples
from cthulhu_backend.evaluate import harness
from cthulhu_backend.watermark import common

FRAMES = samples.make_video_frames(16, 320, 240, seed=0)
AUDIO = samples.make_audio(3.0, seed=0)
BITS = common.payload_bits(1, 64)


def test_qim_detection_strongly_separates_clean_and_watermarked():
    report = harness.run_detection_video_harness(["qim"], FRAMES, BITS, [], seed=0)
    assert report["qim"]["watermarked"] - report["clean"]["qim"] > 0.5


def test_ss_detection_separates_clean_and_watermarked():
    report = harness.run_detection_video_harness(["ss"], FRAMES, BITS, [], seed=0)
    assert report["ss"]["watermarked"] > report["clean"]["ss"] + 0.4


def test_echo_detection_separates_clean_and_watermarked():
    report = harness.run_detection_audio_harness(AUDIO, BITS, [], seed=0)
    assert report["watermarked"] > report["clean"] + 0.05


def test_lsb_low_payload_is_not_falsely_flagged():
    """低负载 LSB 覆盖嵌入不可盲检，检测器应保持中性（诚实局限）。"""
    report = harness.run_detection_video_harness(["lsb"], FRAMES, BITS, [], seed=0)
    assert abs(report["lsb"]["watermarked"] - report["clean"]["lsb"]) < 0.05


def test_requant_degrades_qim_detectability():
    """重量化攻击后 QIM 置信度显著下降，作为清除效果的量化证据。"""
    report = harness.run_detection_video_harness(
        ["qim"], FRAMES, BITS, ["requant"], seed=0,
    )
    assert report["qim"]["attacked"]["requant"] < report["qim"]["watermarked"] - 0.3
