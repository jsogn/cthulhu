"""参考水印基准库的往返与攻击鲁棒性测试。"""

from __future__ import annotations

import numpy as np

from cthulhu_backend import samples
from cthulhu_backend.attacks import dct as dct_attacks
from cthulhu_backend.attacks import geometric, spatial, temporal
from cthulhu_backend.evaluate import harness, metrics
from cthulhu_backend.watermark import common, dwt, echo, lsb, qim, ss
from cthulhu_backend.watermark import temporal as temporal_wm

BITS = common.payload_bits(7, 64)
FRAME = samples.make_video_frames(1, 320, 240, seed=3)[0]
AUDIO = samples.make_audio(1.0, seed=3)


def test_lsb_roundtrip():
    out = lsb.embed(FRAME, BITS, seed=1)
    extracted = lsb.extract(out, len(BITS), seed=1)
    assert common.bit_error_rate(common.SYNC + BITS, extracted) == 0.0


def test_ss_roundtrip():
    out = ss.embed(FRAME, BITS, seed=2)
    extracted = ss.extract(out, len(BITS), seed=2)
    assert common.bit_error_rate(common.SYNC + BITS, extracted) < 0.01


def test_qim_roundtrip():
    out = qim.embed(FRAME, BITS)
    extracted = qim.extract(out, len(BITS))
    assert common.bit_error_rate(common.SYNC + BITS, extracted) == 0.0


def test_echo_roundtrip():
    short_bits = common.payload_bits(7, 8)
    out = echo.embed(AUDIO, short_bits, 16000, segment=0.05)
    extracted = echo.extract(out, len(short_bits), 16000, segment=0.05)
    assert common.bit_error_rate(common.SYNC + short_bits, extracted) < 0.05


def test_dwt_roundtrip():
    out = dwt.embed(FRAME, BITS, seed=3)
    extracted = dwt.extract(out, len(BITS), seed=3)
    assert common.bit_error_rate(common.SYNC + BITS, extracted) == 0.0


def test_temporal_roundtrip():
    frames = samples.make_cut_video(4, 30, 320, 240, seed=5)
    out = temporal_wm.embed_frames(frames, BITS)
    extracted = temporal_wm.extract_frames(out, len(BITS))
    assert common.bit_error_rate(common.SYNC + BITS, extracted) == 0.0


def test_lsb_randomize_kills_lsb():
    out = lsb.embed(FRAME, BITS, seed=1)
    attacked = spatial.randomize_lsb(np.asarray([out]))[0]
    extracted = lsb.extract(attacked, len(BITS), seed=1)
    assert common.bit_error_rate(common.SYNC + BITS, extracted) > 0.2


def test_median_degrades_ss():
    out = ss.embed(FRAME, BITS, seed=2)
    attacked = spatial.median(np.asarray([out]), size=3)[0]
    extracted = ss.extract(attacked, len(BITS), seed=2)
    assert common.bit_error_rate(common.SYNC + BITS, extracted) > 0.1


def test_requant_dct_degrades_qim():
    out = qim.embed(FRAME, BITS)
    attacked = dct_attacks.requant_dct(out, step=12)
    extracted = qim.extract(attacked, len(BITS))
    assert common.bit_error_rate(common.SYNC + BITS, extracted) > 0.1


def test_geometric_and_temporal_run():
    frames = samples.make_video_frames(8, 320, 240, seed=5)
    assert geometric.crop_rotate_rescale(frames).shape == frames.shape
    assert temporal.drop_duplicate(frames).shape == frames.shape


def test_metrics_psnr_ssim_boundaries():
    assert metrics.psnr(FRAME, FRAME) == float("inf")
    assert metrics.ssim(FRAME, FRAME) > 0.999
    assert abs(metrics.psnr(FRAME, FRAME + 0.1) - 20.0) < 0.01


def test_harness_end_to_end():
    frames = samples.make_video_frames(8, 320, 240, seed=9)
    report = harness.run_video_harness("ss", frames, BITS, ["median", "geometric"], seed=2)
    assert set(report["attacks"]) == {"median", "geometric"}
    assert all(0.0 <= r["ber"] <= 1.0 for r in report["attacks"].values())
