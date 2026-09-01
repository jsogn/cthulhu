"""通杀验收测试：清洗管线对各基准水印方案的破坏力。"""

from __future__ import annotations

import numpy as np

from cthulhu_backend import samples
from cthulhu_backend.evaluate import harness, metrics
from cthulhu_backend.watermark import common, dwt, qim, qim_rep, ss, temporal

BITS = common.payload_bits(1, 64)
BALANCED = {
    "reorder": True,
    "speed": 0.955,
    "recrop": 0.03,
    "regrade": True,
    "perturb": 0.2,
    "audio_remix": False,
    "denoise": True,
    "seed": 0,
}


def test_balanced_cleanse_destroys_ss_and_qim():
    """平衡档应把空域扩频与 DCT-QIM 强变体破坏到接近随机（BER > 0.35）。"""
    clean = samples.make_cut_video(3, 12, 320, 240, seed=2)
    variants = {
        "ss": {
            "embed": lambda f, b: ss.embed(f, b, seed=0, alpha=0.25),
            "extract": lambda f: ss.extract(f, 64, seed=0),
        },
        "qim": {
            "embed": lambda f, b: qim.embed(f, b, delta=40),
            "extract": lambda f: qim.extract(f, 64, delta=40),
        },
        "qim-rep": {
            "embed": lambda f, b: qim_rep.embed(f, b, delta=20, rep=3),
            "extract": lambda f: qim_rep.extract(f, 64, delta=20, rep=3),
        },
        "dwt": {
            "embed": lambda f, b: dwt.embed(f, b, seed=0),
            "extract": lambda f: dwt.extract(f, 64, seed=0),
        },
        "temporal": {
            "embed": lambda f, b: temporal.embed_frames(f, b),
            "extract": lambda f, n: temporal.extract_frames(f, n),
            "segment": True,
        },
    }
    report = harness.run_cleanse_matrix(clean, variants, {"平衡": BALANCED}, bits=BITS)
    assert report["ss"]["levels"]["平衡"] > 0.35
    assert report["qim"]["levels"]["平衡"] > 0.35
    assert report["qim-rep"]["levels"]["平衡"] > 0.35
    assert report["dwt"]["levels"]["平衡"] > 0.35
    assert report["temporal"]["levels"]["平衡"] > 0.35


def test_audio_cleanse_destroys_echo():
    signal = samples.make_audio(6.0, seed=2)
    bits = common.payload_bits(1, 32)
    result = harness.run_audio_cleanse(signal, bits)
    assert result["before_ber"] < 0.05
    assert result["after_ber"] > 0.3


def test_lpc_whiten_destroys_echo():
    """LPC 残差白化应把回声水印破坏到接近随机。"""
    from cthulhu_backend.transform import audio as audio_transform
    from cthulhu_backend.watermark import echo

    signal = samples.make_audio(6.0, seed=2)
    bits = common.payload_bits(1, 32)
    watermarked = echo.embed(signal, bits, 16000, segment=0.05)
    before = metrics.ber(common.SYNC + bits, echo.extract(watermarked, 32, 16000, segment=0.05))
    attacked = audio_transform.lpc_whiten(
        watermarked, 16000, strength=1.0, rng=np.random.default_rng(0)
    )
    after = metrics.ber(common.SYNC + bits, echo.extract(attacked, 32, 16000, segment=0.05))
    assert before < 0.05
    assert after > 0.3
