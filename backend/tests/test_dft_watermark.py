"""DFT 域扩频水印测试：往返与 FFT 相位攻击的破坏力。"""

from __future__ import annotations

import numpy as np

from cthulhu_backend import samples
from cthulhu_backend.evaluate import metrics
from cthulhu_backend.transform import regenerate
from cthulhu_backend.watermark import common, dft

BITS = common.payload_bits(1, 64)
REF = common.SYNC + BITS


def _ber(frames: np.ndarray) -> float:
    return float(np.mean([metrics.ber(REF, dft.extract(f, 64, seed=0)) for f in frames]))


def test_dft_roundtrip_clean():
    clean = samples.make_cut_video(3, 12, 320, 240, seed=2)
    watermarked = np.stack([dft.embed(f, BITS, seed=0, alpha=8.0) for f in clean])
    assert _ber(watermarked) < 0.05


def test_fft_phase_destroys_dft():
    clean = samples.make_cut_video(3, 12, 320, 240, seed=2)
    watermarked = np.stack([dft.embed(f, BITS, seed=0, alpha=8.0) for f in clean])
    attacked = regenerate.fft_phase(watermarked, 1.0, np.random.default_rng(8))
    # 全强度相位打散把载荷相关性压到内容噪声水平（约 0.32），显著高于
    # 未攻击基线（0.026），但不保证到 0.5——内容自身在频带内有残余相关。
    assert _ber(attacked) > 0.3
