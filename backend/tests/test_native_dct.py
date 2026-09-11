"""原生 DCT 重量化的等价性与对抗门控（库缺失时自动跳过）。"""

from __future__ import annotations

import numpy as np
import pytest

from cthulhu_backend import native_dct
from cthulhu_backend.transform import extra_attacks
from cthulhu_backend.watermark import common, qim

BITS = common.payload_bits(7, 64)
STEP = 12.0 / 255.0


def _native_ready() -> bool:
    return native_dct._load_lib() is not None


pytestmark = pytest.mark.skipif(not _native_ready(), reason="需要编译原生 DCT 库")


def test_native_matches_numpy(monkeypatch):
    """原生实现与 numpy 逐位口径一致（半偶舍入 + float32）。"""
    rng = np.random.default_rng(40)
    frames = rng.random((6, 96, 64), dtype=np.float32)
    expected = extra_attacks.dct_requant_plane(frames, STEP)
    monkeypatch.setenv("CTHULHU_NATIVE_DCT", "1")
    actual = extra_attacks.dct_requant_plane(frames, STEP)
    diff = np.abs(actual - expected)
    # 语义口径：两条路径应落在同一量化格内（差异小于半个量化步）；
    # BLAS 与标量求和的 float32 顺序噪声允许存在，但不允许跨格。
    assert float(diff.max()) < STEP / 2, f"最大差异 {float(diff.max()):.2e} 超过半个量化步"


def test_native_preserves_qim_destruction(monkeypatch):
    """原生路径对 QIM 的破坏力不低于 numpy 口径，且两者相近。"""
    rng = np.random.default_rng(41)
    frames = rng.random((4, 96, 64), dtype=np.float32)
    watermarked = np.stack([qim.embed(frame, BITS) for frame in frames])

    numpy_attacked = extra_attacks.dct_requant_plane(watermarked, STEP)
    monkeypatch.setenv("CTHULHU_NATIVE_DCT", "1")
    native_attacked = extra_attacks.dct_requant_plane(watermarked, STEP)

    def ber(attacked: np.ndarray) -> float:
        extracted = [qim.extract(frame, len(BITS)) for frame in attacked]
        return float(np.mean([common.bit_error_rate(common.SYNC + BITS, bits) for bits in extracted]))

    numpy_ber = ber(numpy_attacked)
    native_ber = ber(native_attacked)
    assert numpy_ber > 0.1, "numpy 路径应破坏 QIM"
    assert native_ber > 0.1, "原生路径应破坏 QIM"
    assert abs(native_ber - numpy_ber) <= 0.05, "两条路径的破坏力不应显著漂移"
