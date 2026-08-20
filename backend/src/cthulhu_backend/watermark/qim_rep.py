"""带重复冗余的 QIM 水印：每比特多块冗余，模拟平台的纠错编码。

用于检验清洗管线对「抗破坏增强」水印的破坏力；真实平台常宣称
数百百分比的纠错冗余，本方案以 rep 倍冗余逼近该特征。
"""

from __future__ import annotations

from cthulhu_backend.watermark import qim
from cthulhu_backend.watermark.common import with_sync


def embed(frame, bits: list[int], delta: float = 20.0, rep: int = 3):
    synced = with_sync(bits)
    repeated = [bit for bit in synced for _ in range(rep)]
    return qim.embed_synced(frame, repeated, delta)


def extract(frame, n_bits: int, delta: float = 20.0, rep: int = 3) -> list[int]:
    total = len(with_sync([0] * n_bits))
    raw = qim.extract_total(frame, total * rep, delta)
    bits = []
    for index in range(total):
        group = raw[index * rep : (index + 1) * rep]
        bits.append(1 if group and sum(group) > rep / 2 else 0)
    return bits
