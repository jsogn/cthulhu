"""时间域水印：帧间差分域加性嵌入，信息分散在帧序列上。

在连续帧的差分残差上叠加伪随机图案，与空域/频域方案互补；
解码需要整段帧序列，检验清洗管线对时序结构的破坏。
"""

from __future__ import annotations

import numpy as np

from cthulhu_backend.watermark.common import with_sync


def _pattern(shape: tuple[int, ...], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pattern = rng.standard_normal(shape)
    return pattern / np.sqrt(np.mean(pattern**2) + 1e-12)


def embed_frames(
    frames: np.ndarray,
    bits: list[int],
    seed: int = 0,
    alpha: float = 0.05,
) -> np.ndarray:
    """把比特循环绝对调制到每帧的加性图案上。"""
    bits = with_sync(bits)
    pattern = _pattern(frames.shape[1:], seed)
    out = []
    for index, frame in enumerate(frames):
        sign = 1.0 if bits[index % len(bits)] else -1.0
        out.append(np.clip(np.asarray(frame, dtype=np.float64) + alpha * sign * pattern, 0, 1))
    return np.stack(out)


def extract_frames(frames: np.ndarray, n_bits: int, seed: int = 0) -> list[int]:
    """帧间差分相关解码：符号跃迁积分，同步头消除绝对方向歧义。"""
    total = len(with_sync([0] * n_bits))
    pattern = _pattern(frames.shape[1:], seed)
    differences = np.diff(frames, axis=0)
    threshold = 0.01
    transitions = []
    for frame_diff in differences:
        correlation = float(np.mean(frame_diff * pattern))
        if correlation > threshold:
            transitions.append(1)
        elif correlation < -threshold:
            transitions.append(-1)
        else:
            transitions.append(0)
    stream = [0]
    for transition in transitions:
        stream.append(stream[-1] ^ (1 if transition != 0 else 0))

    def aggregate(flip: bool) -> list[int]:
        bits = []
        for index in range(total):
            votes = [stream[position] for position in range(index, len(stream), total)]
            value = 1 if votes and sum(votes) > len(votes) / 2 else 0
            bits.append(value ^ (1 if flip else 0))
        return bits

    candidates = [aggregate(False), aggregate(True)]
    return max(
        candidates,
        key=lambda candidate: sum(
            left == right for left, right in zip(candidate, with_sync([0] * n_bits), strict=False)
        ),
    )
