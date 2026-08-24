"""判重代理基准的纯函数回归：哈希位长、确定性与距离语义。"""

from __future__ import annotations

import numpy as np

from cthulhu_backend.evaluate import dedup_harness as dh


def test_hash_families_are_64bit_and_deterministic():
    frame = np.random.default_rng(0).random((108, 192))
    for name in ("phash", "dhash", "whash", "blockhash"):
        first = getattr(dh, name)(frame)
        second = getattr(dh, name)(frame)
        assert first.shape == (64,)
        assert np.array_equal(first, second)


def test_unrelated_frames_sit_near_chance_distance():
    a = np.random.default_rng(1).random((108, 192))
    b = np.random.default_rng(2).random((108, 192))
    distance = dh._hamming(dh.phash(a), dh.phash(b))
    assert 0.3 < distance < 0.7


def test_identical_frames_have_zero_distance():
    frames = np.random.default_rng(3).random((6, 108, 192))
    distances = dh._frame_dists(frames, frames.copy())
    assert all(value == 0.0 for value in distances.values())
    assert dh._temporal_sequence_distance(frames, frames.copy()) == 0.0


def test_synthetic_reencode_distance_is_small():
    base = np.random.default_rng(4).random((6, 108, 192))
    reencoded = np.clip(base + np.random.default_rng(5).normal(0, 0.02, base.shape), 0, 1)
    distances = dh._frame_dists(base, reencoded, search=30)
    assert all(value < 0.2 for value in distances.values())
