#!/usr/bin/env python3
"""eot_core 自检：注册表、损失下降与扰动预算（研究侧，无需 pytest）。

运行：research/.venv/bin/python research/scripts/check_eot_core.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import eot_core as eot  # noqa: E402


def main() -> None:
    eot.register_scheme("probe", lambda frames, bits: frames, lambda x: x)
    assert eot.get_scheme("probe").name == "probe"
    try:
        eot.get_scheme("missing")
    except KeyError:
        pass
    else:
        raise AssertionError("未注册方案应抛 KeyError")

    import torch

    rng = np.random.default_rng(0)
    frames = rng.uniform(0.0, 1.0, (12, 16, 16, 3)).astype(np.float32)
    pattern = torch.randn(3, 16, 16, generator=torch.Generator().manual_seed(1))

    def decode(x):
        return (x * pattern.to(x.device)).flatten(1).mean(dim=1, keepdim=True).repeat(1, 4)

    def logits_sq(x) -> float:
        return float((decode(x) ** 2).mean().item())

    before = logits_sq(torch.from_numpy(frames).permute(0, 3, 1, 2))
    options = eot.EotOptions(eps=2.0 / 255.0, steps=10, chunk=5, mode="modulated", eot=False)
    out = eot.run(frames, decode, options, seed=0)
    after = logits_sq(torch.from_numpy(out).permute(0, 3, 1, 2))
    assert after < before, (before, after)
    assert float(np.abs(out - frames).max()) <= 2.0 / 255.0 * 1.45 + 1e-4

    frames_static = np.repeat(frames[:1], 4, axis=0)
    static = eot.run(
        frames_static,
        decode,
        eot.EotOptions(eps=2.0 / 255.0, steps=4, chunk=4, mode="static", eot=False),
        seed=0,
    )
    for index in range(1, len(static)):
        np.testing.assert_array_equal(static[0], static[index])
    print("eot_core OK")


if __name__ == "__main__":
    main()
