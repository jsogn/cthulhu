"""预设一致性审计：payload 与 schema 校验、开启项清单、映射到后端选项。"""

from __future__ import annotations

import sys

from cthulhu_backend import db
from cthulhu_backend.schemas import TemplatePayload

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from bench_preset_matrix import payload_to_options


def main() -> None:
    print(f"PRESET_VERSION={db.PRESET_VERSION}，档位数={len(db.PRESET_TEMPLATES)}")
    removed = {
        "transcodeChain", "srRewrite", "clahe", "deepAttack",
        "transcode_chain", "sr_rewrite", "deep_attack", "mirror",
        # §7 无效参数清理（2026-09-11）：payload 与后端选项两侧都不得再出现。
        "fftMag", "nonintRatio", "flowDisturb", "textureInject", "multiscale",
        "complexityTrap", "temporalBlur",
        "fft_mag", "nonint_ratio", "flow_disturb", "texture_inject",
        "complexity_trap", "temporal_blur",
    }
    for preset in db.PRESET_TEMPLATES:
        payload = preset["payload"]
        # schema 校验（未知字段静默丢弃的字段会被抓出来）。
        validated = TemplatePayload(**payload).model_dump()
        unknown = set(payload) - set(validated)
        stale = set(payload) & removed
        on = sorted(k for k, v in validated.items() if v not in (False, 0, 0.0, None, ""))
        opts = payload_to_options(payload)
        opts_stale = set(opts) & removed
        flags = []
        if unknown:
            flags.append(f"未知字段={sorted(unknown)}")
        if stale:
            flags.append(f"残留过时字段={sorted(stale)}")
        if opts_stale:
            flags.append(f"映射残留={sorted(opts_stale)}")
        status = f"  {'⚠ ' + '; '.join(flags) if flags else 'OK'}"
        print(f"{preset['name']}: {status}")
        print(f"    开启项: {on}")


if __name__ == "__main__":
    main()
