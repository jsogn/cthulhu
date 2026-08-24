"""多片段 × 多种子标定：汇总攻击矩阵，给出三档参数的稳定风险区间。"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from cthulhu_backend.evaluate import attack_matrix


def run_calibration(
    clips: list[str],
    out_dir: str,
    seeds: tuple[int, ...] = (0, 1),
) -> dict:
    rows: list[dict] = []
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    for clip in clips:
        stem = Path(clip).stem
        for seed in seeds:
            work = root / f"{stem}-s{seed}"
            report = attack_matrix.run_matrix(clip, str(work), seed=seed)
            for row in report["rows"]:
                rows.append({"clip": stem, "seed": seed, **row})

    presets = sorted({row["preset"] for row in rows})
    summary = {}
    for preset in presets:
        subset = [row for row in rows if row["preset"] == preset]
        risks = [row["duplicate_risk"] for row in subset]
        ssims = [row["ssim_aligned"] or 0 for row in subset]
        summary[preset] = {
            "n": len(subset),
            "risk_mean": round(statistics.mean(risks), 4),
            "risk_std": round(statistics.stdev(risks), 4) if len(risks) > 1 else 0,
            "risk_min": round(min(risks), 4),
            "risk_max": round(max(risks), 4),
            "ssim_mean": round(statistics.mean(ssims), 4),
        }
    report = {"clips": clips, "seeds": list(seeds), "summary": summary, "rows": rows}
    (root / "calibration.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report
