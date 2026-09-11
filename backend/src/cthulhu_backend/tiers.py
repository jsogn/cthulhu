"""清晰度档位（画面重建的「档」）：面板、模板编辑器与内置预置模板的唯一真值。

这条决策以前散落在 `schemas` 默认值、`db.PRESET_TEMPLATES`、
`cleanPanel.CLEAN_DEFAULTS`、`templates.DEFAULT_TEMPLATE` 与两个视图的映射里，
已经造成过两次真实故障（预置 σ 改了但 PRESET_VERSION 没升；「画质优先」在模板
编辑器与工作台取到不同的长边）。现在：

- 后端：`TIERS` 是唯一表，`db.PRESET_TEMPLATES`、`schemas` 默认值都从它派生；
- 前端：`ui/src/lib/tiers.generated.ts` 由 `backend/scripts/export_openapi.py`
  生成（与 `api-types.ts` 同一条类型同步流水线），视图只按 `id` 枚举取表。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Tier:
    """一个清晰度档位：净化长边、批大小、细节回填策略与该档的默认强度。"""

    id: str
    label: str
    summary: str
    max_edge: int
    batch: int
    detail_wide: bool
    strength: float
    temporal: float
    # 面板与模板编辑器的下拉只列 selectable 档位；`max` 供「深度清剿」预置专用。
    selectable: bool = True


TIERS: tuple[Tier, ...] = (
    Tier(
        id="quality",
        label="画质优先",
        summary="画面清晰、人脸与字幕正常；水印清除较弱，适合要成片质量的场景",
        max_edge=512,
        batch=8,
        detail_wide=True,
        strength=0.15,
        temporal=0.0,
    ),
    Tier(
        id="balanced",
        label="平衡",
        summary="画面略软，清除率比画质优先更好，是人人都能接受的中间档",
        max_edge=256,
        batch=8,
        detail_wide=False,
        strength=0.15,
        temporal=0.0,
    ),
    Tier(
        id="strong",
        label="清除优先",
        summary="水印清除最彻底；画面会明显变软、字幕可能难以辨认",
        max_edge=192,
        batch=8,
        detail_wide=False,
        strength=0.35,
        temporal=0.5,
    ),
    Tier(
        id="max",
        label="彻底清剿",
        summary="最激进的长边，只供「深度清剿」预置使用",
        max_edge=128,
        batch=8,
        detail_wide=False,
        strength=0.35,
        temporal=0.5,
        selectable=False,
    ),
)

TIER_BY_ID: dict[str, Tier] = {tier.id: tier for tier in TIERS}

# 面板与模板编辑器的出厂默认档（＝预置「画质优先（推荐）」）。
DEFAULT_TIER_ID = "quality"
# 接口字段默认值取中间档：客户端不传净化参数时行为与历史一致。
SCHEMA_DEFAULT_TIER_ID = "balanced"

# 细节回注强度／带宽（σ=0 表示按分辨率自动换算），四个档位一致。
TIER_DETAIL = 1.0
TIER_DETAIL_SIGMA = 0.0


def tier_clarity(tier_id: str) -> dict:
    """档位的清晰度字段（camelCase，与 TemplatePayload 同名），供预置与前端派生。"""
    tier = TIER_BY_ID[tier_id]
    return {
        "purifyMaxEdge": tier.max_edge,
        "purifyBatch": tier.batch,
        "purifyDetailWide": tier.detail_wide,
        "purifyDetail": TIER_DETAIL,
        "purifyDetailSigma": TIER_DETAIL_SIGMA,
    }


def tier_purify(tier_id: str) -> dict:
    """档位完整字段：清晰度 + 该档默认强度/时序，供预置模板与面板默认值复用。"""
    return {
        **tier_clarity(tier_id),
        "purifyStrength": TIER_BY_ID[tier_id].strength,
        "purifyTemporal": TIER_BY_ID[tier_id].temporal,
    }


def tiers_document() -> list[dict]:
    """给前端生成的只读档位表（顺序即展示顺序）。"""
    return [asdict(tier) for tier in TIERS]
