"""离线导出 OpenAPI 文档与档位表，供前端生成类型/只读数据并做 CI 漂移检查。

产物：
- `backend/openapi.json`：请求/响应类型源，`pnpm --dir ui generate-api` 消费；
- `ui/src/lib/tiers.generated.ts`：清晰度档位表（唯一真值在 `tiers.py`），
  面板与模板编辑器都从这里取数值，避免同一决策多处手写后漂移（审计 R3）；
- `ui/src/lib/schemes.generated.ts`：已知来源方案家族（唯一真值在
  `transform/profile.py`），前端只展示，不再自己维护一份映射（审计 R2）。
"""

from __future__ import annotations

import json
import os
import pathlib

os.environ.setdefault("CTHULHU_AUTH_TOKEN", "openapi-export")

from cthulhu_backend.main import app
from cthulhu_backend.tiers import (
    DEFAULT_TIER_ID,
    SCHEMA_DEFAULT_TIER_ID,
    TIER_DETAIL,
    TIER_DETAIL_SIGMA,
    tiers_document,
)
from cthulhu_backend.transform.profile import scheme_document

TARGET = pathlib.Path(__file__).resolve().parents[1] / "openapi.json"
TIERS_TARGET = (
    pathlib.Path(__file__).resolve().parents[2] / "ui" / "src" / "lib" / "tiers.generated.ts"
)
SCHEMES_TARGET = (
    pathlib.Path(__file__).resolve().parents[2] / "ui" / "src" / "lib" / "schemes.generated.ts"
)


def _tiers_module() -> str:
    """把 `tiers.TIERS` 渲染成前端只读表（含 id 枚举与默认档）。"""
    tiers = tiers_document()
    rows = ",\n".join(
        "  {\n"
        f'    id: "{tier["id"]}",\n'
        f'    label: "{tier["label"]}",\n'
        f'    summary: "{tier["summary"]}",\n'
        f'    max_edge: {tier["max_edge"]},\n'
        f'    batch: {tier["batch"]},\n'
        f'    detail_wide: {str(tier["detail_wide"]).lower()},\n'
        f'    strength: {tier["strength"]},\n'
        f'    temporal: {tier["temporal"]},\n'
        f'    selectable: {str(tier["selectable"]).lower()},\n'
        "  }"
        for tier in tiers
    )
    ids = " | ".join(f'"{tier["id"]}"' for tier in tiers)
    return (
        "/** 由 backend/scripts/export_openapi.py 从 cthulhu_backend/tiers.py 生成；请勿手改。 */\n"
        "\n"
        f"export type TierId = {ids};\n"
        "\n"
        "export interface Tier {\n"
        "  id: TierId;\n"
        "  label: string;\n"
        "  summary: string;\n"
        "  max_edge: number;\n"
        "  batch: number;\n"
        "  detail_wide: boolean;\n"
        "  strength: number;\n"
        "  temporal: number;\n"
        "  selectable: boolean;\n"
        "}\n"
        "\n"
        f"export const TIER_DETAIL = {TIER_DETAIL};\n"
        f"export const TIER_DETAIL_SIGMA = {TIER_DETAIL_SIGMA};\n"
        "\n"
        "export const TIERS: readonly Tier[] = [\n" + rows + ",\n];\n"
        "\n"
        f'export const DEFAULT_TIER_ID: TierId = "{DEFAULT_TIER_ID}";\n'
        f'export const SCHEMA_DEFAULT_TIER_ID: TierId = "{SCHEMA_DEFAULT_TIER_ID}";\n'
        "\n"
        "export const SELECTABLE_TIERS: readonly Tier[] = TIERS.filter(\n"
        "  (tier) => tier.selectable,\n"
        ");\n"
    )


def _schemes_module() -> str:
    """把 `profile.SCHEME_FAMILIES` 渲染成前端只读表。"""
    families = scheme_document()
    rows = ",\n".join(
        "  {\n"
        f'    id: "{family["id"]}",\n'
        f'    label: "{family["label"]}",\n'
        f'    summary: "{family["summary"]}",\n'
        f'    attack: "{family["attack"]}",\n'
        f'    max_edge: {family["max_edge"]},\n'
        "  }"
        for family in families
    )
    ids = " | ".join(f'"{family["id"]}"' for family in families)
    return (
        "/** 由 backend/scripts/export_openapi.py 从 transform/profile.py 生成；请勿手改。 */\n"
        "\n"
        f"export type SchemeFamilyId = {ids};\n"
        "\n"
        "export interface SchemeFamily {\n"
        "  id: SchemeFamilyId;\n"
        "  label: string;\n"
        "  summary: string;\n"
        "  attack: string;\n"
        "  max_edge: number;\n"
        "}\n"
        "\n"
        "export const SCHEME_FAMILIES: readonly SchemeFamily[] = [\n" + rows + ",\n];\n"
    )


def main() -> None:
    spec = app.openapi()
    TARGET.write_text(
        json.dumps(spec, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"已写入 {TARGET}")
    TIERS_TARGET.write_text(_tiers_module(), encoding="utf-8")
    print(f"已写入 {TIERS_TARGET}")
    SCHEMES_TARGET.write_text(_schemes_module(), encoding="utf-8")
    print(f"已写入 {SCHEMES_TARGET}")


if __name__ == "__main__":
    main()
