import {
  DEFAULT_TIER_ID,
  SCHEMA_DEFAULT_TIER_ID,
  SELECTABLE_TIERS,
  TIER_DETAIL,
  TIER_DETAIL_SIGMA,
  TIERS,
  type Tier,
  type TierId,
} from "@/lib/tiers.generated";

export {
  DEFAULT_TIER_ID,
  SCHEMA_DEFAULT_TIER_ID,
  SELECTABLE_TIERS,
  TIER_DETAIL,
  TIER_DETAIL_SIGMA,
  TIERS,
  type Tier,
  type TierId,
};

/** 出厂默认档：面板与模板编辑器共用（数值来自后端 tiers.py）。 */
export const DEFAULT_TIER: Tier =
  TIERS.find((tier) => tier.id === DEFAULT_TIER_ID) ?? TIERS[0];

/** 按 id 取档位（枚举值由后端生成，这里只做查表）。 */
export const TIER_BY_ID: Record<TierId, Tier> = Object.fromEntries(
  TIERS.map((tier) => [tier.id, tier]),
) as Record<TierId, Tier>;

/** 接口字段默认档：老模板缺净化字段时按它回填（与后端 schemas 默认值同源）。 */
export const SCHEMA_DEFAULT_TIER: Tier = TIER_BY_ID[SCHEMA_DEFAULT_TIER_ID];

export function tierById(id: string): Tier | undefined {
  return TIERS.find((tier) => tier.id === id);
}

/** 档位 → 面板与模板编辑器共用的清晰度字段。 */
export function tierClarity(tier: Tier): {
  purifyMaxEdge: number;
  purifyBatch: number;
  purifyDetailWide: boolean;
} {
  return {
    purifyMaxEdge: tier.max_edge,
    purifyBatch: tier.batch,
    purifyDetailWide: tier.detail_wide,
  };
}

/**
 * 从当前长边与回填策略反查档位；非常规组合回落到「平衡」
 * （与旧的「其余情况算平衡档」行为一致）。
 */
export function tierIdOf(maxEdge: number, detailWide: boolean): TierId {
  const hit = SELECTABLE_TIERS.find(
    (tier) => tier.max_edge === maxEdge && tier.detail_wide === detailWide,
  );
  return hit?.id ?? "balanced";
}
