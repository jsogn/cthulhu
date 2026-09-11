import { describe, expect, it } from "vitest";

import {
  DEFAULT_TIER,
  DEFAULT_TIER_ID,
  SCHEMA_DEFAULT_TIER,
  SELECTABLE_TIERS,
  TIER_BY_ID,
  TIERS,
  tierById,
  tierClarity,
  tierIdOf,
} from "@/lib/tiers";

describe("tiers（后端 tiers.py 生成）", () => {
  it("档位表内容与后端一致：画质优先 512/宽带回填，平衡 256，清除优先 192", () => {
    expect(DEFAULT_TIER_ID).toBe("quality");
    expect(DEFAULT_TIER.max_edge).toBe(512);
    expect(DEFAULT_TIER.detail_wide).toBe(true);
    expect(DEFAULT_TIER.strength).toBe(0.15);
    expect(TIER_BY_ID.balanced.max_edge).toBe(256);
    expect(TIER_BY_ID.strong.max_edge).toBe(192);
    expect(TIER_BY_ID.strong.temporal).toBe(0.5);
    expect(SCHEMA_DEFAULT_TIER.max_edge).toBe(256);
  });

  it("下拉只列可选档位（max 仅服务「深度清剿」预置）", () => {
    expect(SELECTABLE_TIERS.map((tier) => tier.id)).toEqual([
      "quality",
      "balanced",
      "strong",
    ]);
    expect(TIERS.find((tier) => tier.id === "max")?.selectable).toBe(false);
  });

  it("档位映射与反查一一对应", () => {
    for (const tier of SELECTABLE_TIERS) {
      expect(tierClarity(tier)).toEqual({
        purifyMaxEdge: tier.max_edge,
        purifyBatch: tier.batch,
        purifyDetailWide: tier.detail_wide,
      });
      expect(tierIdOf(tier.max_edge, tier.detail_wide)).toBe(tier.id);
    }
    // 非常规组合回落到平衡档（历史行为）。
    expect(tierIdOf(320, false)).toBe("balanced");
    expect(tierById("max")?.max_edge).toBe(128);
  });
});
