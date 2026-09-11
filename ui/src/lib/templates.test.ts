import { describe, expect, it } from "vitest";

import type { TemplateInfo } from "@/lib/backend";
import { DEFAULT_TEMPLATE, payloadOf } from "@/lib/templates";
import {
  SCHEMA_DEFAULT_TIER,
  TIER_DETAIL,
  TIER_DETAIL_SIGMA,
} from "@/lib/tiers";

function template(payload: Record<string, unknown>): TemplateInfo {
  return { id: "t1", name: "模板", payload, created_at: 1.0 };
}

describe("payloadOf", () => {
  it("空 payload 走兼容旧模板的默认档位", () => {
    const payload = payloadOf(template({}));
    expect(payload.rotate).toBe(0);
    expect(payload.regradeOn).toBe(true);
    expect(payload.sharpness).toBe(true);
    expect(payload.colorRestore).toBe(true);
    expect(payload.denoise).toBe(true);
    expect(payload.audioRemix).toBe(true);
  });

  it("兼容旧版 audio 字段并保留显式字段", () => {
    const payload = payloadOf(template({ audio: false, rotate: 1.2, denoise: false }));
    expect(payload.audioRemix).toBe(false);
    expect(payload.rotate).toBe(1.2);
    expect(payload.denoise).toBe(false);
  });

  it("净化与嵌入域字段原样回填", () => {
    const payload = payloadOf(
      template({
        purifyStrength: 0.25,
        purifyDetail: 0.7,
        purifyDetailSigma: 1.8,
        purifyTemporal: 0.25,
        purifyMaxEdge: 320,
        purifyBatch: 8,
        embeddingAttack: "luma",
        embeddingStrength: 0.6,
        embeddingVariant: "legacy",
        embeddingAggressive: true,
        autoProfile: true,
      }),
    );
    expect(payload.purifyStrength).toBe(0.25);
    expect(payload.purifyDetail).toBe(0.7);
    expect(payload.purifyDetailSigma).toBe(1.8);
    expect(payload.purifyTemporal).toBe(0.25);
    expect(payload.purifyMaxEdge).toBe(320);
    expect(payload.purifyBatch).toBe(8);
    expect(payload.embeddingAttack).toBe("luma");
    expect(payload.embeddingStrength).toBe(0.6);
    expect(payload.embeddingVariant).toBe("legacy");
    expect(payload.embeddingAggressive).toBe(true);
    expect(payload.autoProfile).toBe(true);
  });

  it("新版完整参数优先于兼容默认", () => {
    const payload = payloadOf(
      template({ audioRemix: true, codec: "H.265", resolution: "1280x720", recropOn: true }),
    );
    expect(payload.audioRemix).toBe(true);
    expect(payload.codec).toBe("H.265");
    expect(payload.resolution).toBe("1280x720");
    expect(payload.recropOn).toBe(true);
  });
});

describe("DEFAULT_TEMPLATE", () => {
  it("默认档位从关闭状态开始", () => {
    expect(DEFAULT_TEMPLATE.rotate).toBe(0);
    expect(DEFAULT_TEMPLATE.codec).toBe("H.264");
    expect(DEFAULT_TEMPLATE.resolution).toBe("保持原始分辨率");
    expect(DEFAULT_TEMPLATE.audioRemix).toBe(false);
    expect(DEFAULT_TEMPLATE.hashEpsilon).toBe(0.08);
    expect(DEFAULT_TEMPLATE.hashMode).toBe("phash");
    expect(DEFAULT_TEMPLATE.nativeTemporal).toBe(true);
    expect(DEFAULT_TEMPLATE.purifyStrength).toBe(0);
    expect(DEFAULT_TEMPLATE.purifyDetail).toBe(TIER_DETAIL);
    expect(DEFAULT_TEMPLATE.purifyDetailSigma).toBe(TIER_DETAIL_SIGMA);
    expect(DEFAULT_TEMPLATE.purifyTemporal).toBe(0);
    // 清晰度字段跟随接口默认档（后端 tiers.py 的 balanced），不再手写。
    expect(DEFAULT_TEMPLATE.purifyMaxEdge).toBe(SCHEMA_DEFAULT_TIER.max_edge);
    expect(DEFAULT_TEMPLATE.purifyDetailWide).toBe(
      SCHEMA_DEFAULT_TIER.detail_wide,
    );
    expect(DEFAULT_TEMPLATE.purifyBatch).toBe(SCHEMA_DEFAULT_TIER.batch);
    expect(DEFAULT_TEMPLATE.embeddingAttack).toBe("");
    expect(DEFAULT_TEMPLATE.embeddingStrength).toBe(0);
    expect(DEFAULT_TEMPLATE.embeddingVariant).toBe("v2");
    expect(DEFAULT_TEMPLATE.embeddingAggressive).toBe(false);
    expect(DEFAULT_TEMPLATE.autoProfile).toBe(false);
  });
});
