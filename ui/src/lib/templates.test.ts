import { describe, expect, it } from "vitest";

import type { TemplateInfo } from "@/lib/backend";
import { DEFAULT_TEMPLATE, payloadOf } from "@/lib/templates";

function template(payload: Record<string, unknown>): TemplateInfo {
  return { id: "t1", name: "模板", payload, created_at: 1.0 };
}

describe("payloadOf", () => {
  it("空 payload 走兼容旧模板的默认档位", () => {
    const payload = payloadOf(template({}));
    expect(payload.anti).toBe("标准");
    expect(payload.regradeOn).toBe(true);
    expect(payload.sharpness).toBe(true);
    expect(payload.colorRestore).toBe(true);
    expect(payload.denoise).toBe(true);
    expect(payload.audioRemix).toBe(true);
  });

  it("兼容旧版 audio 字段并保留显式字段", () => {
    const payload = payloadOf(template({ audio: false, anti: "轻度", denoise: false }));
    expect(payload.audioRemix).toBe(false);
    expect(payload.anti).toBe("轻度");
    expect(payload.denoise).toBe(false);
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
    expect(DEFAULT_TEMPLATE.anti).toBe("关闭");
    expect(DEFAULT_TEMPLATE.codec).toBe("H.264");
    expect(DEFAULT_TEMPLATE.resolution).toBe("保持原始分辨率");
    expect(DEFAULT_TEMPLATE.audioRemix).toBe(false);
  });
});
