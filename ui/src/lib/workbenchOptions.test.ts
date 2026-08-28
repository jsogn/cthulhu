import { describe, expect, it } from "vitest";

import {
  antiLevelOf,
  cleanOptionRows,
  outputTimeLabel,
  riskLabel,
} from "@/lib/workbenchOptions";

describe("antiLevelOf", () => {
  it("识别标准档参数组合", () => {
    expect(antiLevelOf({ rotate: 0.2, requant: 96, noise: 0.003 })).toBe("标准");
  });

  it("自定义组合返回 null", () => {
    expect(antiLevelOf({ rotate: 0.2 })).toBeNull();
    expect(antiLevelOf({})).toBeNull();
  });
});

describe("cleanOptionRows", () => {
  it("remux 模式标注为只换壳", () => {
    const rows = cleanOptionRows({ output_mode: "remux" });
    expect(rows[0]).toEqual({ label: "清洗方式", value: "重新封装 · 只换壳" });
  });

  it("标准档展示档位名与输出编码", () => {
    const rows = cleanOptionRows({
      output_mode: "reencode",
      rotate: 0.2,
      requant: 96,
      noise: 0.003,
      codec: "libx265",
    });
    expect(rows).toContainEqual({ label: "指纹对抗", value: "标准" });
    expect(rows).toContainEqual({ label: "输出编码", value: "H.265" });
  });

  it("重新构图按百分比展示裁剪幅度", () => {
    const rows = cleanOptionRows({ output_mode: "reencode", recrop: 0.03 });
    expect(rows).toContainEqual({ label: "重新构图", value: "裁左/下各 3%" });
  });
});

describe("outputTimeLabel / riskLabel", () => {
  it("输出本地时间标签与风险文案", () => {
    expect(outputTimeLabel(1724803200)).toMatch(/^\d{4}-\d{2}-\d{2} \d{2}:\d{2}$/);
    expect(riskLabel("有疑似特征")).toBe("有疑似特征");
  });
});
