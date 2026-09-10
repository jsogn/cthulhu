import { describe, expect, it } from "vitest";

import { cleanOptionRows, outputTimeLabel } from "@/lib/workbenchOptions";

describe("cleanOptionRows", () => {
  it("经典指纹按实际原语展示与输出编码", () => {
    const rows = cleanOptionRows({
      rotate: 0.2,
      requant: 96,
      noise: 0.003,
      codec: "libx265",
    });
    expect(rows).toContainEqual({ label: "经典指纹", value: "噪声扰动 · 像素重量化" });
    expect(rows).toContainEqual({ label: "输出编码", value: "H.265" });
  });

  it("重新构图按百分比展示裁剪幅度", () => {
    const rows = cleanOptionRows({ recrop: 0.03 });
    expect(rows).toContainEqual({ label: "重新构图", value: "裁左/下各 3%" });
  });
});

describe("outputTimeLabel", () => {
  it("输出本地时间标签", () => {
    expect(outputTimeLabel(1724803200)).toMatch(/^\d{4}-\d{2}-\d{2} \d{2}:\d{2}$/);
  });
});
