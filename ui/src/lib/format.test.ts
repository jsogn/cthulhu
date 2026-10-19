import { describe, expect, it } from "vitest";

import { fmtClock, fmtFps } from "@/lib/format";

describe("fmtFps", () => {
  it("去掉整数帧率的浮点尾差", () => {
    expect(fmtFps(30.000051542796154)).toBe("30");
    expect(fmtFps(25)).toBe("25");
  });

  it("保留常见的非整数帧率", () => {
    expect(fmtFps(29.97)).toBe("29.97");
    expect(fmtFps(23.976)).toBe("23.976");
    expect(fmtFps(59.9401)).toBe("59.94");
  });

  it("无效帧率显示占位符", () => {
    expect(fmtFps(null)).toBe("—");
    expect(fmtFps(0)).toBe("—");
  });
});

describe("fmtClock", () => {
  it("按视频时长展示，不拼接帧号", () => {
    expect(fmtClock(0)).toBe("00:00");
    expect(fmtClock(1.27)).toBe("00:01");
    expect(fmtClock(189.43)).toBe("03:09");
    expect(fmtClock(3661)).toBe("01:01:01");
  });
});
