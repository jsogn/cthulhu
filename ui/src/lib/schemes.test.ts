import { describe, expect, it } from "vitest";

import { SCHEME_FAMILIES } from "@/lib/schemes.generated";

describe("schemes（后端 profile.py 生成）", () => {
  it("下拉只列引擎能定向的两个家族，且带收窄规则说明", () => {
    expect(SCHEME_FAMILIES.map((family) => family.id)).toEqual(["luma", "chroma"]);
    for (const family of SCHEME_FAMILIES) {
      expect(family.attack).toBe(family.id);
      expect(family.label).not.toBe("");
      expect(family.summary).not.toBe("");
    }
  });

  it("亮度型必须比色度型收得更狠（research §19.7）", () => {
    const luma = SCHEME_FAMILIES.find((family) => family.id === "luma");
    const chroma = SCHEME_FAMILIES.find((family) => family.id === "chroma");
    expect(luma?.max_edge).toBe(192);
    expect(chroma?.max_edge).toBe(256);
  });
});
