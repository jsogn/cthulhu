// @vitest-environment jsdom
import "@testing-library/jest-dom/vitest";
import { cleanup, renderHook, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("@/lib/backend", () => ({
  listTemplates: vi.fn(),
  fetchOutputs: vi.fn(),
  deleteOutput: vi.fn(),
  listVariants: vi.fn(),
}));

import { fetchOutputs, listTemplates, type TemplateInfo } from "@/lib/backend";
import { useOutputsFlow } from "@/components/workbench/useOutputsFlow";
import { useTemplateFlow } from "@/components/workbench/useTemplateFlow";
import { useCleanPanel } from "@/stores/cleanPanel";
import type { Material } from "@/stores/materials";

function material(overrides: Partial<Material> = {}): Material {
  return {
    id: "m1",
    name: "素材.mp4",
    dur: "00:10",
    res: "1280×720",
    fps: "30",
    size: "10MB",
    tags: [],
    frame: "/frame.png",
    path: "/tmp/素材.mp4",
    ...overrides,
  };
}

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
  useCleanPanel.getState().resetCleanDefaults();
  useCleanPanel.getState().setTemplateId("manual");
});

describe("useTemplateFlow", () => {
  it("加载模板列表并支持套用模板", async () => {
    const tpl: TemplateInfo = {
      id: "t1",
      name: "标准 · 均衡",
      payload: { rotate: 0.2, requant: 96, noise: 0.003 },
      created_at: 1,
    };
    vi.mocked(listTemplates).mockResolvedValue([tpl]);
    const { result } = renderHook(() => useTemplateFlow());
    await waitFor(() => expect(result.current.templateList).toEqual([tpl]));

    result.current.applyTemplateById("t1");
    expect(useCleanPanel.getState().templateId).toBe("t1");
  });

  it("选择手动模板时重置清洗参数", async () => {
    vi.mocked(listTemplates).mockResolvedValue([]);
    const { result } = renderHook(() => useTemplateFlow());
    await waitFor(() => expect(result.current.templateList).toEqual([]));
    useCleanPanel.getState().setHashOn(true);
    result.current.applyTemplateById("manual");
    expect(useCleanPanel.getState().hashOn).toBe(false);
  });
});

describe("useOutputsFlow", () => {
  it("按素材路径刷新产物清单", async () => {
    vi.mocked(fetchOutputs).mockResolvedValue({
      source: "/tmp/素材.mp4",
      outputs: [
        { kind: "cleaned", path: "/tmp/清洗.mp4", name: "清洗.mp4", size: 1024, mtime: 1 },
      ],
    });
    const { result } = renderHook(() => useOutputsFlow(material()));
    await waitFor(() => expect(result.current.outputs).toHaveLength(1));
    expect(result.current.outputs[0].name).toBe("清洗.mp4");
  });
});
