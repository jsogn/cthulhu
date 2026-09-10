// @vitest-environment jsdom
import "@testing-library/jest-dom/vitest";
import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { Tabs } from "@/components/ui/tabs";
import { CleanPane } from "@/components/workbench/panes/CleanPane";
import { useCleanPanel } from "@/stores/cleanPanel";
import type { TemplateInfo } from "@/lib/backend";
import type { Material } from "@/stores/materials";

const initialCleanState = useCleanPanel.getState();

beforeEach(() => {
  useCleanPanel.setState(initialCleanState, true);
});

afterEach(cleanup);

describe("CleanPane", () => {
  it("默认展示手动模板与重新编码说明", () => {
    render(
      <Tabs defaultValue="清洗去重">
        <CleanPane
          material={null}
          templateList={[]}
          applyTemplateById={vi.fn()}
          cleanSubmitting={false}
          runClean={vi.fn()}
        />
      </Tabs>,
    );
    expect(screen.getByText("不使用模板（手动）")).toBeInTheDocument();
    expect(screen.getByText("画面变换 · 去同步与色彩微扰")).toBeInTheDocument();
    expect(screen.getByText("去水印 · 画面重建")).toBeInTheDocument();
  });

  it("提交中禁用执行按钮并回传 runClean", async () => {
    const user = userEvent.setup();
    const runClean = vi.fn();
    render(
      <Tabs defaultValue="清洗去重">
        <CleanPane
          material={null}
          templateList={[]}
          applyTemplateById={vi.fn()}
          cleanSubmitting
          runClean={runClean}
        />
      </Tabs>,
    );
    const button = screen.getByRole("button", { name: "已加入队列" });
    expect(button).toBeDisabled();
    await user.click(button);
    expect(runClean).not.toHaveBeenCalled();
  });

  it("模板下拉选择调用 applyTemplateById", async () => {
    const user = userEvent.setup();
    const applyTemplateById = vi.fn();
    const template: TemplateInfo = {
      id: "t1",
      name: "标准 · 均衡",
      payload: {},
      created_at: 1,
    };
    render(
      <Tabs defaultValue="清洗去重">
        <CleanPane
          material={null}
          templateList={[template]}
          applyTemplateById={applyTemplateById}
          cleanSubmitting={false}
          runClean={vi.fn()}
        />
      </Tabs>,
    );
    await user.click(screen.getAllByRole("combobox")[0]);
    await user.click(await screen.findByText("标准 · 均衡"));
    expect(applyTemplateById).toHaveBeenCalledWith("t1");
  });


  it("潜空间净化档位切换会写入边缘与批处理，σ 可调", async () => {
    useCleanPanel.getState().setPurifyOn(true);
    render(
      <Tabs defaultValue="清洗去重">
        <CleanPane
          material={null}
          templateList={[]}
          applyTemplateById={vi.fn()}
          cleanSubmitting={false}
          runClean={vi.fn()}
        />
      </Tabs>,
    );
    // 净化只有一个入口：档位选择器（默认画质优先），σ 与字幕增强由档位决定。
    expect(screen.queryByText("快捷预设")).not.toBeInTheDocument();
    expect(screen.getByText("清晰度档位")).toBeInTheDocument();
    expect(
      screen.getByText(/画面清晰、人脸与字幕正常；水印清除较弱/),
    ).toBeInTheDocument();
    expect(useCleanPanel.getState().purifyMaxEdge).toBe(512);
    expect(useCleanPanel.getState().purifyDetailWide).toBe(true);
  });

  it("展示选中素材的文件信息", () => {
    const material: Material = {
      id: "m1",
      name: "素材.mp4",
      dur: "00:10",
      res: "1280×720",
      fps: "30fps",
      size: "10 MB",
      codec: "h264",
      tags: ["本地"],
      frame: "/frame.png",
      path: "/tmp/素材.mp4",
    };
    render(
      <Tabs defaultValue="清洗去重">
        <CleanPane
          material={material}
          templateList={[]}
          applyTemplateById={vi.fn()}
          cleanSubmitting={false}
          runClean={vi.fn()}
        />
      </Tabs>,
    );
    expect(screen.getByText("文件信息")).toBeInTheDocument();
    expect(screen.getByText("h264 · 1280×720")).toBeInTheDocument();
    expect(screen.getByText("30fps · 00:10")).toBeInTheDocument();
  });
});
