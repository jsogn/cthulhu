// @vitest-environment jsdom
import "@testing-library/jest-dom/vitest";
import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { Tabs } from "@/components/ui/tabs";
import { CleanPane } from "@/components/workbench/panes/CleanPane";
import { useCleanPanel } from "@/stores/cleanPanel";
import type { TemplateInfo } from "@/lib/backend";

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
          templateList={[]}
          applyTemplateById={vi.fn()}
          cleanSubmitting={false}
          runClean={vi.fn()}
        />
      </Tabs>,
    );
    expect(screen.getByText("不使用模板（手动）")).toBeInTheDocument();
    expect(
      screen.getByText("重新解码并压缩画面与音轨，下方对抗设置真正生效，清洗更彻底"),
    ).toBeInTheDocument();
  });

  it("切到重新封装后写入 store 并提示无损语义", async () => {
    const user = userEvent.setup();
    render(
      <Tabs defaultValue="清洗去重">
        <CleanPane
          templateList={[]}
          applyTemplateById={vi.fn()}
          cleanSubmitting={false}
          runClean={vi.fn()}
        />
      </Tabs>,
    );
    await user.click(screen.getByRole("button", { name: "重新封装" }));
    expect(useCleanPanel.getState().outputMode).toBe("remux");
    expect(
      screen.getByText(/仅重写封装格式与元数据/),
    ).toBeInTheDocument();
  });

  it("提交中禁用执行按钮并回传 runClean", async () => {
    const user = userEvent.setup();
    const runClean = vi.fn();
    render(
      <Tabs defaultValue="清洗去重">
        <CleanPane
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
});
