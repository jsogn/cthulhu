// @vitest-environment jsdom
import "@testing-library/jest-dom/vitest";
import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { Tabs } from "@/components/ui/tabs";
import { RegionsPane } from "@/components/workbench/panes/RegionsPane";
import { useRegionsStore } from "@/stores/regions";

beforeEach(() => {
  useRegionsStore.setState({ regions: [], activeId: null, undoStack: [], redoStack: [] });
});

afterEach(cleanup);

describe("RegionsPane", () => {
  it("空状态展示提示与添加按钮", () => {
    render(
      <Tabs defaultValue="水印区域">
        <RegionsPane onRunRepair={vi.fn()} />
      </Tabs>,
    );
    expect(screen.getByText("还没有水印区域，点击下方按钮添加")).toBeInTheDocument();
  });

  it("添加区域后出现卡片并可撤销", async () => {
    const user = userEvent.setup();
    render(
      <Tabs defaultValue="水印区域">
        <RegionsPane onRunRepair={vi.fn()} />
      </Tabs>,
    );

    await user.click(screen.getByRole("button", { name: "添加水印区域" }));
    expect(screen.getByText("区域1 · 水印")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "撤销" }));
    expect(screen.getByText("还没有水印区域，点击下方按钮添加")).toBeInTheDocument();
  });

  it("执行修复回传回调", async () => {
    const user = userEvent.setup();
    const onRunRepair = vi.fn();
    render(
      <Tabs defaultValue="水印区域">
        <RegionsPane onRunRepair={onRunRepair} />
      </Tabs>,
    );
    await user.click(screen.getByRole("button", { name: "执行修复" }));
    expect(onRunRepair).toHaveBeenCalledOnce();
  });
});
