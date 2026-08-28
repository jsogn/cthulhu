// @vitest-environment jsdom
import "@testing-library/jest-dom/vitest";
import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { Tabs } from "@/components/ui/tabs";
import { ContextTabs, TAB_KEYS } from "@/components/workbench/ContextTabs";

afterEach(() => {
  cleanup();
});

describe("ContextTabs", () => {
  it("渲染全部标签页", () => {
    render(
      <Tabs defaultValue="清洗去重">
        <ContextTabs setTab={vi.fn()} />
      </Tabs>,
    );
    for (const key of TAB_KEYS) {
      expect(screen.getByRole("tab", { name: key })).toBeInTheDocument();
    }
  });

  it("点击标签回传新值", async () => {
    const user = userEvent.setup();
    const setTab = vi.fn();
    render(
      <Tabs defaultValue="清洗去重" onValueChange={setTab}>
        <ContextTabs setTab={setTab} />
      </Tabs>,
    );
    await user.click(screen.getByRole("tab", { name: "检测参考" }));
    expect(setTab).toHaveBeenCalledWith("检测参考");
  });
});
