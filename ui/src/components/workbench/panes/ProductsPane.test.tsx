// @vitest-environment jsdom
import "@testing-library/jest-dom/vitest";
import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { Tabs } from "@/components/ui/tabs";
import { ProductsPane } from "@/components/workbench/panes/ProductsPane";
import type { OutputInfo } from "@/lib/backend";

const outputs: OutputInfo[] = [
  { kind: "cleaned", path: "/out/清洗.mp4", name: "清洗.mp4", size: 1024, mtime: 1724803200 },
  { kind: "repaired", path: "/out/修复.mp4", name: "修复.mp4", size: 2048, mtime: 1724803260 },
];

function renderPane(props: Partial<React.ComponentProps<typeof ProductsPane>> = {}) {
  return render(
    <Tabs defaultValue="处理产物">
      <ProductsPane
        outputs={outputs}
        compareOriginal={vi.fn()}
        showVariant={vi.fn()}
        setPlayerPath={vi.fn()}
        setPendingDelete={vi.fn()}
        {...props}
      />
    </Tabs>,
  );
}

afterEach(cleanup);

describe("ProductsPane", () => {
  it("列出产物名称、类型与元信息", () => {
    renderPane();
    expect(screen.getByText("清洗.mp4")).toBeInTheDocument();
    expect(screen.getByText("修复.mp4")).toBeInTheDocument();
    expect(screen.getAllByText("清洗")).toHaveLength(1);
    expect(screen.getAllByText("修复")).toHaveLength(1);
  });

  it("空列表展示引导文案", () => {
    renderPane({ outputs: [] });
    expect(screen.getByText("还没有处理产物，清洗或修复后会自动出现。")).toBeInTheDocument();
  });

  it("各操作按钮回传对应回调", async () => {
    const user = userEvent.setup();
    const compareOriginal = vi.fn();
    const showVariant = vi.fn();
    const setPlayerPath = vi.fn();
    const setPendingDelete = vi.fn();
    renderPane({ compareOriginal, showVariant, setPlayerPath, setPendingDelete });

    await user.click(screen.getAllByText("对比原片")[0]);
    expect(compareOriginal).toHaveBeenCalledWith(outputs[0]);

    await user.click(screen.getAllByText("播放")[0]);
    expect(setPlayerPath).toHaveBeenCalledWith("/out/清洗.mp4");

    await user.click(screen.getAllByText("参数")[0]);
    expect(showVariant).toHaveBeenCalledWith("/out/清洗.mp4");

    await user.click(screen.getAllByText("删除")[0]);
    expect(setPendingDelete).toHaveBeenCalledWith(outputs[0]);
  });
});
