// @vitest-environment jsdom
import "@testing-library/jest-dom/vitest";
import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { DeleteProductDialog } from "@/components/workbench/dialogs/DeleteProductDialog";
import { ProductPlayerDialog } from "@/components/workbench/dialogs/ProductPlayerDialog";
import { VariantDetailDialog } from "@/components/workbench/dialogs/VariantDetailDialog";
import type { OutputInfo, VariantInfo } from "@/lib/backend";

afterEach(cleanup);

const output: OutputInfo = {
  kind: "cleaned",
  path: "/out/清洗.mp4",
  name: "清洗.mp4",
  size: 1024,
  mtime: 1724803200,
};

const variant: VariantInfo = {
  id: "v1",
  source: "/src/素材.mp4",
  output: "/out/清洗.mp4",
  options: { rotate: 0.2, requant: 96, noise: 0.003 },
  seed: 42,
  template_id: null,
  metrics: { ssim: 0.93 },
  created_at: 1724803200,
};

describe("VariantDetailDialog", () => {
  it("展示源素材、种子与清洗参数摘要", () => {
    render(<VariantDetailDialog variantDetail={variant} onClose={vi.fn()} />);
    expect(screen.getByText("产物参数")).toBeInTheDocument();
    expect(screen.getByText("/src/素材.mp4")).toBeInTheDocument();
    expect(screen.getByText("42")).toBeInTheDocument();
    expect(screen.getByText("噪声扰动 · 像素重量化")).toBeInTheDocument();
  });
});

describe("ProductPlayerDialog", () => {
  it("播放给定产物路径", () => {
    render(<ProductPlayerDialog playerPath="/out/清洗.mp4" onClose={vi.fn()} />);
    const video = document.querySelector("video");
    expect(video).not.toBeNull();
    expect(video?.getAttribute("src")).toContain("/api/media?path=%2Fout%2F%E6%B8%85%E6%B4%97.mp4");
  });
});

describe("DeleteProductDialog", () => {
  it("确认删除回传 onConfirm", async () => {
    const user = userEvent.setup();
    const onConfirm = vi.fn();
    render(
      <DeleteProductDialog pendingDelete={output} onClose={vi.fn()} onConfirm={onConfirm} />,
    );
    expect(screen.getByText(/将删除「清洗.mp4」/)).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "确认删除" }));
    expect(onConfirm).toHaveBeenCalledOnce();
  });
});
