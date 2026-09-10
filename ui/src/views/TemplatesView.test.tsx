// @vitest-environment jsdom
import "@testing-library/jest-dom/vitest";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  createTemplate,
  listTemplates,
  updateTemplate,
  type TemplateInfo,
} from "@/lib/backend";
import TemplatesView from "@/views/TemplatesView";

vi.mock("@/lib/backend", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/backend")>();
  return {
    ...actual,
    listTemplates: vi.fn(),
    createTemplate: vi.fn(),
    updateTemplate: vi.fn(),
    deleteTemplate: vi.fn(),
  };
});

const rotateTemplate: TemplateInfo = {
  id: "t1",
  name: "几何模板",
  payload: {
    rotate: 1.2,
  },
  created_at: 1,
};

beforeEach(() => {
  vi.mocked(listTemplates).mockResolvedValue([]);
  vi.mocked(createTemplate).mockResolvedValue({
    id: "new",
    name: "新模板",
    payload: {},
    created_at: 1,
  });
  vi.mocked(updateTemplate).mockResolvedValue({
    id: "t1",
    name: "几何模板",
    payload: {},
    created_at: 1,
  });
});

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe("TemplatesView", () => {
  it("数值武器打开后显示滑杆，并按当前强度保存", async () => {
    const user = userEvent.setup();
    render(<TemplatesView />);
    await waitFor(() => expect(listTemplates).toHaveBeenCalled());

    await user.click(screen.getByRole("button", { name: "+ 新建模板" }));
    await user.click(screen.getByRole("switch", { name: "几何微旋转" }));

    expect(screen.getByText("旋转振幅 0.4°")).toBeInTheDocument();

    await user.type(screen.getByPlaceholderText("如：抖音投流-轻度"), "旋转模板");
    await user.click(screen.getByRole("button", { name: "保存模板" }));

    await waitFor(() => {
      expect(createTemplate).toHaveBeenCalledWith(
        "旋转模板",
        expect.objectContaining({ rotate: 0.4 }),
      );
    });
  });

  it("编辑已有模板时保留滑杆强度，关闭再打开不重置", async () => {
    const user = userEvent.setup();
    vi.mocked(listTemplates).mockResolvedValue([rotateTemplate]);
    render(<TemplatesView />);
    await screen.findByText("几何模板");

    await user.click(screen.getByRole("button", { name: "编辑" }));
    expect(screen.getByText("旋转振幅 1.2°")).toBeInTheDocument();

    const rotateSwitch = screen.getByRole("switch", { name: "几何微旋转" });
    await user.click(rotateSwitch);
    await user.click(rotateSwitch);

    expect(screen.getByText("旋转振幅 1.2°")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "保存模板" }));
    await waitFor(() => {
      expect(updateTemplate).toHaveBeenCalledWith(
        "t1",
        "几何模板",
        expect.objectContaining({ rotate: 1.2 }),
      );
    });
  });

  it("补齐哈希预算、画质门控目标和色彩微扰控件", async () => {
    const user = userEvent.setup();
    render(<TemplatesView />);
    await waitFor(() => expect(listTemplates).toHaveBeenCalled());

    await user.click(screen.getByRole("button", { name: "+ 新建模板" }));
    await user.click(screen.getByRole("switch", { name: "哈希签名对抗（pHash/dHash）" }));
    await user.click(screen.getByRole("switch", { name: "画质保护（PSNR/SSIM 门控）" }));

    expect(screen.getByText("扰动预算 ε 0.080")).toBeInTheDocument();
    expect(screen.getByText("PSNR 目标 38 dB")).toBeInTheDocument();
    expect(screen.getByText("SSIM 目标 0.94")).toBeInTheDocument();
    expect(
      screen.getByRole("switch", { name: "色彩微扰（伽马/亮度+色相）" }),
    ).toBeInTheDocument();
  });
});
