// @vitest-environment jsdom
import "@testing-library/jest-dom/vitest";
import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { Tabs } from "@/components/ui/tabs";
import { DetectPane } from "@/components/workbench/panes/DetectPane";
import type { DetectReport } from "@/lib/backend";
import type { Material } from "@/stores/materials";

const report: DetectReport = {
  probe: { codec: "h264", width: 1280, height: 720, fps: 30, duration: 321.9 },
  container: { suspicious: ["xmp 异常"] },
  sei_count: 0,
  bitstream: { score: 25, level: "疑似", flags: ["QP 序列周期性较强"] },
  blind: { ss: 0.99, qim: 0.78, lsb: 0.1 },
  blind_confidence: { ss: 0.99, qim: 0.89 },
  hits: ["ss"],
};

function material(overrides: Partial<Material> = {}): Material {
  return {
    id: "m1",
    name: "素材.mp4",
    dur: "05:22",
    res: "1280×720",
    fps: "30",
    size: "10MB",
    risk: "有疑似特征",
    score: 25,
    tags: [],
    frame: "/frame.png",
    path: "/tmp/素材.mp4",
    ...overrides,
  };
}

function renderPane(props: Partial<React.ComponentProps<typeof DetectPane>> = {}) {
  return render(
    <Tabs defaultValue="检测参考">
      <DetectPane
        material={material()}
        report={report}
        risk="有疑似特征"
        score={25}
        detecting={false}
        detectSubmitting={false}
        audio={null}
        audioMissing={false}
        hashOn={false}
        setHashOn={vi.fn()}
        setDctOn={vi.fn()}
        onDetect={vi.fn()}
        {...props}
      />
    </Tabs>,
  );
}

afterEach(cleanup);

describe("DetectPane", () => {
  it("有报告时展示文件信息与重新检测", async () => {
    const user = userEvent.setup();
    const onDetect = vi.fn();
    renderPane({ onDetect });
    expect(screen.getByText("h264 · 1280×720")).toBeInTheDocument();
    expect(screen.getByText("QP 序列周期性较强")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "重新检测" }));
    expect(onDetect).toHaveBeenCalledOnce();
  });

  it("高分时给出经典指纹一键开启建议", async () => {
    const user = userEvent.setup();
    const setHashOn = vi.fn();
    const setDctOn = vi.fn();
    renderPane({ score: 60, setHashOn, setDctOn });
    await user.click(screen.getByRole("button", { name: "一键开启" }));
    expect(setHashOn).toHaveBeenCalledWith(true);
    expect(setDctOn).toHaveBeenCalledWith(true);
  });

  it("无报告时展示检测引导", () => {
    renderPane({ report: null, material: material({ path: "/tmp/素材.mp4" }) });
    expect(screen.getByRole("button", { name: "检测此素材" })).toBeInTheDocument();
    expect(screen.getByText(/尚未检测，检测后此处展示/)).toBeInTheDocument();
  });
});
