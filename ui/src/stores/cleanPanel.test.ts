// @vitest-environment jsdom
import { afterEach, describe, expect, it, vi } from "vitest";

import { useCleanPanel } from "@/stores/cleanPanel";

afterEach(() => {
  vi.unstubAllGlobals();
  useCleanPanel.getState().resetCleanDefaults();
});

describe("cleanPanel store", () => {
  it("初始状态从关闭档开始", () => {
    const state = useCleanPanel.getState();
    expect(state.antiLevel).toBe("关闭");
    expect(state.outputMode).toBe("reencode");
    expect(state.codec).toBe("H.264");
    expect(state.resolution).toBe("保持原始分辨率");
    expect(state.templateId).toBe("manual");
  });

  it("applyTemplatePayload 把模板字段映射到面板状态", () => {
    useCleanPanel.getState().applyTemplatePayload({
      audioRemix: true,
      echoDefeat: true,
      antiReembed: true,
      anti: "强力",
      regradeOn: true,
      recropOn: true,
      detailProtectOn: true,
      sharpness: true,
      colorRestore: true,
      denoise: true,
      spoof: true,
      codec: "H.265",
      lossless: true,
      resolution: "1280x720",
      bitrate: 4000,
      gop: 60,
      fpsOut: 30,
    });
    const state = useCleanPanel.getState();
    expect(state.audioClean).toBe(true);
    expect(state.echoDefeat).toBe(true);
    expect(state.antiLevel).toBe("强力");
    expect(state.codec).toBe("H.265");
    expect(state.lossless).toBe(true);
    expect(state.resolution).toBe("1280x720");
  });

  it("resetCleanDefaults 只复位清洗参数，不动输出编码设置", () => {
    useCleanPanel.getState().setAntiLevel("全兵器");
    useCleanPanel.getState().setOutputMode("remux");
    useCleanPanel.getState().setCodec("H.265");
    expect(useCleanPanel.getState().antiLevel).toBe("全兵器");
    expect(useCleanPanel.getState().outputMode).toBe("remux");
    expect(useCleanPanel.getState().codec).toBe("H.265");
    useCleanPanel.getState().resetCleanDefaults();
    expect(useCleanPanel.getState().antiLevel).toBe("关闭");
    expect(useCleanPanel.getState().outputMode).toBe("remux");
    expect(useCleanPanel.getState().codec).toBe("H.265");
  });

  it("loadSettings 拉取后端设置并写回面板", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        export_dir: "/导出目录",
        naming: "时间戳 + 原文件名",
        filter_scale: "480",
      }),
    });
    vi.stubGlobal("fetch", fetchMock);
    await useCleanPanel.getState().loadSettings();
    const state = useCleanPanel.getState();
    expect(state.settingsExportDir).toBe("/导出目录");
    expect(state.settingsNaming).toBe("时间戳 + 原文件名");
    expect(state.settingsFilterScale).toBe("480");
  });
});
