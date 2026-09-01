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
      rotate: 1.2,
      hashAttack: true,
      hashEpsilon: 0.05,
      hashMode: "dhash",
      requant: 64,
      noise: 0.004,
      dctStep: 12,
      audioStrong: true,
      regradeOn: true,
      recropOn: true,
      temporalSub: 1.0,
      fftPhase: 0.4,
      fftMag: 0.7,
      dwtDetail: 0,
      hsvJitter: 0,
      nonintRatio: 0.02,
      jitter: 0.01,
      perspective: 0.02,
      warp: 0.004,
      flowDisturb: 1.5,
      textureInject: 0.03,
      multiscale: 0.03,
      complexityTrap: 0.07,
      facePerturb: 0.05,
      temporalBlur: 0.3,
      lpcAttack: 0.9,
      copyAttack: 0.06,
      nativeTemporal: true,
      qualityProtect: true,
      psnrTarget: 40,
      ssimTarget: 0.96,
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
    expect(state.rotateOn).toBe(true);
    expect(state.rotate).toBe(1.2);
    expect(state.hashOn).toBe(true);
    expect(state.hashEps).toBe(0.05);
    expect(state.hashMode).toBe("dhash");
    expect(state.requantOn).toBe(true);
    expect(state.requant).toBe(64);
    expect(state.noiseOn).toBe(true);
    expect(state.noise).toBe(0.004);
    expect(state.dctOn).toBe(true);
    expect(state.audioStrongOn).toBe(true);
    expect(state.codec).toBe("H.265");
    expect(state.lossless).toBe(true);
    expect(state.resolution).toBe("1280x720");
    expect(state.temporalSubOn).toBe(true);
    expect(state.temporalSub).toBe(1.0);
    expect(state.fftOn).toBe(true);
    expect(state.fftPhase).toBe(0.4);
    expect(state.fftMag).toBe(0.7);
    expect(state.dwtDetailOn).toBe(false);
    expect(state.warpOn).toBe(true);
    expect(state.warp).toBe(0.004);
    expect(state.shearOn).toBe(true);
    expect(state.shear).toBe(0.02);
    expect(state.jitterOn).toBe(true);
    expect(state.jitter).toBe(0.01);
    expect(state.nonintOn).toBe(true);
    expect(state.nonintRatio).toBe(0.02);
    expect(state.flowOn).toBe(true);
    expect(state.flow).toBe(1.5);
    expect(state.textureOn).toBe(true);
    expect(state.texture).toBe(0.03);
    expect(state.multiscaleOn).toBe(true);
    expect(state.multiscale).toBe(0.03);
    expect(state.faceOn).toBe(true);
    expect(state.face).toBe(0.05);
    expect(state.temporalBlurOn).toBe(true);
    expect(state.temporalBlur).toBe(0.3);
    expect(state.lpcOn).toBe(true);
    expect(state.lpc).toBe(0.9);
    expect(state.neuralOn).toBe(true);
    expect(state.neural).toBe(0.06);
    expect(state.nativeTemporalOn).toBe(true);
    expect(state.qualityProtectOn).toBe(true);
    expect(state.psnrTarget).toBe(40);
    expect(state.ssimTarget).toBe(0.96);
  });

  it("resetCleanDefaults 只复位清洗参数，不动输出编码设置", () => {
    useCleanPanel.getState().setHashOn(true);
    useCleanPanel.getState().setOutputMode("remux");
    useCleanPanel.getState().setCodec("H.265");
    expect(useCleanPanel.getState().hashOn).toBe(true);
    expect(useCleanPanel.getState().outputMode).toBe("remux");
    expect(useCleanPanel.getState().codec).toBe("H.265");
    useCleanPanel.getState().resetCleanDefaults();
    expect(useCleanPanel.getState().hashOn).toBe(false);
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
