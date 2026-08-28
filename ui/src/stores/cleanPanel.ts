import { create } from "zustand";

import { getSettings } from "@/lib/backend";
import type { AntiLevel, Codec, TemplatePayload } from "@/lib/templates";

export type OutputMode = "reencode" | "remux";

/** 清洗参数默认值：初始状态与 resetCleanDefaults 共用，避免两处手写漂移。 */
const CLEAN_DEFAULTS = {
  audioClean: false,
  echoDefeat: false,
  antiReembed: false,
  antiLevel: "关闭" as AntiLevel,
  regradeOn: false,
  recropOn: false,
  detailProtectOn: false,
  sharpness: false,
  colorFix: false,
  aiDenoise: false,
  spoof: false,
};

/** 清洗设置唯一数据源：面板可写，批量清洗只读，单条/批量共用同一份参数。 */
export interface CleanPanelState {
  audioClean: boolean;
  echoDefeat: boolean;
  antiReembed: boolean;
  antiLevel: AntiLevel;
  outputMode: OutputMode;
  regradeOn: boolean;
  recropOn: boolean;
  detailProtectOn: boolean;
  sharpness: boolean;
  colorFix: boolean;
  aiDenoise: boolean;
  spoof: boolean;
  codec: Codec;
  lossless: boolean;
  resolution: string;
  templateId: string;
  settingsExportDir: string;
  settingsNaming: string;
  settingsMetricsMode: string;
  settingsFilterScale: string;
  setAudioClean: (value: boolean) => void;
  setEchoDefeat: (value: boolean) => void;
  setAntiReembed: (value: boolean) => void;
  setAntiLevel: (value: AntiLevel) => void;
  setOutputMode: (value: OutputMode) => void;
  setRegradeOn: (value: boolean) => void;
  setRecropOn: (value: boolean) => void;
  setDetailProtectOn: (value: boolean) => void;
  setSharpness: (value: boolean) => void;
  setColorFix: (value: boolean) => void;
  setAiDenoise: (value: boolean) => void;
  setSpoof: (value: boolean) => void;
  setCodec: (value: Codec) => void;
  setLossless: (value: boolean) => void;
  setResolution: (value: string) => void;
  setTemplateId: (value: string) => void;
  setSettingsExportDir: (value: string) => void;
  setSettingsNaming: (value: string) => void;
  setSettingsMetricsMode: (value: string) => void;
  setSettingsFilterScale: (value: string) => void;
  applyTemplatePayload: (payload: TemplatePayload) => void;
  resetCleanDefaults: () => void;
  loadSettings: () => Promise<void>;
}

export const useCleanPanel = create<CleanPanelState>((set) => ({
  ...CLEAN_DEFAULTS,
  outputMode: "reencode",
  codec: "H.264",
  lossless: false,
  resolution: "保持原始分辨率",
  templateId: "manual",
  settingsExportDir: "",
  settingsNaming: "原文件名 + 时间戳",
  settingsMetricsMode: "full",
  settingsFilterScale: "720",
  setAudioClean: (audioClean) => set({ audioClean }),
  setEchoDefeat: (echoDefeat) => set({ echoDefeat }),
  setAntiReembed: (antiReembed) => set({ antiReembed }),
  setAntiLevel: (antiLevel) => set({ antiLevel }),
  setOutputMode: (outputMode) => set({ outputMode }),
  setRegradeOn: (regradeOn) => set({ regradeOn }),
  setRecropOn: (recropOn) => set({ recropOn }),
  setDetailProtectOn: (detailProtectOn) => set({ detailProtectOn }),
  setSharpness: (sharpness) => set({ sharpness }),
  setColorFix: (colorFix) => set({ colorFix }),
  setAiDenoise: (aiDenoise) => set({ aiDenoise }),
  setSpoof: (spoof) => set({ spoof }),
  setCodec: (codec) => set({ codec }),
  setLossless: (lossless) => set({ lossless }),
  setResolution: (resolution) => set({ resolution }),
  setTemplateId: (templateId) => set({ templateId }),
  setSettingsExportDir: (settingsExportDir) => set({ settingsExportDir }),
  setSettingsNaming: (settingsNaming) => set({ settingsNaming }),
  setSettingsMetricsMode: (settingsMetricsMode) => set({ settingsMetricsMode }),
  setSettingsFilterScale: (settingsFilterScale) => set({ settingsFilterScale }),
  applyTemplatePayload: (payload) =>
    set({
      audioClean: payload.audioRemix,
      echoDefeat: payload.echoDefeat,
      antiReembed: payload.antiReembed,
      antiLevel: payload.anti,
      regradeOn: payload.regradeOn,
      recropOn: payload.recropOn,
      detailProtectOn: payload.detailProtectOn,
      sharpness: payload.sharpness,
      colorFix: payload.colorRestore,
      aiDenoise: payload.denoise,
      spoof: payload.spoof,
      codec: payload.codec,
      lossless: payload.lossless,
      resolution: payload.resolution,
    }),
  resetCleanDefaults: () => set(CLEAN_DEFAULTS),
  loadSettings: async () => {
    try {
      const settings = await getSettings();
      set({
        settingsExportDir: (settings.export_dir as string) ?? "",
        settingsNaming: (settings.naming as string) ?? "原文件名 + 时间戳",
        settingsMetricsMode: (settings.metrics_mode as string) ?? "fast",
        settingsFilterScale: (settings.filter_scale as string) ?? "720",
      });
    } catch {
      // 后端未就绪时保持默认设置。
    }
  },
}));
