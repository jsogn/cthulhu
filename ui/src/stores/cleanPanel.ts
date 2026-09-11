import { create } from "zustand";

import { getSettings } from "@/lib/backend";
import type { Codec, TemplatePayload } from "@/lib/templates";

export type PresetBase = "balanced" | "aggressive" | "extreme" | "custom";

/** 清洗参数默认值：初始状态与 resetCleanDefaults 共用，避免两处手写漂移。 */
const CLEAN_DEFAULTS = {
  presetBase: "custom" as PresetBase,
  presetModified: false,
  audioClean: false,
  echoDefeat: false,
  rotateOn: false,
  rotate: 0.4,
  hashOn: false,
  hashEps: 0.08,
  hashMode: "phash" as "phash" | "dhash" | "joint",
  requantOn: false,
  requant: 64,
  noiseOn: false,
  noise: 0.004,
  dctOn: false,
  dctStep: 12,
  audioStrongOn: false,
  regradeOn: false,
  recropOn: false,
  sharpness: false,
  colorFix: false,
  aiDenoise: false,
  spoof: false,
  temporalSubOn: false,
  temporalSub: 0.6,
  nativeTemporalOn: true,
  fftOn: false,
  fftPhase: 0.5,
  dwtDetailOn: false,
  dwtDetail: 0.8,
  warpOn: false,
  warp: 0.005,
  shearOn: false,
  shear: 0.01,
  jitterOn: false,
  jitter: 0.005,
  faceOn: false,
  face: 0.04,
  lpcOn: false,
  lpc: 0.5,
  neuralOn: false,
  neural: 0.05,
  qualityProtectOn: false,
  psnrTarget: 38,
  ssimTarget: 0.94,
  purifyOn: false,
  purifyStrength: 0.15,
  purifyDetail: 1.0,
  /** σ=0：按分辨率自动（引擎负责换算 + 后置锐化）。 */
  purifyDetailSigma: 0,
  /** 默认＝画质优先：宽带回注 + 字幕增强（字幕可辨，清除率打折）。 */
  purifyDetailWide: true,
  /** 已知来源水印家族：""=不确定，luma=亮度型，chroma=色度型（限制清晰度档位）。 */
  knownScheme: "" as "" | "luma" | "chroma",
  purifyTemporal: 0.0,
  purifyMaxEdge: 512,
  purifyBatch: 8,
  embeddingOn: false,
  embeddingAttack: "chroma" as "auto" | "luma" | "chroma" | "both",
  embeddingStrength: 0.25,
  embeddingVariant: "v2" as "legacy" | "v2",
  embeddingAggressive: false,
  autoProfile: false,
};

/** 清洗设置唯一数据源：面板可写，批量清洗只读，单条/批量共用同一份参数。 */
export interface CleanPanelState {
  presetBase: PresetBase;
  presetModified: boolean;
  audioClean: boolean;
  echoDefeat: boolean;
  rotateOn: boolean;
  rotate: number;
  hashOn: boolean;
  hashEps: number;
  hashMode: "phash" | "dhash" | "joint";
  requantOn: boolean;
  requant: number;
  noiseOn: boolean;
  noise: number;
  dctOn: boolean;
  dctStep: number;
  audioStrongOn: boolean;
  regradeOn: boolean;
  recropOn: boolean;
  sharpness: boolean;
  colorFix: boolean;
  aiDenoise: boolean;
  spoof: boolean;
  temporalSubOn: boolean;
  temporalSub: number;
  nativeTemporalOn: boolean;
  fftOn: boolean;
  fftPhase: number;
  dwtDetailOn: boolean;
  dwtDetail: number;
  warpOn: boolean;
  warp: number;
  shearOn: boolean;
  shear: number;
  jitterOn: boolean;
  jitter: number;
  faceOn: boolean;
  face: number;
  lpcOn: boolean;
  lpc: number;
  neuralOn: boolean;
  neural: number;
  qualityProtectOn: boolean;
  psnrTarget: number;
  ssimTarget: number;
  purifyOn: boolean;
  purifyStrength: number;
  purifyDetail: number;
  purifyDetailSigma: number;
  purifyDetailWide: boolean;
  knownScheme: "" | "luma" | "chroma";
  purifyTemporal: number;
  purifyMaxEdge: number;
  purifyBatch: number;
  embeddingOn: boolean;
  embeddingAttack: "auto" | "luma" | "chroma" | "both";
  embeddingStrength: number;
  embeddingVariant: "legacy" | "v2";
  embeddingAggressive: boolean;
  autoProfile: boolean;
  codec: Codec;
  lossless: boolean;
  resolution: string;
  templateId: string;
  settingsExportDir: string;
  settingsNaming: string;
  settingsFilterScale: string;
  setAudioClean: (value: boolean) => void;
  setEchoDefeat: (value: boolean) => void;
  setRotateOn: (value: boolean) => void;
  setRotate: (value: number) => void;
  setHashOn: (value: boolean) => void;
  setHashEps: (value: number) => void;
  setHashMode: (value: "phash" | "dhash" | "joint") => void;
  setRequantOn: (value: boolean) => void;
  setRequant: (value: number) => void;
  setNoiseOn: (value: boolean) => void;
  setNoise: (value: number) => void;
  setDctOn: (value: boolean) => void;
  setDctStep: (value: number) => void;
  setAudioStrongOn: (value: boolean) => void;
  setRegradeOn: (value: boolean) => void;
  setRecropOn: (value: boolean) => void;
  setSharpness: (value: boolean) => void;
  setColorFix: (value: boolean) => void;
  setAiDenoise: (value: boolean) => void;
  setSpoof: (value: boolean) => void;
  setTemporalSubOn: (value: boolean) => void;
  setTemporalSub: (value: number) => void;
  setNativeTemporalOn: (value: boolean) => void;
  setFftOn: (value: boolean) => void;
  setFftPhase: (value: number) => void;
  setDwtDetailOn: (value: boolean) => void;
  setDwtDetail: (value: number) => void;
  setWarpOn: (value: boolean) => void;
  setWarp: (value: number) => void;
  setShearOn: (value: boolean) => void;
  setShear: (value: number) => void;
  setJitterOn: (value: boolean) => void;
  setJitter: (value: number) => void;
  setFaceOn: (value: boolean) => void;
  setFace: (value: number) => void;
  setLpcOn: (value: boolean) => void;
  setLpc: (value: number) => void;
  setNeuralOn: (value: boolean) => void;
  setNeural: (value: number) => void;
  setQualityProtectOn: (value: boolean) => void;
  setPsnrTarget: (value: number) => void;
  setSsimTarget: (value: number) => void;
  setPurifyOn: (value: boolean) => void;
  setPurifyStrength: (value: number) => void;
  setPurifyDetail: (value: number) => void;
  setPurifyDetailSigma: (value: number) => void;
  setPurifyDetailWide: (value: boolean) => void;
  setKnownScheme: (value: "" | "luma" | "chroma") => void;
  setPurifyTemporal: (value: number) => void;
  setPurifyMaxEdge: (value: number) => void;
  setPurifyBatch: (value: number) => void;
  setEmbeddingOn: (value: boolean) => void;
  setEmbeddingAttack: (value: "auto" | "luma" | "chroma" | "both") => void;
  setEmbeddingStrength: (value: number) => void;
  setEmbeddingVariant: (value: "legacy" | "v2") => void;
  setEmbeddingAggressive: (value: boolean) => void;
  applyBlackBoxPreset: (preset: "balanced" | "aggressive" | "extreme") => void;
  setAutoProfile: (value: boolean) => void;
  setCodec: (value: Codec) => void;
  setLossless: (value: boolean) => void;
  setResolution: (value: string) => void;
  setTemplateId: (value: string) => void;
  setSettingsExportDir: (value: string) => void;
  setSettingsNaming: (value: string) => void;
  setSettingsFilterScale: (value: string) => void;
  applyTemplatePayload: (payload: TemplatePayload) => void;
  resetCleanDefaults: () => void;
  loadSettings: () => Promise<void>;
}

const near = (value: number, target: number) => Math.abs(value - target) < 1e-6;

/** 仅用于模板回填时判断它是否恰好等于某个快捷预设；运行期不靠数值反推。 */
export function detectPresetBase(state: CleanPanelState): PresetBase {
  if (
    state.purifyOn &&
    near(state.purifyStrength, 0.1) &&
    near(state.purifyDetail, 1.0) &&
    near(state.purifyDetailSigma, 0) &&
    !state.purifyDetailWide &&
    near(state.purifyTemporal, 0) &&
    state.purifyMaxEdge === 192 &&
    state.purifyBatch === 8 &&
    !state.embeddingOn &&
    state.autoProfile
  ) {
    return "extreme";
  }
  if (
    state.purifyOn &&
    near(state.purifyStrength, 0.15) &&
    near(state.purifyDetail, 1.0) &&
    near(state.purifyDetailSigma, 0) &&
    state.purifyDetailWide &&
    near(state.purifyTemporal, 0) &&
    state.purifyMaxEdge === 512 &&
    state.purifyBatch === 8 &&
    !state.embeddingOn &&
    state.autoProfile
  ) {
    return "balanced";
  }
  if (
    state.purifyOn &&
    near(state.purifyStrength, 0.35) &&
    near(state.purifyDetail, 1.0) &&
    near(state.purifyDetailSigma, 0) &&
    !state.purifyDetailWide &&
    near(state.purifyTemporal, 0.5) &&
    state.purifyMaxEdge === 256 &&
    state.purifyBatch === 8 &&
    state.embeddingOn &&
    state.embeddingAttack === "both" &&
    near(state.embeddingStrength, 0.6) &&
    state.embeddingVariant === "v2" &&
    state.embeddingAggressive &&
    !state.autoProfile
  ) {
    return "aggressive";
  }
  return "custom";
}

const presetPatch = (state: CleanPanelState) =>
  state.presetBase === "custom" ? {} : { presetModified: true };

export const useCleanPanel = create<CleanPanelState>((set, get) => ({
  ...CLEAN_DEFAULTS,
  codec: "H.264",
  lossless: false,
  resolution: "保持原始分辨率",
  templateId: "manual",
  settingsExportDir: "",
  settingsNaming: "原文件名 + 时间戳",
  settingsFilterScale: "720",
  setAudioClean: (audioClean) => set({ audioClean }),
  setEchoDefeat: (echoDefeat) => set({ echoDefeat }),
  setRotateOn: (rotateOn) => set({ rotateOn }),
  setRotate: (rotate) => set({ rotate }),
  setHashOn: (hashOn) => set({ hashOn }),
  setHashEps: (hashEps) => set({ hashEps }),
  setHashMode: (hashMode) => set({ hashMode }),
  setRequantOn: (requantOn) => set({ requantOn }),
  setRequant: (requant) => set({ requant }),
  setNoiseOn: (noiseOn) => set({ noiseOn }),
  setNoise: (noise) => set({ noise }),
  setDctOn: (dctOn) => set({ dctOn }),
  setDctStep: (dctStep) => set({ dctStep }),
  setAudioStrongOn: (audioStrongOn) => set({ audioStrongOn }),
  setRegradeOn: (regradeOn) => set({ regradeOn }),
  setRecropOn: (recropOn) => set({ recropOn }),
  setSharpness: (sharpness) => set({ sharpness }),
  setColorFix: (colorFix) => set({ colorFix }),
  setAiDenoise: (aiDenoise) => set({ aiDenoise }),
  setSpoof: (spoof) => set({ spoof }),
  setTemporalSubOn: (temporalSubOn) => set({ temporalSubOn }),
  setTemporalSub: (temporalSub) => set({ temporalSub }),
  setNativeTemporalOn: (nativeTemporalOn) => set({ nativeTemporalOn }),
  setFftOn: (fftOn) => set({ fftOn }),
  setFftPhase: (fftPhase) => set({ fftPhase }),
  setDwtDetailOn: (dwtDetailOn) => set({ dwtDetailOn }),
  setDwtDetail: (dwtDetail) => set({ dwtDetail }),
  setWarpOn: (warpOn) => set({ warpOn }),
  setWarp: (warp) => set({ warp }),
  setShearOn: (shearOn) => set({ shearOn }),
  setShear: (shear) => set({ shear }),
  setJitterOn: (jitterOn) => set({ jitterOn }),
  setJitter: (jitter) => set({ jitter }),
  setFaceOn: (faceOn) => set({ faceOn }),
  setFace: (face) => set({ face }),
  setLpcOn: (lpcOn) => set({ lpcOn }),
  setLpc: (lpc) => set({ lpc }),
  setNeuralOn: (neuralOn) => set({ neuralOn }),
  setNeural: (neural) => set({ neural }),
  setQualityProtectOn: (qualityProtectOn) => set({ qualityProtectOn }),
  setPsnrTarget: (psnrTarget) => set({ psnrTarget }),
  setSsimTarget: (ssimTarget) => set({ ssimTarget }),
  setPurifyOn: (purifyOn) =>
    set((state) => ({ purifyOn, ...presetPatch(state) })),
  setPurifyStrength: (purifyStrength) =>
    set((state) => ({ purifyStrength, ...presetPatch(state) })),
  setPurifyDetail: (purifyDetail) =>
    set((state) => ({ purifyDetail, ...presetPatch(state) })),
  setPurifyDetailSigma: (purifyDetailSigma) =>
    set((state) => ({ purifyDetailSigma, ...presetPatch(state) })),
  setPurifyDetailWide: (purifyDetailWide) =>
    set((state) => ({ purifyDetailWide, ...presetPatch(state) })),
  setKnownScheme: (knownScheme) =>
    set((state) => ({ knownScheme, ...presetPatch(state) })),
  setPurifyTemporal: (purifyTemporal) =>
    set((state) => ({ purifyTemporal, ...presetPatch(state) })),
  setPurifyMaxEdge: (purifyMaxEdge) =>
    set((state) => ({ purifyMaxEdge, ...presetPatch(state) })),
  setPurifyBatch: (purifyBatch) =>
    set((state) => ({ purifyBatch, ...presetPatch(state) })),
  setEmbeddingOn: (embeddingOn) =>
    set((state) => ({ embeddingOn, ...presetPatch(state) })),
  setEmbeddingAttack: (embeddingAttack) =>
    set((state) => ({ embeddingAttack, ...presetPatch(state) })),
  setEmbeddingStrength: (embeddingStrength) =>
    set((state) => ({ embeddingStrength, ...presetPatch(state) })),
  setEmbeddingVariant: (embeddingVariant) =>
    set((state) => ({ embeddingVariant, ...presetPatch(state) })),
  setEmbeddingAggressive: (embeddingAggressive) =>
    set((state) => ({ embeddingAggressive, ...presetPatch(state) })),
  applyBlackBoxPreset: (preset) =>
    set(
      preset === "extreme"
        ? {
            presetBase: preset,
            presetModified: false,
            purifyOn: true,
            purifyStrength: 0.10,
            purifyDetail: 1.0,
            purifyDetailSigma: 0,
            purifyDetailWide: false,
            purifyTemporal: 0.0,
            purifyMaxEdge: 192,
            purifyBatch: 8,
            embeddingOn: false,
            embeddingAttack: "both",
            embeddingStrength: 0.25,
            embeddingVariant: "v2",
            embeddingAggressive: false,
            autoProfile: true,
          }
        : preset === "aggressive"
        ? {
            presetBase: preset,
            presetModified: false,
            purifyOn: true,
            purifyStrength: 0.35,
            purifyDetail: 1.0,
            purifyDetailSigma: 0,
            purifyDetailWide: false,
            purifyTemporal: 0.5,
            purifyMaxEdge: 256,
            purifyBatch: 8,
            embeddingOn: true,
            embeddingAttack: "both",
            embeddingStrength: 0.6,
            embeddingVariant: "v2",
            embeddingAggressive: true,
            autoProfile: false,
          }
        : {
            presetBase: preset,
            presetModified: false,
            purifyOn: true,
            purifyStrength: 0.15,
            purifyDetail: 1.0,
            purifyDetailSigma: 0,
            // B 档：宽带回注（1080p≈6.5）换字幕可读，水印会部分回流。
            purifyDetailWide: true,
            purifyTemporal: 0.0,
            // 画质优先：长边 512（1080p 下 3.75× 放大，画面明显更实）；
            // 代价是部分公开方案水印回流，追求清除率请选「强力/极速清除」。
            purifyMaxEdge: 512,
            purifyBatch: 8,
            embeddingOn: false,
            embeddingAttack: "both",
            embeddingStrength: 0.25,
            embeddingVariant: "v2",
            embeddingAggressive: false,
            autoProfile: true,
          },
    ),
  setAutoProfile: (autoProfile) =>
    set((state) => ({ autoProfile, ...presetPatch(state) })),
  setCodec: (codec) => set({ codec }),
  setLossless: (lossless) => set({ lossless }),
  setResolution: (resolution) => set({ resolution }),
  setTemplateId: (templateId) => set({ templateId }),
  setSettingsExportDir: (settingsExportDir) => set({ settingsExportDir }),
  setSettingsNaming: (settingsNaming) => set({ settingsNaming }),
  setSettingsFilterScale: (settingsFilterScale) => set({ settingsFilterScale }),
  applyTemplatePayload: (payload) => {
    set({
      audioClean: payload.audioRemix,
      echoDefeat: payload.echoDefeat,
      rotateOn: (payload.rotate ?? 0) > 0,
      rotate:
        (payload.rotate ?? 0) > 0 ? (payload.rotate ?? CLEAN_DEFAULTS.rotate) : CLEAN_DEFAULTS.rotate,
      hashOn: payload.hashAttack ?? false,
      hashEps: payload.hashAttack
        ? (payload.hashEpsilon ?? CLEAN_DEFAULTS.hashEps)
        : CLEAN_DEFAULTS.hashEps,
      hashMode: payload.hashAttack
        ? ((payload.hashMode as "phash" | "dhash" | "joint") ?? CLEAN_DEFAULTS.hashMode)
        : CLEAN_DEFAULTS.hashMode,
      requantOn: (payload.requant ?? 0) > 0,
      requant:
        (payload.requant ?? 0) > 0 ? (payload.requant ?? CLEAN_DEFAULTS.requant) : CLEAN_DEFAULTS.requant,
      noiseOn: (payload.noise ?? 0) > 0,
      noise: (payload.noise ?? 0) > 0 ? (payload.noise ?? CLEAN_DEFAULTS.noise) : CLEAN_DEFAULTS.noise,
      dctOn: (payload.dctStep ?? 0) > 0 || (payload.antiReembed ?? false),
      dctStep:
        (payload.dctStep ?? 0) > 0 ? (payload.dctStep ?? CLEAN_DEFAULTS.dctStep) : CLEAN_DEFAULTS.dctStep,
      audioStrongOn: payload.audioStrong ?? false,
      regradeOn: payload.regradeOn,
      recropOn: payload.recropOn,
      sharpness: payload.sharpness,
      colorFix: payload.colorRestore,
      aiDenoise: payload.denoise,
      spoof: payload.spoof,
      temporalSubOn: (payload.temporalSub ?? 0) > 0,
      temporalSub:
        (payload.temporalSub ?? 0) > 0
          ? (payload.temporalSub ?? CLEAN_DEFAULTS.temporalSub)
          : CLEAN_DEFAULTS.temporalSub,
      nativeTemporalOn: payload.nativeTemporal ?? CLEAN_DEFAULTS.nativeTemporalOn,
      fftOn: (payload.fftPhase ?? 0) > 0,
      fftPhase:
        (payload.fftPhase ?? 0) > 0 ? (payload.fftPhase ?? CLEAN_DEFAULTS.fftPhase) : CLEAN_DEFAULTS.fftPhase,
      dwtDetailOn: (payload.dwtDetail ?? 0) > 0,
      dwtDetail:
        (payload.dwtDetail ?? 0) > 0 ? (payload.dwtDetail ?? CLEAN_DEFAULTS.dwtDetail) : CLEAN_DEFAULTS.dwtDetail,
      warpOn: (payload.warp ?? 0) > 0,
      warp: (payload.warp ?? 0) > 0 ? (payload.warp ?? CLEAN_DEFAULTS.warp) : CLEAN_DEFAULTS.warp,
      shearOn: (payload.perspective ?? 0) > 0,
      shear:
        (payload.perspective ?? 0) > 0 ? (payload.perspective ?? CLEAN_DEFAULTS.shear) : CLEAN_DEFAULTS.shear,
      jitterOn: (payload.jitter ?? 0) > 0,
      jitter:
        (payload.jitter ?? 0) > 0 ? (payload.jitter ?? CLEAN_DEFAULTS.jitter) : CLEAN_DEFAULTS.jitter,
      faceOn: (payload.facePerturb ?? 0) > 0,
      face:
        (payload.facePerturb ?? 0) > 0
          ? (payload.facePerturb ?? CLEAN_DEFAULTS.face)
          : CLEAN_DEFAULTS.face,
      lpcOn: (payload.lpcAttack ?? 0) > 0,
      lpc:
        (payload.lpcAttack ?? 0) > 0
          ? (payload.lpcAttack ?? CLEAN_DEFAULTS.lpc)
          : CLEAN_DEFAULTS.lpc,
      neuralOn: (payload.copyAttack ?? 0) > 0,
      neural:
        (payload.copyAttack ?? 0) > 0
          ? (payload.copyAttack ?? CLEAN_DEFAULTS.neural)
          : CLEAN_DEFAULTS.neural,
      qualityProtectOn: payload.qualityProtect ?? false,
      psnrTarget: payload.psnrTarget ?? 38,
      ssimTarget: payload.ssimTarget ?? 0.94,
      purifyOn: (payload.purifyStrength ?? 0) > 0,
      purifyStrength:
        (payload.purifyStrength ?? 0) > 0
          ? (payload.purifyStrength ?? CLEAN_DEFAULTS.purifyStrength)
          : CLEAN_DEFAULTS.purifyStrength,
      purifyDetail: payload.purifyDetail ?? CLEAN_DEFAULTS.purifyDetail,
      purifyDetailSigma:
        payload.purifyDetailSigma ?? CLEAN_DEFAULTS.purifyDetailSigma,
      purifyDetailWide:
        payload.purifyDetailWide ?? CLEAN_DEFAULTS.purifyDetailWide,
      knownScheme:
        (payload.knownScheme as "" | "luma" | "chroma") ?? CLEAN_DEFAULTS.knownScheme,
      purifyTemporal: payload.purifyTemporal ?? CLEAN_DEFAULTS.purifyTemporal,
      purifyMaxEdge: payload.purifyMaxEdge ?? CLEAN_DEFAULTS.purifyMaxEdge,
      purifyBatch: payload.purifyBatch ?? CLEAN_DEFAULTS.purifyBatch,
      embeddingOn:
        (payload.embeddingStrength ?? 0) > 0 && (payload.embeddingAttack ?? "") !== "",
      embeddingAttack:
        ((payload.embeddingAttack as "auto" | "luma" | "chroma" | "both" | "") || "") ||
        CLEAN_DEFAULTS.embeddingAttack,
      embeddingStrength:
        (payload.embeddingStrength ?? 0) > 0
          ? (payload.embeddingStrength ?? CLEAN_DEFAULTS.embeddingStrength)
          : CLEAN_DEFAULTS.embeddingStrength,
      embeddingVariant: payload.embeddingVariant ?? CLEAN_DEFAULTS.embeddingVariant,
      embeddingAggressive:
        payload.embeddingAggressive ?? CLEAN_DEFAULTS.embeddingAggressive,
      autoProfile: payload.autoProfile ?? false,
      codec: payload.codec,
      lossless: payload.lossless,
      resolution: payload.resolution,
    });
    set({
      presetBase: detectPresetBase(get()),
      presetModified: false,
    });
  },
  resetCleanDefaults: () => set(CLEAN_DEFAULTS),
  loadSettings: async () => {
    try {
      const settings = await getSettings();
      set({
        settingsExportDir: (settings.export_dir as string) ?? "",
        settingsNaming: (settings.naming as string) ?? "原文件名 + 时间戳",
        settingsFilterScale: (settings.filter_scale as string) ?? "720",
      });
    } catch {
      // 后端未就绪时保持默认设置。
    }
  },
}));
