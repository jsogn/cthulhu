import { create } from "zustand";

import { getSettings } from "@/lib/backend";
import type { Codec, TemplatePayload } from "@/lib/templates";

export type OutputMode = "reencode" | "remux";

/** 清洗参数默认值：初始状态与 resetCleanDefaults 共用，避免两处手写漂移。 */
const CLEAN_DEFAULTS = {
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
  fftMag: 0.1,
  dwtDetailOn: false,
  dwtDetail: 0.8,
  warpOn: false,
  warp: 0.005,
  shearOn: false,
  shear: 0.01,
  jitterOn: false,
  jitter: 0.005,
  nonintOn: false,
  nonintRatio: 0.01,
  flowOn: false,
  flow: 1.5,
  textureOn: false,
  texture: 0.04,
  multiscaleOn: false,
  multiscale: 0.02,
  faceOn: false,
  face: 0.04,
  temporalBlurOn: false,
  temporalBlur: 0.25,
  lpcOn: false,
  lpc: 0.5,
  neuralOn: false,
  neural: 0.05,
  qualityProtectOn: false,
  psnrTarget: 38,
  ssimTarget: 0.94,
};

/** 清洗设置唯一数据源：面板可写，批量清洗只读，单条/批量共用同一份参数。 */
export interface CleanPanelState {
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
  audioStrongOn: boolean;
  outputMode: OutputMode;
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
  fftMag: number;
  dwtDetailOn: boolean;
  dwtDetail: number;
  warpOn: boolean;
  warp: number;
  shearOn: boolean;
  shear: number;
  jitterOn: boolean;
  jitter: number;
  nonintOn: boolean;
  nonintRatio: number;
  flowOn: boolean;
  flow: number;
  textureOn: boolean;
  texture: number;
  multiscaleOn: boolean;
  multiscale: number;
  faceOn: boolean;
  face: number;
  temporalBlurOn: boolean;
  temporalBlur: number;
  lpcOn: boolean;
  lpc: number;
  neuralOn: boolean;
  neural: number;
  qualityProtectOn: boolean;
  psnrTarget: number;
  ssimTarget: number;
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
  setAudioStrongOn: (value: boolean) => void;
  setOutputMode: (value: OutputMode) => void;
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
  setFftMag: (value: number) => void;
  setDwtDetailOn: (value: boolean) => void;
  setDwtDetail: (value: number) => void;
  setWarpOn: (value: boolean) => void;
  setWarp: (value: number) => void;
  setShearOn: (value: boolean) => void;
  setShear: (value: number) => void;
  setJitterOn: (value: boolean) => void;
  setJitter: (value: number) => void;
  setNonintOn: (value: boolean) => void;
  setNonintRatio: (value: number) => void;
  setFlowOn: (value: boolean) => void;
  setFlow: (value: number) => void;
  setTextureOn: (value: boolean) => void;
  setTexture: (value: number) => void;
  setMultiscaleOn: (value: boolean) => void;
  setMultiscale: (value: number) => void;
  setFaceOn: (value: boolean) => void;
  setFace: (value: number) => void;
  setTemporalBlurOn: (value: boolean) => void;
  setTemporalBlur: (value: number) => void;
  setLpcOn: (value: boolean) => void;
  setLpc: (value: number) => void;
  setNeuralOn: (value: boolean) => void;
  setNeural: (value: number) => void;
  setQualityProtectOn: (value: boolean) => void;
  setPsnrTarget: (value: number) => void;
  setSsimTarget: (value: number) => void;
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

export const useCleanPanel = create<CleanPanelState>((set) => ({
  ...CLEAN_DEFAULTS,
  outputMode: "reencode",
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
  setAudioStrongOn: (audioStrongOn) => set({ audioStrongOn }),
  setOutputMode: (outputMode) => set({ outputMode }),
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
  setFftMag: (fftMag) => set({ fftMag }),
  setDwtDetailOn: (dwtDetailOn) => set({ dwtDetailOn }),
  setDwtDetail: (dwtDetail) => set({ dwtDetail }),
  setWarpOn: (warpOn) => set({ warpOn }),
  setWarp: (warp) => set({ warp }),
  setShearOn: (shearOn) => set({ shearOn }),
  setShear: (shear) => set({ shear }),
  setJitterOn: (jitterOn) => set({ jitterOn }),
  setJitter: (jitter) => set({ jitter }),
  setNonintOn: (nonintOn) => set({ nonintOn }),
  setNonintRatio: (nonintRatio) => set({ nonintRatio }),
  setFlowOn: (flowOn) => set({ flowOn }),
  setFlow: (flow) => set({ flow }),
  setTextureOn: (textureOn) => set({ textureOn }),
  setTexture: (texture) => set({ texture }),
  setMultiscaleOn: (multiscaleOn) => set({ multiscaleOn }),
  setMultiscale: (multiscale) => set({ multiscale }),
  setFaceOn: (faceOn) => set({ faceOn }),
  setFace: (face) => set({ face }),
  setTemporalBlurOn: (temporalBlurOn) => set({ temporalBlurOn }),
  setTemporalBlur: (temporalBlur) => set({ temporalBlur }),
  setLpcOn: (lpcOn) => set({ lpcOn }),
  setLpc: (lpc) => set({ lpc }),
  setNeuralOn: (neuralOn) => set({ neuralOn }),
  setNeural: (neural) => set({ neural }),
  setQualityProtectOn: (qualityProtectOn) => set({ qualityProtectOn }),
  setPsnrTarget: (psnrTarget) => set({ psnrTarget }),
  setSsimTarget: (ssimTarget) => set({ ssimTarget }),
  setCodec: (codec) => set({ codec }),
  setLossless: (lossless) => set({ lossless }),
  setResolution: (resolution) => set({ resolution }),
  setTemplateId: (templateId) => set({ templateId }),
  setSettingsExportDir: (settingsExportDir) => set({ settingsExportDir }),
  setSettingsNaming: (settingsNaming) => set({ settingsNaming }),
  setSettingsFilterScale: (settingsFilterScale) => set({ settingsFilterScale }),
  applyTemplatePayload: (payload) =>
    set({
      audioClean: payload.audioRemix,
      echoDefeat: payload.echoDefeat,
      rotateOn: (payload.rotate ?? 0) > 0,
      rotate: (payload.rotate ?? 0) > 0 ? (payload.rotate ?? 0.2) : 0.2,
      hashOn: payload.hashAttack ?? false,
      hashEps: payload.hashAttack ? (payload.hashEpsilon ?? 0.045) : 0.045,
      hashMode: (payload.hashMode as "phash" | "dhash" | "joint") ?? "joint",
      requantOn: (payload.requant ?? 0) > 0,
      requant: (payload.requant ?? 0) > 0 ? (payload.requant ?? 64) : 64,
      noiseOn: (payload.noise ?? 0) > 0,
      noise: (payload.noise ?? 0) > 0 ? (payload.noise ?? 0.004) : 0.004,
      dctOn: (payload.dctStep ?? 0) > 0 || (payload.antiReembed ?? false),
      audioStrongOn: payload.audioStrong ?? false,
      regradeOn: payload.regradeOn,
      recropOn: payload.recropOn,
      sharpness: payload.sharpness,
      colorFix: payload.colorRestore,
      aiDenoise: payload.denoise,
      spoof: payload.spoof,
      temporalSubOn: (payload.temporalSub ?? 0) > 0,
      temporalSub: (payload.temporalSub ?? 0) > 0 ? (payload.temporalSub ?? 0.8) : 0.8,
      nativeTemporalOn: payload.nativeTemporal ?? false,
      fftOn: (payload.fftPhase ?? 0) > 0 || (payload.fftMag ?? 0) > 0,
      fftPhase: (payload.fftPhase ?? 0) > 0 ? (payload.fftPhase ?? 0.5) : 0.5,
      fftMag: (payload.fftMag ?? 0) > 0 ? (payload.fftMag ?? 0.5) : 0.5,
      dwtDetailOn: (payload.dwtDetail ?? 0) > 0,
      dwtDetail: (payload.dwtDetail ?? 0) > 0 ? (payload.dwtDetail ?? 0.8) : 0.8,
      warpOn: (payload.warp ?? 0) > 0,
      warp: (payload.warp ?? 0) > 0 ? (payload.warp ?? 0.005) : 0.005,
      shearOn: (payload.perspective ?? 0) > 0,
      shear: (payload.perspective ?? 0) > 0 ? (payload.perspective ?? 0.01) : 0.01,
      jitterOn: (payload.jitter ?? 0) > 0,
      jitter: (payload.jitter ?? 0) > 0 ? (payload.jitter ?? 0.005) : 0.005,
      nonintOn: (payload.nonintRatio ?? 0) > 0,
      nonintRatio: (payload.nonintRatio ?? 0) > 0 ? (payload.nonintRatio ?? 0.01) : 0.01,
      flowOn: (payload.flowDisturb ?? 0) > 0,
      flow: (payload.flowDisturb ?? 0) > 0 ? (payload.flowDisturb ?? 2) : 2,
      textureOn: (payload.textureInject ?? 0) > 0 || (payload.complexityTrap ?? 0) > 0,
      texture:
        (payload.textureInject ?? 0) > 0
          ? (payload.textureInject ?? 0.04)
          : (payload.complexityTrap ?? 0) > 0
            ? Math.min(0.05, (payload.complexityTrap ?? 0.06) / 2.5)
            : 0.04,
      multiscaleOn: (payload.multiscale ?? 0) > 0,
      multiscale: (payload.multiscale ?? 0) > 0 ? (payload.multiscale ?? 0.02) : 0.02,
      faceOn: (payload.facePerturb ?? 0) > 0,
      face: (payload.facePerturb ?? 0) > 0 ? (payload.facePerturb ?? 0.04) : 0.04,
      temporalBlurOn: (payload.temporalBlur ?? 0) > 0,
      temporalBlur: (payload.temporalBlur ?? 0) > 0 ? (payload.temporalBlur ?? 0.25) : 0.25,
      lpcOn: (payload.lpcAttack ?? 0) > 0,
      lpc: (payload.lpcAttack ?? 0) > 0 ? (payload.lpcAttack ?? 0.85) : 0.85,
      neuralOn: (payload.copyAttack ?? 0) > 0,
      neural: (payload.copyAttack ?? 0) > 0 ? (payload.copyAttack ?? 0.05) : 0.05,
      qualityProtectOn: payload.qualityProtect ?? false,
      psnrTarget: payload.psnrTarget ?? 38,
      ssimTarget: payload.ssimTarget ?? 0.94,
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
        settingsFilterScale: (settings.filter_scale as string) ?? "720",
      });
    } catch {
      // 后端未就绪时保持默认设置。
    }
  },
}));
