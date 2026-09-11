import type { DesensitizeOptions } from "@/lib/backend";
import type { CleanPanelState } from "@/stores/cleanPanel";

// 基础微扰与裁剪为固定保守值：微扰实测收益接近噪声，裁剪幅度在 3%~4% 附近
// 到达收益拐点且非单调，因此不做连续调节，只保留「重新构图」开关。
export const FIXED_PERTURB = 0.15;
export const FIXED_GAMMA = 0.03 + 0.2 * FIXED_PERTURB;
export const FIXED_BRIGHTNESS = 0.02 + 0.06 * FIXED_PERTURB;
export const FIXED_CROP = 0.03;

/** 从清洗设置单源组装后端清洗参数（每次调用生成独立随机种子）。 */
export function makeCleanOptions(state: CleanPanelState): DesensitizeOptions {
  return {
    reorder: false,
    speed: 1.0,
    recrop: state.recropOn ? FIXED_CROP : 0,
    perturb: FIXED_PERTURB,
    regrade: state.regradeOn,
    ...(state.regradeOn ? { hsv_jitter: 6, chroma_levels: 32 } : {}),
    audio_remix: state.audioClean,
    echo_defeat: state.echoDefeat,
    skip_vmaf: true,
    filter_scale: Number(state.settingsFilterScale) || 0,
    sharpness: state.sharpness,
    color_restore: state.colorFix,
    denoise: state.aiDenoise,
    ...(state.rotateOn ? { rotate: state.rotate } : {}),
    ...(state.hashOn
      ? state.hashMode === "phash"
        ? { phash_attack: true, phash_epsilon: state.hashEps }
        : state.hashMode === "dhash"
          ? { dhash_attack: true, phash_epsilon: state.hashEps }
          : { multi_hash_attack: true, phash_epsilon: state.hashEps }
      : {}),
    ...(state.requantOn ? { requant: state.requant } : {}),
    ...(state.noiseOn ? { noise: state.noise } : {}),
    ...(state.dctOn ? { dct_step: state.dctStep } : {}),
    ...(state.audioStrongOn ? { audio_strong: true } : {}),
    ...(state.temporalSubOn ? { temporal_sub: state.temporalSub } : {}),
    ...(state.temporalSubOn && state.nativeTemporalOn ? { native_temporal: true } : {}),
    ...(state.fftOn ? { fft_phase: state.fftPhase } : {}),
    ...(state.dwtDetailOn ? { dwt_detail: state.dwtDetail } : {}),
    ...(state.warpOn ? { warp: state.warp } : {}),
    ...(state.shearOn ? { perspective: state.shear } : {}),
    ...(state.jitterOn ? { jitter: state.jitter } : {}),
    ...(state.flowOn ? { flow_disturb: state.flow } : {}),
    ...(state.textureOn
      ? { texture_inject: state.texture, complexity_trap: state.texture * 2.5 }
      : {}),
    ...(state.multiscaleOn ? { multiscale: state.multiscale } : {}),
    ...(state.faceOn ? { face_perturb: state.face } : {}),
    ...(state.temporalBlurOn ? { temporal_blur: state.temporalBlur } : {}),
    ...(state.lpcOn ? { lpc_attack: state.lpc } : {}),
    ...(state.neuralOn ? { copy_attack: state.neural } : {}),
    ...(state.purifyOn
      ? {
          purify_strength: state.purifyStrength,
          purify_detail: state.purifyDetail,
          purify_detail_sigma: state.purifyDetailSigma,
          purify_detail_wide: state.purifyDetailWide,
          known_scheme: state.knownScheme,
          purify_temporal: state.purifyTemporal,
          purify_max_edge: state.purifyMaxEdge,
          purify_batch: state.purifyBatch,
        }
      : {}),
    ...(state.embeddingOn
      ? {
          embedding_attack: state.embeddingAttack,
          embedding_strength: state.embeddingStrength,
          embedding_variant: state.embeddingVariant,
          embedding_aggressive: state.embeddingAggressive,
        }
      : {}),
    ...(state.autoProfile ? { auto_profile: true } : {}),
    ...(state.qualityProtectOn
      ? {
          quality_protect: true,
          psnr_target: state.psnrTarget,
          ssim_target: state.ssimTarget,
        }
      : {}),
    seed: Math.floor(Math.random() * 1_000_000),
    codec: state.codec === "H.265" ? "libx265" : "libx264",
    lossless: state.lossless,
    spoof: state.spoof,
    ...(state.resolution !== "保持原始分辨率" ? { resolution: state.resolution } : {}),
  };
}

/** 清洗产物输出路径：始终写入导出目录，绝不与源视频同目录。 */
export function cleanOutputPath(
  srcPath: string,
  naming: string,
  exportDir: string,
  ts: string,
  index?: number,
): string {
  const srcStem = srcPath.replace(/\.(mp4|mov|mkv|avi|flv|ts)$/i, "");
  const srcName = srcStem.split(/[\\/]/).pop() ?? srcStem;
  const suffix = index ? `_${index}` : "";
  const fileName =
    naming === "时间戳 + 原文件名"
      ? `${ts}_${srcName}_清洗${suffix}.mp4`
      : `${srcName}_清洗_${ts}${suffix}.mp4`;
  const baseDir = exportDir || "~/Documents/Cthulhu";
  return `${baseDir.replace(/\/+$/, "")}/${fileName}`;
}

/** 共谋平均输出路径：以第一个副本命名，标注共谋平均与时间戳。 */
export function collusionOutputPath(
  srcPath: string,
  exportDir: string,
  ts: string,
): string {
  const srcStem = srcPath.replace(/\.(mp4|mov|mkv|avi|flv|ts)$/i, "");
  const srcName = srcStem.split(/[\\/]/).pop() ?? srcStem;
  const baseDir = exportDir || "~/Documents/Cthulhu";
  return `${baseDir.replace(/\/+$/, "")}/${srcName}_共谋平均_${ts}.mp4`;
}
