import type { DesensitizeOptions } from "@/lib/backend";
import type { CleanPanelState } from "@/stores/cleanPanel";

// 基础微扰与裁剪为固定保守值：微扰实测收益接近噪声，裁剪幅度在 3%~4% 附近
// 到达收益拐点且非单调，因此不做连续调节，只保留「重新构图」开关。
export const FIXED_PERTURB = 0.15;
export const FIXED_GAMMA = 0.03 + 0.2 * FIXED_PERTURB;
export const FIXED_BRIGHTNESS = 0.02 + 0.06 * FIXED_PERTURB;
export const FIXED_CROP = 0.03;

// 指纹对抗档：静态几何去同步（低频微旋转）+ 签名域扰动 + 静态频域/色度原语。
// 硬约束：不含可见运动（逐帧抖动/透视/局部扭曲/抽帧复制）与时间域攻击
// （逐镜头变速/切点删帧/音频变速），保证观感与音画同步。
export type AntiPreset = {
  rotate: number;
  epsilon?: number;
  median?: number;
  noise?: number;
  requant?: number;
  dctStep?: number;
  dropEvery?: number;
  jitter?: number;
  perspective?: number;
  warp?: number;
  mirror?: boolean;
  chromaLevels?: number;
  subtractBeta?: number;
  transcodeChain?: boolean;
  jointAttack?: boolean;
  saliency?: number;
  detailProtect?: number;
  shotRetime?: boolean;
  cutJitter?: number;
  audioStrong?: boolean;
};

export const ANTI_PRESETS: Record<string, AntiPreset | undefined> = {
  关闭: undefined,
  轻度: { rotate: 0.15 },
  标准: {
    rotate: 0.2,
    requant: 96,
    noise: 0.003,
    audioStrong: true,
  },
  强力: {
    rotate: 0.25,
    epsilon: 0.045,
    requant: 64,
    noise: 0.004,
    audioStrong: true,
  },
  全兵器: {
    rotate: 0.3,
    epsilon: 0.05,
    dctStep: 12,
    requant: 32,
    chromaLevels: 32,
    noise: 0.008,
    subtractBeta: 1.2,
    transcodeChain: true,
    jointAttack: true,
    audioStrong: true,
  },
};

/** 从清洗设置单源组装后端清洗参数（每次调用生成独立随机种子）。 */
export function makeCleanOptions(state: CleanPanelState): DesensitizeOptions {
  // 只换壳模式不经过任何对抗/处理管线，记录与展示都不应包含无效参数。
  if (state.outputMode === "remux") {
    return {
      reorder: false,
      speed: 1.0,
      recrop: 0,
      perturb: 0,
      regrade: false,
      output_mode: "remux",
      audio_remix: false,
      echo_defeat: false,
      sharpness: false,
      color_restore: false,
      denoise: false,
      anti_reembed: false,
      seed: Math.floor(Math.random() * 1_000_000),
    };
  }
  const anti = ANTI_PRESETS[state.antiLevel];
  return {
    reorder: false,
    speed: 1.0,
    recrop: state.recropOn ? FIXED_CROP : 0,
    perturb: FIXED_PERTURB,
    regrade: state.regradeOn,
    output_mode: state.outputMode,
    audio_remix: state.audioClean,
    echo_defeat: state.echoDefeat,
    skip_vmaf: true,
    filter_scale: Number(state.settingsFilterScale) || 0,
    sharpness: state.sharpness,
    color_restore: state.colorFix,
    denoise: state.aiDenoise,
    anti_reembed: state.antiReembed,
    detail_protect: state.detailProtectOn ? 0.5 : 0,
    seed: Math.floor(Math.random() * 1_000_000),
    codec: state.codec === "H.265" ? "libx265" : "libx264",
    lossless: state.lossless,
    spoof: state.spoof,
    ...(state.resolution !== "保持原始分辨率" ? { resolution: state.resolution } : {}),
    ...(anti
      ? {
          rotate: anti.rotate,
          phash_attack: !anti.jointAttack && anti.epsilon != null,
          phash_epsilon: anti.epsilon,
          ...(anti.median ? { median: anti.median } : {}),
          ...(anti.noise ? { noise: anti.noise } : {}),
          ...(anti.requant ? { requant: anti.requant } : {}),
          ...(anti.dctStep ? { dct_step: anti.dctStep } : {}),
          ...(anti.dropEvery ? { drop_every: anti.dropEvery } : {}),
          ...(anti.jitter ? { jitter: anti.jitter } : {}),
          ...(anti.perspective ? { perspective: anti.perspective } : {}),
          ...(anti.warp ? { warp: anti.warp } : {}),
          ...(anti.mirror ? { mirror: true } : {}),
          ...(anti.chromaLevels ? { chroma_levels: anti.chromaLevels } : {}),
          ...(anti.subtractBeta ? { subtract_beta: anti.subtractBeta } : {}),
          ...(anti.transcodeChain ? { transcode_chain: true } : {}),
          ...(anti.jointAttack ? { multi_hash_attack: true } : {}),
          ...(anti.saliency ? { saliency: anti.saliency } : {}),
          ...(anti.detailProtect ? { detail_protect: anti.detailProtect } : {}),
          ...(anti.shotRetime ? { shot_retime: true } : {}),
          ...(anti.cutJitter ? { cut_jitter: anti.cutJitter } : {}),
          ...(anti.audioStrong ? { audio_strong: true } : {}),
        }
      : {}),
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
