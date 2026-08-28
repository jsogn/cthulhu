import type { OutputInfo } from "@/lib/backend";
import type { RiskLevel } from "@/stores/materials";

export const OUTPUT_KIND_LABEL: Record<OutputInfo["kind"], string> = {
  cleaned: "清洗",
  repaired: "修复",
};

export type OptionRow = { label: string; value: string };

// 产物参数可视化：把 snake_case 原始参数翻译成用户能看懂的中文摘要。
const ANTI_LEVEL_MATCHES: { level: string; checks: [string, unknown][] }[] = [
  { level: "轻度", checks: [["rotate", 0.15]] },
  {
    level: "标准",
    checks: [
      ["rotate", 0.2],
      ["requant", 96],
      ["noise", 0.003],
    ],
  },
  {
    level: "强力",
    checks: [
      ["rotate", 0.25],
      ["requant", 64],
      ["noise", 0.004],
    ],
  },
  {
    level: "全兵器",
    checks: [
      ["dct_step", 12],
      ["requant", 32],
      ["chroma_levels", 32],
      ["transcode_chain", true],
    ],
  },
];

const ANTI_TAG_LABELS: [string, string][] = [
  ["phash_attack", "感知哈希对抗"],
  ["median", "中值滤波"],
  ["noise", "噪声扰动"],
  ["requant", "像素重量化"],
  ["dct_step", "DCT 重量化"],
  ["chroma_levels", "色度量化"],
  ["drop_every", "抽帧"],
  ["jitter", "抖动"],
  ["perspective", "透视"],
  ["warp", "扭曲"],
  ["mirror", "镜像"],
  ["subtract_beta", "残差抑制"],
  ["transcode_chain", "二次转码"],
  ["multi_hash_attack", "多哈希对抗"],
  ["audio_strong", "音频强化"],
];

export function antiLevelOf(options: Record<string, unknown>): string | null {
  for (const preset of ANTI_LEVEL_MATCHES) {
    if (preset.checks.every(([key, value]) => options[key] === value)) return preset.level;
  }
  return null;
}

export function cleanOptionRows(options: Record<string, unknown>): OptionRow[] {
  const rows: OptionRow[] = [];
  const num = (value: unknown) => (typeof value === "number" ? value : 0);
  const enabled = (value: unknown) => value === true || num(value) > 0;

  rows.push({
    label: "清洗方式",
    value: options.output_mode === "remux" ? "重新封装 · 只换壳" : "重新编码",
  });

  const antiLevel = antiLevelOf(options);
  const activeTags = ANTI_TAG_LABELS.filter(([key]) => enabled(options[key])).map(
    ([, label]) => label,
  );
  if (antiLevel) {
    rows.push({ label: "指纹对抗", value: antiLevel });
  } else if (activeTags.length > 0) {
    rows.push({ label: "指纹对抗", value: "自定义组合" });
  }
  if (activeTags.length > 0) {
    rows.push({ label: "对抗原语", value: activeTags.join(" · ") });
  }

  if (enabled(options.audio_remix)) rows.push({ label: "音频重混", value: "开启" });
  if (enabled(options.echo_defeat)) rows.push({ label: "音频回声扰动", value: "开启" });
  if (enabled(options.anti_reembed)) rows.push({ label: "抗二次检测", value: "开启" });
  if (num(options.recrop) > 0) {
    rows.push({
      label: "重新构图",
      value: `裁左/下各 ${(num(options.recrop) * 100).toFixed(0)}%`,
    });
  }
  if (enabled(options.regrade)) rows.push({ label: "调光微扰", value: "开启" });
  if (num(options.detail_protect) > 0) rows.push({ label: "细节保护", value: "开启" });
  if (enabled(options.sharpness)) rows.push({ label: "锐度补偿", value: "开启" });
  if (enabled(options.color_restore)) rows.push({ label: "调色修复", value: "开启" });
  if (enabled(options.denoise)) rows.push({ label: "空间降噪", value: "开启" });
  if (enabled(options.spoof)) rows.push({ label: "反爬伪装", value: "开启" });
  if (num(options.saliency) > 0) rows.push({ label: "显著性伪装", value: "开启" });
  if (num(options.speed) !== 0 && options.speed !== 1) {
    rows.push({ label: "变速", value: `×${options.speed}` });
  }
  if (enabled(options.reorder)) rows.push({ label: "帧重排", value: "开启" });
  if (enabled(options.shot_retime)) rows.push({ label: "镜头变速", value: "开启" });
  if (num(options.cut_jitter) > 0) rows.push({ label: "切点抖动", value: "开启" });

  const codecLabel =
    options.codec === "libx265" ? "H.265" : options.codec === "libx264" ? "H.264" : null;
  const codecBits = [codecLabel, enabled(options.lossless) ? "无损" : null]
    .filter(Boolean)
    .join(" · ");
  if (codecBits) rows.push({ label: "输出编码", value: codecBits });
  if (typeof options.resolution === "string" && options.resolution !== "保持原始分辨率") {
    rows.push({ label: "输出分辨率", value: options.resolution });
  }
  if (typeof options.bitrate_kbps === "number") {
    rows.push({ label: "输出码率", value: `${options.bitrate_kbps} kbps` });
  }
  if (typeof options.gop === "number") {
    rows.push({ label: "关键帧间隔", value: `${options.gop} 帧` });
  }
  if (typeof options.fps_out === "number") {
    rows.push({ label: "输出帧率", value: `${options.fps_out} fps` });
  }
  if (enabled(options.hardware)) rows.push({ label: "硬件编码", value: "开启" });
  if (enabled(options.skip_vmaf)) rows.push({ label: "快速模式", value: "跳过质量评估" });

  return rows;
}

export function outputTimeLabel(mtime: number): string {
  const date = new Date(mtime * 1000);
  const pad = (value: number) => String(value).padStart(2, "0");
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${pad(
    date.getHours(),
  )}:${pad(date.getMinutes())}`;
}

export function riskLabel(risk: RiskLevel): string {
  return risk;
}
