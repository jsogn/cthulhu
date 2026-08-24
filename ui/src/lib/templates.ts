import type { TemplateInfo } from "@/lib/backend";

export type CleanLevel = "轻度" | "平衡" | "深度";
export type AntiLevel = "关闭" | "轻度" | "标准" | "强力" | "全兵器";
export type Codec = "H.264" | "H.265";

/** 与右侧「清洗去重」面板参数一一对应，模板保存完整配置。 */
export interface TemplatePayload {
  level: CleanLevel;
  retime: number;
  perturb: number;
  audioRemix: boolean;
  antiReembed: boolean;
  anti: AntiLevel;
  recropOn: boolean;
  detailProtectOn: boolean;
  sharpness: boolean;
  colorRestore: boolean;
  denoise: boolean;
  spoof: boolean;
  codec: Codec;
  lossless: boolean;
  resolution: string;
  bitrate: number | null;
  gop: number | null;
  fpsOut: number | null;
}

export const LEVEL_PRESETS: Record<CleanLevel, { retime: number; perturb: number }> = {
  轻度: { retime: 25, perturb: 15 },
  平衡: { retime: 30, perturb: 20 },
  深度: { retime: 40, perturb: 30 },
};

export const DEFAULT_TEMPLATE: TemplatePayload = {
  level: "平衡",
  retime: 30,
  perturb: 20,
  audioRemix: true,
  antiReembed: false,
  anti: "关闭",
  recropOn: false,
  detailProtectOn: false,
  sharpness: true,
  colorRestore: true,
  denoise: true,
  spoof: false,
  codec: "H.264",
  lossless: false,
  resolution: "保持原始分辨率",
  bitrate: null,
  gop: null,
  fpsOut: null,
};

/** 读取模板参数；兼容旧版 restruct/audio 字段。 */
export function payloadOf(template: TemplateInfo): TemplatePayload {
  const raw = (template.payload ?? {}) as Record<string, unknown>;
  return {
    level: (raw.level as CleanLevel) ?? "平衡",
    retime: (raw.retime as number) ?? (raw.restruct as number) ?? 30,
    perturb: (raw.perturb as number) ?? 20,
    audioRemix: (raw.audioRemix as boolean) ?? (raw.audio as boolean) ?? true,
    antiReembed: (raw.antiReembed as boolean) ?? false,
    anti: (raw.anti as AntiLevel) ?? "关闭",
    recropOn: (raw.recropOn as boolean) ?? false,
    detailProtectOn: (raw.detailProtectOn as boolean) ?? false,
    sharpness: (raw.sharpness as boolean) ?? true,
    colorRestore: (raw.colorRestore as boolean) ?? true,
    denoise: (raw.denoise as boolean) ?? true,
    spoof: (raw.spoof as boolean) ?? false,
    codec: (raw.codec as Codec) ?? "H.264",
    lossless: (raw.lossless as boolean) ?? false,
    resolution: (raw.resolution as string) ?? "保持原始分辨率",
    bitrate: (raw.bitrate as number) ?? null,
    gop: (raw.gop as number) ?? null,
    fpsOut: (raw.fpsOut as number) ?? null,
  };
}
