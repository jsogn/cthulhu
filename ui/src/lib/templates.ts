import type { TemplateInfo } from "@/lib/backend";

export type AntiLevel = "关闭" | "轻度" | "标准" | "强力" | "全兵器";
export type Codec = "H.264" | "H.265";

/** 与右侧「清洗去重」面板参数一一对应，模板保存完整配置。 */
export interface TemplatePayload {
  audioRemix: boolean;
  echoDefeat: boolean;
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

export const DEFAULT_TEMPLATE: TemplatePayload = {
  audioRemix: true,
  echoDefeat: false,
  antiReembed: false,
  anti: "轻度",
  recropOn: true,
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
    audioRemix: (raw.audioRemix as boolean) ?? (raw.audio as boolean) ?? true,
    echoDefeat: (raw.echoDefeat as boolean) ?? false,
    antiReembed: (raw.antiReembed as boolean) ?? false,
    anti: (raw.anti as AntiLevel) ?? "轻度",
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
