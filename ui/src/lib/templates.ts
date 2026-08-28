import type { components } from "@/lib/api-types";
import type { TemplateInfo } from "@/lib/backend";

/** 模板 payload：与后端 Pydantic TemplatePayload 契约一致，由 OpenAPI 生成。 */
export type TemplatePayload = components["schemas"]["TemplatePayload"];
export type AntiLevel = TemplatePayload["anti"];
export type Codec = TemplatePayload["codec"];

export const DEFAULT_TEMPLATE: TemplatePayload = {
  audioRemix: false,
  echoDefeat: false,
  antiReembed: false,
  anti: "关闭",
  regradeOn: false,
  recropOn: false,
  detailProtectOn: false,
  sharpness: false,
  colorRestore: false,
  denoise: false,
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
    anti: (raw.anti as AntiLevel) ?? "标准",
    regradeOn: (raw.regradeOn as boolean) ?? true,
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
