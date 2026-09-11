/** 由 backend/scripts/export_openapi.py 从 transform/profile.py 生成；请勿手改。 */

export type SchemeFamilyId = "luma" | "chroma";

export interface SchemeFamily {
  id: SchemeFamilyId;
  label: string;
  summary: string;
  attack: string;
  max_edge: number;
}

export const SCHEME_FAMILIES: readonly SchemeFamily[] = [
  {
    id: "luma",
    label: "亮度型水印（VideoSeal / PixelSeal 系）",
    summary: "亮度/低频嵌入：长边必须压到 192 才有效，选了会自动收窄清晰度档位",
    attack: "luma",
    max_edge: 192,
  },
  {
    id: "chroma",
    label: "色度型水印（WAM 系）",
    summary: "色度/低频嵌入：长边 256 就够，不必牺牲画质",
    attack: "chroma",
    max_edge: 256,
  },
];
