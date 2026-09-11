/** 由 backend/scripts/export_openapi.py 从 cthulhu_backend/tiers.py 生成；请勿手改。 */

export type TierId = "quality" | "balanced" | "strong" | "max";

export interface Tier {
  id: TierId;
  label: string;
  summary: string;
  max_edge: number;
  batch: number;
  detail_wide: boolean;
  strength: number;
  temporal: number;
  selectable: boolean;
}

export const TIER_DETAIL = 1.0;
export const TIER_DETAIL_SIGMA = 0.0;

export const TIERS: readonly Tier[] = [
  {
    id: "quality",
    label: "画质优先",
    summary: "画面清晰、人脸与字幕正常；水印清除较弱，适合要成片质量的场景",
    max_edge: 512,
    batch: 8,
    detail_wide: true,
    strength: 0.15,
    temporal: 0.0,
    selectable: true,
  },
  {
    id: "balanced",
    label: "平衡",
    summary: "画面略软，清除率比画质优先更好，是人人都能接受的中间档",
    max_edge: 256,
    batch: 8,
    detail_wide: false,
    strength: 0.15,
    temporal: 0.0,
    selectable: true,
  },
  {
    id: "strong",
    label: "清除优先",
    summary: "水印清除最彻底；画面会明显变软、字幕可能难以辨认",
    max_edge: 192,
    batch: 8,
    detail_wide: false,
    strength: 0.35,
    temporal: 0.5,
    selectable: true,
  },
  {
    id: "max",
    label: "彻底清剿",
    summary: "最激进的长边，只供「深度清剿」预置使用",
    max_edge: 128,
    batch: 8,
    detail_wide: false,
    strength: 0.35,
    temporal: 0.5,
    selectable: false,
  },
];

export const DEFAULT_TIER_ID: TierId = "quality";
export const SCHEMA_DEFAULT_TIER_ID: TierId = "balanced";

export const SELECTABLE_TIERS: readonly Tier[] = TIERS.filter(
  (tier) => tier.selectable,
);
