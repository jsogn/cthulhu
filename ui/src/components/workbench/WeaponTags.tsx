export type LayerKind = "wm" | "fp" | "dual" | "audio" | "face" | "quality";
export type CostKind = "fast" | "mid" | "slow";
export type QualityKind = "none" | "mild" | "heavy";

const LAYER_TEXT: Record<Exclude<LayerKind, "dual">, string> = {
  wm: "暗水印",
  fp: "判重指纹",
  audio: "音频",
  face: "人脸",
  quality: "画质",
};

const LAYER_STYLE: Record<Exclude<LayerKind, "dual">, string> = {
  wm: "layer-wm",
  fp: "layer-fp",
  audio: "layer-misc",
  face: "layer-misc",
  quality: "layer-misc",
};

const COST_TEXT: Record<CostKind, string> = { fast: "快", mid: "中", slow: "慢" };

const QUALITY_TEXT: Record<QualityKind, string> = {
  none: "近无损",
  mild: "轻微",
  heavy: "明显",
};

/** 武器标签行：目标层（双目标拆两枚）+ 耗时档 + 画质档，按固定顺序排列。 */
export function WeaponTags({
  layer,
  cost,
  quality,
}: {
  layer: LayerKind;
  cost: CostKind;
  quality: QualityKind;
}) {
  const layers: Exclude<LayerKind, "dual">[] = layer === "dual" ? ["wm", "fp"] : [layer];
  return (
    <div className="weapon-tags">
      {layers.map((tag) => (
        <span key={tag} className={`layer-tag ${LAYER_STYLE[tag]}`}>
          {LAYER_TEXT[tag]}
        </span>
      ))}
      <span className="weapon-meta">
        耗时：{COST_TEXT[cost]} · 画质：{QUALITY_TEXT[quality]}
      </span>
    </div>
  );
}
