import { useEffect, useState, type ReactNode } from "react";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import {
  WeaponTags,
  type CostKind,
  type LayerKind,
  type QualityKind,
} from "@/components/workbench/WeaponTags";
import {
  createTemplate,
  deleteTemplate,
  listTemplates,
  updateTemplate,
  type TemplateInfo,
} from "@/lib/backend";
import {
  DEFAULT_TEMPLATE,
  payloadOf,
  type Codec,
  type TemplatePayload,
} from "@/lib/templates";
import {
  DEFAULT_TIER,
  SCHEMA_DEFAULT_TIER,
  SELECTABLE_TIERS,
  tierById,
  tierIdOf,
} from "@/lib/tiers";
import { useAppStore } from "@/stores/app";
import { toast } from "@/stores/toasts";

const NUMERIC_KEYS = [
  "rotate",
  "warp",
  "perspective",
  "jitter",
  "requant",
  "noise",
  "dctStep",
  "temporalSub",
  "fftPhase",
  "dwtDetail",
  "facePerturb",
  "lpcAttack",
  "copyAttack",
] as const;

type NumericKey = (typeof NUMERIC_KEYS)[number];
type NumericEnabled = Record<NumericKey, boolean>;

/** 数值武器用独立开关保留上次强度，避免关闭再打开时被重置。 */
function numericEnabled(payload: TemplatePayload): NumericEnabled {
  const enabled = Object.fromEntries(
    NUMERIC_KEYS.map((key) => [key, (payload[key] ?? 0) > 0]),
  ) as NumericEnabled;
  return enabled;
}

export default function TemplatesView() {
  const [templates, setTemplates] = useState<TemplateInfo[]>([]);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [name, setName] = useState("");
  const [form, setForm] = useState<TemplatePayload>({ ...DEFAULT_TEMPLATE });
  const [enabled, setEnabled] = useState<NumericEnabled>(() => numericEnabled(DEFAULT_TEMPLATE));
  const [editing, setEditing] = useState<TemplateInfo | null>(null);
  const [pendingDelete, setPendingDelete] = useState<TemplateInfo | null>(null);
  const queueTemplate = useAppStore((state) => state.queueTemplate);
  const setView = useAppStore((state) => state.setView);

  const setField = <K extends keyof TemplatePayload>(key: K, value: TemplatePayload[K]) =>
    setForm((current) => ({ ...current, [key]: value }));

  const setNumericEnabled = (key: NumericKey, value: boolean) =>
    setEnabled((current) => ({ ...current, [key]: value }));

  const refresh = async () => {
    try {
      setTemplates(await listTemplates());
    } catch {
      // 后端未就绪时保持现状
    }
  };

  useEffect(() => {
    void refresh();
  }, []);

  const save = async () => {
    try {
      const finalName = name.trim();
      if (!finalName) {
        toast("请先填写模板名称");
        return;
      }
      const payload = { ...form };
      for (const key of NUMERIC_KEYS) {
        if (!enabled[key]) payload[key] = 0;
      }
      if (editing) {
        await updateTemplate(editing.id, finalName, payload);
      } else {
        await createTemplate(finalName, payload);
      }
      await refresh();
      setDialogOpen(false);
      setName("");
      setEditing(null);
      toast("模板已保存");
    } catch {
      toast("模板操作失败，请确认引擎在线");
    }
  };

  const openCreate = () => {
    const payload = { ...DEFAULT_TEMPLATE };
    setEditing(null);
    setName("");
    setForm(payload);
    setEnabled(numericEnabled(payload));
    setDialogOpen(true);
  };

  const openEdit = (template: TemplateInfo) => {
    const payload = payloadOf(template);
    setEditing(template);
    setName(template.name);
    setForm(payload);
    setEnabled(numericEnabled(payload));
    setDialogOpen(true);
  };

  const remove = async () => {
    if (!pendingDelete) return;
    try {
      await deleteTemplate(pendingDelete.id);
      await refresh();
      toast(`已删除模板：${pendingDelete.name}`);
    } catch {
      toast("删除失败，请确认引擎在线");
    } finally {
      setPendingDelete(null);
    }
  };

  const duplicate = async (template: TemplateInfo) => {
    try {
      await createTemplate(`${template.name} 副本`, template.payload);
      await refresh();
      toast("已复制模板");
    } catch {
      toast("复制失败，请确认引擎在线");
    }
  };

  const useTemplate = (template: TemplateInfo) => {
    queueTemplate({ id: template.id, name: template.name, payload: payloadOf(template) });
    setView("workbench");
    toast(`已套用模板：${template.name}`, "参数已回填到右侧，可调整后执行清洗");
  };

  type SwitchKey = keyof Pick<
    TemplatePayload,
    | "audioRemix"
    | "echoDefeat"
    | "hashAttack"
    | "audioStrong"
    | "nativeTemporal"
    | "recropOn"
    | "regradeOn"
    | "sharpness"
    | "colorRestore"
    | "denoise"
    | "spoof"
    | "qualityProtect"
    | "autoProfile"
    | "lossless"
  >;

  const weaponCard = (
    label: string,
    desc: string,
    layer: LayerKind,
    cost: CostKind,
    quality: QualityKind,
    checked: boolean,
    onCheckedChange: (value: boolean) => void,
    body?: ReactNode,
  ) => (
    <div className="switch-card">
      <div className="switch">
        <div>
          <div className="switch-label">{label}</div>
          <div className="switch-desc">{desc}</div>
          <WeaponTags layer={layer} cost={cost} quality={quality} />
        </div>
        <Switch aria-label={label} checked={checked} onCheckedChange={onCheckedChange} />
      </div>
      {body ? <div className="switch-body">{body}</div> : null}
    </div>
  );

  const switchRow = (
    label: string,
    desc: string,
    layer: LayerKind,
    cost: CostKind,
    quality: QualityKind,
    key: SwitchKey,
  ) =>
    weaponCard(label, desc, layer, cost, quality, form[key], (value) =>
      setField(key, value),
    );

  type SliderSpec = {
    min: number;
    max: number;
    step: number;
    onValue: number;
    ariaLabel: string;
    label: (value: number) => string;
  };

  const sliderRow = (
    label: string,
    desc: string,
    layer: LayerKind,
    cost: CostKind,
    quality: QualityKind,
    key: NumericKey,
    spec: SliderSpec,
  ) => {
    const value = form[key] ?? spec.onValue;
    return weaponCard(
      label,
      desc,
      layer,
      cost,
      quality,
      enabled[key],
      (checked) => {
        setNumericEnabled(key, checked);
        if (checked && (form[key] ?? 0) <= 0) setField(key, spec.onValue);
      },
      enabled[key] ? (
        <div className="field">
          <span className="field-label">{spec.label(value)}</span>
          <Slider
            aria-label={spec.ariaLabel}
            value={[value]}
            min={spec.min}
            max={spec.max}
            step={spec.step}
            onValueChange={(values) => setField(key, values[0] ?? spec.onValue)}
          />
        </div>
      ) : undefined,
    );
  };

  const hashRow = weaponCard(
    "哈希签名对抗（pHash/dHash）",
    "签名域可微扰动，翻转感知哈希符号位（专攻模式效果更强）",
    "fp",
    "mid",
    "heavy",
    form.hashAttack,
    (value) => setField("hashAttack", value),
    form.hashAttack ? (
      <>
        <div className="field">
          <span className="field-label">目标模式</span>
          <Select value={form.hashMode} onValueChange={(value) => setField("hashMode", value)}>
            <SelectTrigger className="form-input h-8 w-full"><SelectValue /></SelectTrigger>
            <SelectContent>
              <SelectItem value="phash">pHash 专攻（推荐）</SelectItem>
              <SelectItem value="dhash">dHash 专攻</SelectItem>
              <SelectItem value="joint">联合（pHash+dHash）</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <div className="field">
          <span className="field-label">扰动预算 ε {form.hashEpsilon.toFixed(3)}</span>
          <Slider
            aria-label="哈希扰动预算"
            value={[form.hashEpsilon]}
            min={0.02}
            max={0.12}
            step={0.01}
            onValueChange={(values) => setField("hashEpsilon", values[0] ?? 0.08)}
          />
        </div>
      </>
    ) : undefined,
  );

  const temporalSubRow = weaponCard(
    "跨帧估计相减",
    "估计帧间固定水印并过减，会削弱静态字幕/背景边缘",
    "wm",
    "mid",
    "heavy",
    enabled.temporalSub,
    (checked) => {
      setNumericEnabled("temporalSub", checked);
      if (checked && (form.temporalSub ?? 0) <= 0) setField("temporalSub", 0.6);
    },
    enabled.temporalSub ? (
      <>
        <div className="field">
          <span className="field-label">过减强度 β {form.temporalSub.toFixed(1)}</span>
          <Slider
            aria-label="跨帧过减强度"
            value={[form.temporalSub]}
            min={0.2}
            max={1.5}
            step={0.1}
            onValueChange={(values) => setField("temporalSub", values[0] ?? 0.6)}
          />
        </div>
        <div className="switch-inline">
          <div>
            <div className="switch-label">原生加速(ffmpeg)</div>
            <div className="switch-desc">8bit 原生链，破坏力更强，不支持分镜重排</div>
          </div>
          <Switch
            aria-label="跨帧估计原生加速"
            checked={form.nativeTemporal}
            onCheckedChange={(value) => setField("nativeTemporal", value)}
          />
        </div>
      </>
    ) : undefined,
  );

  const fftOn = enabled.fftPhase;
  const fftRow = weaponCard(
    "FFT 相位扰动",
    "打散中高频相位，覆盖依赖相位相关的 DFT 域水印",
    "wm",
    "mid",
    "heavy",
    fftOn,
    (checked) => {
      setNumericEnabled("fftPhase", checked);
      if (checked && (form.fftPhase ?? 0) <= 0) setField("fftPhase", 0.5);
    },
    fftOn ? (
      <div className="field">
        <span className="field-label">相位强度 {form.fftPhase.toFixed(2)}</span>
        <Slider
          aria-label="FFT 相位强度"
          value={[form.fftPhase]}
          min={0.1}
          max={1}
          step={0.05}
          onValueChange={(values) => setField("fftPhase", values[0] ?? 0.5)}
        />
      </div>
    ) : undefined,
  );

  const qualityProtectRow = weaponCard(
    "画质保护（PSNR/SSIM 门控）",
    `对攻击/重武器层按目标自动回退，PSNR≥${form.psnrTarget}dB、SSIM≥${form.ssimTarget}（原生滤镜链除外）`,
    "quality",
    "mid",
    "none",
    form.qualityProtect,
    (value) => setField("qualityProtect", value),
    form.qualityProtect ? (
      <>
        <div className="field">
          <span className="field-label">PSNR 目标 {form.psnrTarget} dB</span>
          <Slider
            aria-label="PSNR 目标"
            value={[form.psnrTarget]}
            min={32}
            max={44}
            step={1}
            onValueChange={(values) => setField("psnrTarget", values[0] ?? 38)}
          />
        </div>
        <div className="field">
          <span className="field-label">SSIM 目标 {form.ssimTarget.toFixed(2)}</span>
          <Slider
            aria-label="SSIM 目标"
            value={[form.ssimTarget]}
            min={0.8}
            max={0.98}
            step={0.01}
            onValueChange={(values) => setField("ssimTarget", values[0] ?? 0.94)}
          />
        </div>
      </>
    ) : undefined,
  );

  const purifyRow = weaponCard(
    "画面重建（去暗水印）",
    "按画面内容重新生成一遍，抹掉嵌进去的暗水印；字幕与人脸是否清楚由下方档位决定",
    "wm",
    "fast",
    "mild",
    (form.purifyStrength ?? 0) > 0,
    (checked) => {
      setField("purifyStrength", checked ? DEFAULT_TIER.strength : 0);
    },
    (form.purifyStrength ?? 0) > 0 ? (
      <>
        <div className="field">
          <span className="field-label">清晰度档位</span>
          <Select
            aria-label="净化档位"
            value={tierIdOf(
              form.purifyMaxEdge ?? SCHEMA_DEFAULT_TIER.max_edge,
              form.purifyDetailWide ?? SCHEMA_DEFAULT_TIER.detail_wide,
            )}
            onValueChange={(value) => {
              const tier = tierById(value);
              if (!tier) return;
              setField("purifyMaxEdge", tier.max_edge);
              setField("purifyBatch", tier.batch);
              setField("purifyDetailWide", tier.detail_wide);
            }}
          >
            <SelectTrigger className="form-input h-8 w-full">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {SELECTABLE_TIERS.map((tier) => (
                <SelectItem key={tier.id} value={tier.id}>
                  {tier.label} · 长边 {tier.max_edge} · 批处理 {tier.batch}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <div className="field">
          <span className="field-label">
            细节回注 {(form.purifyDetail ?? 1).toFixed(2)}
          </span>
          <Slider
            aria-label="净化细节回注"
            value={[form.purifyDetail ?? 1]}
            min={0}
            max={1}
            step={0.05}
            onValueChange={(values) => setField("purifyDetail", values[0] ?? 1)}
          />
        </div>
        <div className="field">
          <span className="field-label">
            细节带宽 σ {form.purifyDetailSigma ? form.purifyDetailSigma.toFixed(2) : "自动"}
          </span>
          <Slider
            aria-label="净化细节带宽"
            value={[form.purifyDetailSigma ?? 0]}
            min={0}
            max={3}
            step={0.1}
            onValueChange={(values) => setField("purifyDetailSigma", values[0] ?? 0)}
          />
          <div className="form-help">
            0 = 按分辨率自动（512p≈1.2、1080p≈2.6）；手动超过 3.0 部分方案水印会回流
          </div>
        </div>
        <div className="field">
          <span className="field-label">
            时序一致性减法 {(form.purifyTemporal ?? 0).toFixed(2)}
          </span>
          <Slider
            aria-label="净化时序减法"
            value={[form.purifyTemporal ?? 0]}
            min={0}
            max={1}
            step={0.05}
            onValueChange={(values) => setField("purifyTemporal", values[0] ?? 0)}
          />
        </div>
      </>
    ) : undefined,
  );

  const embeddingRow = weaponCard(
    "嵌入域定向重写",
    "按水印画像改写低频：VideoSeal 类打 Y 亮度，WAM 类打 Cb/Cr 色度",
    "wm",
    "fast",
    "mild",
    (form.embeddingStrength ?? 0) > 0 && (form.embeddingAttack ?? "") !== "",
    (checked) => {
      setField("embeddingAttack", checked ? form.embeddingAttack || "chroma" : "");
      setField("embeddingStrength", checked ? form.embeddingStrength || 0.25 : 0);
    },
    (form.embeddingStrength ?? 0) > 0 && (form.embeddingAttack ?? "") !== "" ? (
      <>
        <div className="field">
          <span className="field-label">目标域</span>
          <Select
            value={form.embeddingAttack}
            onValueChange={(value) =>
              setField("embeddingAttack", value as TemplatePayload["embeddingAttack"])
            }
          >
            <SelectTrigger className="form-input h-8 w-full">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="auto">自动（已知方案按画像，未知 both）</SelectItem>
              <SelectItem value="luma">Y 亮度低频（VideoSeal 类）</SelectItem>
              <SelectItem value="chroma">Cb/Cr 色度低频（WAM 类）</SelectItem>
              <SelectItem value="both">两者同时</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <div className="field">
          <span className="field-label">
            重写强度 {(form.embeddingStrength ?? 0.25).toFixed(2)}
          </span>
          <Slider
            aria-label="嵌入域重写强度"
            value={[form.embeddingStrength ?? 0.25]}
            min={0.1}
            max={1}
            step={0.05}
            onValueChange={(values) => setField("embeddingStrength", values[0] ?? 0.25)}
          />
        </div>
        <div className="switch">
          <div>
            <div className="switch-label">精确频带重写</div>
            <div className="switch-desc">精确 Top-10 频带 + 随机量化</div>
          </div>
          <Switch
            checked={(form.embeddingVariant ?? "v2") === "v2"}
            onCheckedChange={(checked) =>
              setField("embeddingVariant", checked ? "v2" : "legacy")
            }
          />
        </div>
        <div className="switch">
          <div>
            <div className="switch-label">增强重写</div>
            <div className="switch-desc">随机分块相位 + 多尺度 + 4:2:0 色度对齐</div>
          </div>
          <Switch
            checked={form.embeddingAggressive ?? false}
            onCheckedChange={(checked) => setField("embeddingAggressive", checked)}
          />
        </div>
      </>
    ) : undefined,
  );

  return (
    <>
      <div className="view-head">
        <div>
          <div className="view-title">去重模板管理</div>
          <div className="view-desc">保存完整去重参数，点击「使用」在素材处理页一键回填</div>
        </div>
        <Button onClick={openCreate}>+ 新建模板</Button>
      </div>

      <div className="view-body">
        <div className="tpl-grid">
          {templates.map((template) => {
            const payload = payloadOf(template);
            return (
              <Card key={template.id} className="flex flex-col gap-3 p-4">
                <div className="tpl-name">{template.name}</div>
                <div className="tpl-rows">
                  <div className="tpl-row">
                    <span>经典指纹</span>
                    <b>
                      {[
                        payload.rotate,
                        payload.hashAttack ? 1 : 0,
                        payload.requant,
                        payload.noise,
                        payload.dctStep,
                      ].filter((value) => (value ?? 0) > 0).length}{" "}
                      项
                    </b>
                  </div>
                  <div className="tpl-row">
                    <span>空间降噪 / 音频重混</span>
                    <b>{payload.denoise ? "开" : "关"} / {payload.audioRemix ? "开" : "关"}</b>
                  </div>
                  <div className="tpl-row">
                    <span>重编码与信号扰动</span>
                    <b>
                      {[
                        payload.temporalSub,
                        payload.fftPhase,
                        payload.dwtDetail,
                      ].filter((value) => (value ?? 0) > 0).length}{" "}
                      项
                    </b>
                  </div>
                  <div className="tpl-row">
                    <span>潜空间净化 / 嵌入域</span>
                    <b>
                      {(payload.purifyStrength ?? 0) > 0 ? "开" : "关"} /{" "}
                      {(payload.embeddingStrength ?? 0) > 0 ? "开" : "关"}
                    </b>
                  </div>
                  <div className="tpl-row">
                    <span>输出编码</span>
                    <b>{payload.codec}{payload.lossless ? " · 无损" : ""}</b>
                  </div>
                </div>
                <div className="tpl-actions">
                  <Button size="sm" onClick={() => useTemplate(template)}>
                    使用
                  </Button>
                  <Button variant="secondary" size="sm" onClick={() => openEdit(template)}>
                    编辑
                  </Button>
                  <Button variant="ghost" size="sm" onClick={() => void duplicate(template)}>
                    复制
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="text-destructive"
                    onClick={() => setPendingDelete(template)}
                  >
                    删除
                  </Button>
                </div>
              </Card>
            );
          })}
          {templates.length === 0 && (
            <p className="text-muted-foreground">暂无模板，点击右上角新建。</p>
          )}
        </div>
      </div>

      <Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
        <DialogContent className="sm:max-w-lg">
          <DialogHeader>
            <DialogTitle>{editing ? "编辑去重模板" : "新建去重模板"}</DialogTitle>
          </DialogHeader>
          <ScrollArea className="pr-2" style={{ maxHeight: "65vh" }}>
            <div className="flex flex-col gap-3">
              <div className="field">
                <Label>模板名称</Label>
                <Input
                  className="h-9"
                  placeholder="如：抖音投流-轻度"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                />
              </div>
              {switchRow("重新构图（裁剪回缩）", "对暗水印无效，用于判重指纹；文字位置不固定，可能裁到字幕，仅确认边缘无文字时使用", "fp", "fast", "heavy", "recropOn")}
              {sliderRow(
                "几何微旋转",
                "低频小幅旋转去同步，破坏哈希与块对齐",
                "dual",
                "fast",
                "mild",
                "rotate",
                {
                  min: 0.1,
                  max: 2.5,
                  step: 0.1,
                  onValue: 0.4,
                  ariaLabel: "几何微旋转振幅",
                  label: (value) => `旋转振幅 ${value.toFixed(1)}°`,
                },
              )}
              {sliderRow(
                "局部平滑扭曲",
                "粗网格位移场上采样成平滑光流逐帧重采样，破坏空间对齐",
                "wm",
                "mid",
                "mild",
                "warp",
                {
                  min: 0.001,
                  max: 0.01,
                  step: 0.001,
                  onValue: 0.005,
                  ariaLabel: "局部平滑扭曲位移跨度",
                  label: (value) => `位移跨度 ${(value * 100).toFixed(1)}% 短边`,
                },
              )}
              {sliderRow(
                "透视剪切",
                "逐帧轻微梯形畸变，破坏块对齐与几何同步",
                "wm",
                "mid",
                "mild",
                "perspective",
                {
                  min: 0.005,
                  max: 0.03,
                  step: 0.005,
                  onValue: 0.01,
                  ariaLabel: "透视剪切强度",
                  label: (value) => `剪切强度 ${value.toFixed(3)}`,
                },
              )}
              {sliderRow(
                "平移抖动",
                "低频正弦漂移（相邻帧 ≤1px），破坏逐帧对齐，观感轻微",
                "wm",
                "fast",
                "mild",
                "jitter",
                {
                  min: 0.005,
                  max: 0.03,
                  step: 0.005,
                  onValue: 0.005,
                  ariaLabel: "平移抖动幅度",
                  label: (value) => `抖动幅度 ${(value * 100).toFixed(1)}%`,
                },
              )}
              {switchRow(
                "色彩微扰（伽马/亮度+色相）",
                "逐帧伽马/亮度微调并叠加 ±6° 色相抖动，对抗时域与色彩描述子",
                "dual",
                "fast",
                "mild",
                "regradeOn",
              )}

              <div className="section-title">音频处理</div>
              {switchRow("同步处理音频指纹", "对音轨做等长频谱轻处理，不影响音画同步", "audio", "fast", "none", "audioRemix")}
              {switchRow("音频回声扰动", "同步放慢约 3% 并保音调", "audio", "fast", "mild", "echoDefeat")}
              {sliderRow(
                "LPCAA 音频攻击",
                "LPC 残差白化，破坏语音类指纹 · 高强可听出改变",
                "audio",
                "fast",
                "heavy",
                "lpcAttack",
                {
                  min: 0.2,
                  max: 1,
                  step: 0.05,
                  onValue: 0.5,
                  ariaLabel: "LPCAA 白化强度",
                  label: (value) => `白化强度 ${value.toFixed(2)}`,
                },
              )}
              {switchRow("强音频重混", "变调+EQ 倾斜+底噪，配合音频指纹开关生效", "audio", "fast", "heavy", "audioStrong")}

              <div className="section-title">去水印 · 画面重建</div>
              {purifyRow}
              {embeddingRow}
              {switchRow(
                "自动画像（内容复杂度自适应）",
                "按镜头纹理/运动/时序一致性自适应净化强度与时序减法；边缘、细节带宽与字幕增强按上方档位执行，不参与自适应",
                "quality",
                "fast",
                "none",
                "autoProfile",
              )}

              <div className="section-title">重编码与信号扰动</div>
              {hashRow}
              {sliderRow(
                "像素重量化",
                "把像素压缩到有限级数，破坏低位与细微扰动",
                "wm",
                "fast",
                "none",
                "requant",
                {
                  min: 32,
                  max: 96,
                  step: 8,
                  onValue: 64,
                  ariaLabel: "像素重量化级数",
                  label: (value) => `量化级数 ${value}`,
                },
              )}
              {sliderRow(
                "微噪声",
                "加性微高斯噪声，破坏扩频水印的低位相关",
                "wm",
                "fast",
                "none",
                "noise",
                {
                  min: 0.001,
                  max: 0.01,
                  step: 0.001,
                  onValue: 0.004,
                  ariaLabel: "微噪声强度",
                  label: (value) => `噪声强度 ${value.toFixed(3)}`,
                },
              )}
              {switchRow("空间降噪", "removegrain 轻降噪，破坏扩频/QIM/DWT 水印", "wm", "fast", "none", "denoise")}
              {temporalSubRow}
              {fftRow}
              {sliderRow(
                "小波细节带随机化",
                "随机化 Haar 对角线细节子带，破坏小波嵌入相关",
                "wm",
                "fast",
                "none",
                "dwtDetail",
                {
                  min: 0.1,
                  max: 1,
                  step: 0.05,
                  onValue: 0.8,
                  ariaLabel: "小波细节随机化强度",
                  label: (value) => `随机化强度 ${value.toFixed(2)}`,
                },
              )}
              {sliderRow(
                "DCT 系数扰动（重量化+中频）",
                "8×8 DCT 重量化并扰动中频系数",
                "wm",
                "mid",
                "mild",
                "dctStep",
                {
                  min: 1,
                  max: 256,
                  step: 1,
                  onValue: 12,
                  ariaLabel: "DCT 量化步长",
                  label: (value) => `量化步长 ${value}`,
                },
              )}
              {switchRow("伪水印注入（溯源干扰）", "注入随机干扰水印，对抗上传后二次嵌入 · 需投流实测", "wm", "mid", "mild", "spoof")}
              {sliderRow(
                "神经对抗（DINOv2 判重代理）",
                "黑盒攻击 DINOv2 判重描述子，压拷贝检测 embedding · 较慢",
                "fp",
                "slow",
                "heavy",
                "copyAttack",
                {
                  min: 0.01,
                  max: 0.08,
                  step: 0.01,
                  onValue: 0.05,
                  ariaLabel: "神经对抗扰动预算",
                  label: (value) => `扰动预算 ±${value.toFixed(2)}`,
                },
              )}
              {sliderRow(
                "人脸抗AI扰动",
                "仅在脸部区域注入扰动，破坏人脸识别特征",
                "face",
                "mid",
                "none",
                "facePerturb",
                {
                  min: 0.01,
                  max: 0.08,
                  step: 0.01,
                  onValue: 0.04,
                  ariaLabel: "人脸扰动强度",
                  label: (value) => `扰动强度 ${value.toFixed(2)}`,
                },
              )}

              <div className="section-title">画质优化</div>
              {qualityProtectRow}
              {switchRow("锐度补偿", "抵消清除算法带来的轻微模糊", "quality", "fast", "none", "sharpness")}
              {switchRow("色彩还原", "把处理后亮度均值校准回原片统计", "quality", "fast", "none", "colorRestore")}

              <div className="section-title">输出编码 · MP4</div>
              <div className="field">
                <Label>输出编码</Label>
                <Select value={form.codec} onValueChange={(value) => setField("codec", value as Codec)}>
                  <SelectTrigger className="h-9 w-full"><SelectValue /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="H.264">H.264</SelectItem>
                    <SelectItem value="H.265">H.265</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              {switchRow("无损输出", "极致保留画质，文件体积增大", "quality", "fast", "none", "lossless")}

              <div className="field">
                <span className="field-label">分辨率策略</span>
                <Select value={form.resolution} onValueChange={(value) => setField("resolution", value)}>
                  <SelectTrigger className="h-9 w-full"><SelectValue /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="保持原始分辨率">保持原始分辨率</SelectItem>
                    <SelectItem value="1920x1080">1920×1080</SelectItem>
                    <SelectItem value="1280x720">1280×720</SelectItem>
                  </SelectContent>
                </Select>
              </div>

            </div>
          </ScrollArea>
          <DialogFooter>
            <Button variant="ghost" onClick={() => setDialogOpen(false)}>取消</Button>
            <Button onClick={() => void save()}>保存模板</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <AlertDialog open={!!pendingDelete} onOpenChange={(open) => !open && setPendingDelete(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>确认删除模板</AlertDialogTitle>
            <AlertDialogDescription>
              将删除模板「{pendingDelete?.name}」，此操作不可撤销。
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>取消</AlertDialogCancel>
            <AlertDialogAction onClick={() => void remove()}>确认删除</AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  );
}
