import { useEffect, useState } from "react";
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
import { Slider } from "@/components/ui/slider";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import {
  createTemplate,
  deleteTemplate,
  listTemplates,
  updateTemplate,
  type TemplateInfo,
} from "@/lib/backend";
import { toast } from "@/stores/toasts";

type CleanLevel = "轻度" | "平衡" | "深度";
type AntiLevel = "关闭" | "轻度" | "标准" | "强力" | "全兵器";
type Codec = "H.264" | "H.265";

/** 与右侧「清洗去重」面板参数一一对应，模板保存完整配置。 */
interface TemplatePayload {
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

const LEVEL_PRESETS: Record<CleanLevel, { retime: number; perturb: number }> = {
  轻度: { retime: 25, perturb: 15 },
  平衡: { retime: 30, perturb: 20 },
  深度: { retime: 40, perturb: 30 },
};

const EMPTY_FORM: TemplatePayload = {
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
function payloadOf(template: TemplateInfo): TemplatePayload {
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

export default function TemplatesView() {
  const [templates, setTemplates] = useState<TemplateInfo[]>([]);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [name, setName] = useState("");
  const [form, setForm] = useState<TemplatePayload>({ ...EMPTY_FORM });
  const [editing, setEditing] = useState<TemplateInfo | null>(null);
  const [pendingDelete, setPendingDelete] = useState<TemplateInfo | null>(null);

  const setField = <K extends keyof TemplatePayload>(key: K, value: TemplatePayload[K]) =>
    setForm((current) => ({ ...current, [key]: value }));

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
      const finalName = name.trim() || `${form.level}档模板`;
      if (editing) {
        await updateTemplate(editing.id, finalName, { ...form });
      } else {
        await createTemplate(finalName, { ...form });
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
    setEditing(null);
    setName("");
    setForm({ ...EMPTY_FORM });
    setDialogOpen(true);
  };

  const openEdit = (template: TemplateInfo) => {
    setEditing(template);
    setName(template.name);
    setForm(payloadOf(template));
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

  type SwitchKey = keyof Pick<
    TemplatePayload,
    | "audioRemix"
    | "antiReembed"
    | "recropOn"
    | "detailProtectOn"
    | "sharpness"
    | "colorRestore"
    | "denoise"
    | "spoof"
    | "lossless"
  >;

  const switchRow = (label: string, desc: string, key: SwitchKey) => (
    <div className="switch">
      <div>
        <div className="switch-label">{label}</div>
        <div className="switch-desc">{desc}</div>
      </div>
      <Switch checked={form[key]} onCheckedChange={(value) => setField(key, value)} />
    </div>
  );

  return (
    <>
      <div className="view-head">
        <div>
          <div className="view-title">去重模板管理</div>
          <div className="view-desc">保存完整的去重与对抗参数，作为可复用的清洗预设</div>
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
                  <div className="tpl-row"><span>清除档位</span><b>{payload.level}</b></div>
                  <div className="tpl-row">
                    <span>变速 / 微扰</span>
                    <b>{payload.retime}% / {payload.perturb}%</b>
                  </div>
                  <div className="tpl-row"><span>指纹对抗</span><b>{payload.anti}</b></div>
                  <div className="tpl-row">
                    <span>空间降噪 / 音频重混</span>
                    <b>{payload.denoise ? "开" : "关"} / {payload.audioRemix ? "开" : "关"}</b>
                  </div>
                  <div className="tpl-row">
                    <span>输出编码</span>
                    <b>{payload.codec}{payload.lossless ? " · 无损" : ""}</b>
                  </div>
                </div>
                <div className="tpl-actions">
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
              <div className="field">
                <Label>清除档位</Label>
                <Select
                  value={form.level}
                  onValueChange={(value) => {
                    const next = value as CleanLevel;
                    setForm((current) => ({
                      ...current,
                      level: next,
                      retime: LEVEL_PRESETS[next].retime,
                      perturb: LEVEL_PRESETS[next].perturb,
                    }));
                  }}
                >
                  <SelectTrigger className="h-9"><SelectValue /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="轻度">轻度（画质优先）</SelectItem>
                    <SelectItem value="平衡">平衡（推荐）</SelectItem>
                    <SelectItem value="深度">深度（最强清除）</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="field">
                <span className="field-label">
                  变速幅度 <span className="field-value">{form.retime}%</span>
                </span>
                <Slider value={[form.retime]} min={0} max={100} onValueChange={([v]) => setField("retime", v)} />
              </div>
              <div className="field">
                <span className="field-label">
                  画质微扰强度 <span className="field-value">{form.perturb}%</span>
                </span>
                <Slider value={[form.perturb]} min={0} max={100} onValueChange={([v]) => setField("perturb", v)} />
              </div>

              {switchRow("同步处理音频指纹", "对音轨做频谱轻微处理", "audioRemix")}
              {switchRow("抗二次检测增强", "对 8×8 中频 DCT 系数施加扰动", "antiReembed")}
              {switchRow("重新构图（裁剪回缩）", "对抗内容指纹，画质损失较大", "recropOn")}
              {switchRow("细节保护（人脸/字幕/纹理）", "保护区域回退原帧，保留部分水印特征", "detailProtectOn")}

              <div className="field">
                <span className="field-label">指纹对抗强度</span>
                <Select value={form.anti} onValueChange={(value) => setField("anti", value as AntiLevel)}>
                  <SelectTrigger className="h-9"><SelectValue /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="关闭">关闭（仅基础清洗）</SelectItem>
                    <SelectItem value="轻度">轻度 · 几乎无损，日常推荐</SelectItem>
                    <SelectItem value="标准">标准 · 轻微损失，正常观看</SelectItem>
                    <SelectItem value="强力">强力 · 可见轻微加工，建议预览</SelectItem>
                    <SelectItem value="全兵器">全兵器 · 仅研究测试，不保证观感</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {switchRow("锐度补偿", "抵消清除算法带来的轻微模糊", "sharpness")}
              {switchRow("色彩还原", "把处理后亮度均值校准回原片统计", "colorRestore")}
              {switchRow("空间降噪", "Wiener 滤波，破坏空域扩频水印", "denoise")}
              {switchRow("伪水印注入（溯源干扰）", "注入随机干扰水印，轻微损失画质", "spoof")}

              <div className="field">
                <Label>输出编码</Label>
                <Select value={form.codec} onValueChange={(value) => setField("codec", value as Codec)}>
                  <SelectTrigger className="h-9"><SelectValue /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="H.264">H.264</SelectItem>
                    <SelectItem value="H.265">H.265</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              {switchRow("无损输出", "极致保留画质，文件体积增大", "lossless")}

              <div className="field">
                <span className="field-label">分辨率策略</span>
                <Select value={form.resolution} onValueChange={(value) => setField("resolution", value)}>
                  <SelectTrigger className="h-9"><SelectValue /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="保持原始分辨率">保持原始分辨率</SelectItem>
                    <SelectItem value="1920x1080">1920×1080</SelectItem>
                    <SelectItem value="1280x720">1280×720</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="section-title">编码参数</div>
              <div className="field-row">
                <div className="field">
                  <span className="field-label">码率（kbps）</span>
                  <Input
                    className="h-9"
                    type="number"
                    min={100}
                    value={form.bitrate ?? ""}
                    onChange={(e) => setField("bitrate", e.target.value ? Number(e.target.value) : null)}
                    placeholder="留空使用 CRF"
                  />
                </div>
                <div className="field">
                  <span className="field-label">帧率（fps）</span>
                  <Input
                    className="h-9"
                    type="number"
                    min={1}
                    value={form.fpsOut ?? ""}
                    onChange={(e) => setField("fpsOut", e.target.value ? Number(e.target.value) : null)}
                    placeholder="留空原帧率"
                  />
                </div>
                <div className="field">
                  <span className="field-label">GOP 帧数</span>
                  <Input
                    className="h-9"
                    type="number"
                    min={1}
                    value={form.gop ?? ""}
                    onChange={(e) => setField("gop", e.target.value ? Number(e.target.value) : null)}
                    placeholder="留空自动"
                  />
                </div>
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
