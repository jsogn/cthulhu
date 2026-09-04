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
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
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
import { useAppStore } from "@/stores/app";
import { toast } from "@/stores/toasts";

export default function TemplatesView() {
  const [templates, setTemplates] = useState<TemplateInfo[]>([]);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [name, setName] = useState("");
  const [form, setForm] = useState<TemplatePayload>({ ...DEFAULT_TEMPLATE });
  const [editing, setEditing] = useState<TemplateInfo | null>(null);
  const [pendingDelete, setPendingDelete] = useState<TemplateInfo | null>(null);
  const queueTemplate = useAppStore((state) => state.queueTemplate);
  const setView = useAppStore((state) => state.setView);

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
      const finalName = name.trim();
      if (!finalName) {
        toast("请先填写模板名称");
        return;
      }
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
    setForm({ ...DEFAULT_TEMPLATE });
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
    | "sharpness"
    | "colorRestore"
    | "denoise"
    | "spoof"
    | "qualityProtect"
    | "lossless"
  >;

  const switchRow = (
    label: string,
    desc: string,
    layer: LayerKind,
    cost: CostKind,
    quality: QualityKind,
    key: SwitchKey,
  ) => (
    <div className="switch">
      <div>
        <div className="switch-label">{label}</div>
        <div className="switch-desc">{desc}</div>
        <WeaponTags layer={layer} cost={cost} quality={quality} />
      </div>
      <Switch checked={form[key]} onCheckedChange={(value) => setField(key, value)} />
    </div>
  );

  const regenRow = (
    label: string,
    desc: string,
    layer: LayerKind,
    cost: CostKind,
    quality: QualityKind,
    key:
      | "temporalSub"
      | "rotate"
      | "hashEpsilon"
      | "requant"
      | "noise"
      | "dctStep"
      | "dwtDetail"
      | "nonintRatio"
      | "warp"
      | "perspective"
      | "jitter"
      | "flowDisturb"
      | "multiscale"
      | "facePerturb"
      | "temporalBlur"
      | "lpcAttack"
      | "copyAttack",
    onValue: number,
  ) => (
    <div className="switch">
      <div>
        <div className="switch-label">{label}</div>
        <div className="switch-desc">{desc}</div>
        <WeaponTags layer={layer} cost={cost} quality={quality} />
      </div>
      <Switch
        checked={(form[key] ?? 0) > 0}
        onCheckedChange={(value) => setField(key, value ? onValue : 0)}
      />
    </div>
  );

  const fftRow = (
    <div className="switch">
      <div>
        <div className="switch-label">FFT 扰动（相位+幅度）</div>
        <div className="switch-desc">打散中高频相位并随机缩放幅值</div>
        <WeaponTags layer="wm" cost="mid" quality="heavy" />
      </div>
      <Switch
        checked={(form.fftPhase ?? 0) > 0 || (form.fftMag ?? 0) > 0}
        onCheckedChange={(value) =>
          setForm((current) => ({
            ...current,
            fftPhase: value ? 0.5 : 0,
            fftMag: value ? 0.1 : 0,
          }))
        }
      />
    </div>
  );

  const textureRow = (
    <div className="switch">
      <div>
        <div className="switch-label">纹理/复杂度注入</div>
        <div className="switch-desc">向低纹理区注入纹理并拉平复杂度分布</div>
        <WeaponTags layer="fp" cost="slow" quality="heavy" />
      </div>
      <Switch
        checked={(form.textureInject ?? 0) > 0 || (form.complexityTrap ?? 0) > 0}
        onCheckedChange={(value) =>
          setForm((current) => ({
            ...current,
            textureInject: value ? 0.04 : 0,
            complexityTrap: value ? 0.1 : 0,
          }))
        }
      />
    </div>
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
                    <span>再生重写</span>
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
              {switchRow("同步处理音频指纹", "对音轨做等长频谱轻处理，不影响音画同步", "audio", "fast", "none", "audioRemix")}
              {switchRow("音频回声扰动", "同步放慢约 3% 并保音调，平台效果需实测", "audio", "fast", "mild", "echoDefeat")}
              {switchRow("强音频重混", "变调+EQ 倾斜+底噪，配合音频指纹开关生效", "audio", "fast", "heavy", "audioStrong")}
              {switchRow("重新构图（裁剪回缩）", "对暗水印无效，用于判重指纹；文字位置不固定，可能裁到字幕，仅确认边缘无文字时使用", "fp", "fast", "heavy", "recropOn")}

              {switchRow("画质保护（PSNR/SSIM 门控）", "对攻击/重武器层按目标自动回退（原生滤镜链除外）", "quality", "mid", "none", "qualityProtect")}
              {switchRow("锐度补偿", "抵消清除算法带来的轻微模糊", "quality", "fast", "none", "sharpness")}
              {switchRow("色彩还原", "把处理后亮度均值校准回原片统计", "quality", "fast", "none", "colorRestore")}
              {switchRow("空间降噪", "removegrain 轻降噪，破坏扩频/QIM/DWT 水印", "wm", "fast", "none", "denoise")}
              {switchRow("伪水印注入（溯源干扰）", "注入随机干扰水印，对抗上传后二次嵌入 · 需投流实测", "wm", "mid", "mild", "spoof")}
              {regenRow("几何微旋转", "低频小幅旋转去同步", "dual", "fast", "mild", "rotate", 0.4)}
              {switchRow("哈希签名对抗（pHash/dHash）", "签名域可微扰动，翻转哈希符号位", "fp", "mid", "heavy", "hashAttack")}
              {regenRow("局部平滑扭曲", "粗网格位移场上采样成平滑光流，破坏空间对齐", "wm", "mid", "mild", "warp", 0.005)}
              {regenRow("透视剪切", "逐帧轻微梯形畸变，破坏块对齐与几何同步", "wm", "mid", "mild", "perspective", 0.01)}
              {regenRow("平移抖动", "低频正弦漂移（相邻帧 ≤1px），破坏逐帧对齐", "wm", "fast", "mild", "jitter", 0.005)}
              <div className="field">
                <span className="field-label">哈希目标模式</span>
                <Select
                  value={form.hashMode}
                  onValueChange={(value) => setField("hashMode", value)}
                >
                  <SelectTrigger className="h-9 w-full"><SelectValue /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="phash">pHash 专攻（推荐）</SelectItem>
                    <SelectItem value="dhash">dHash 专攻</SelectItem>
                    <SelectItem value="joint">联合（pHash+dHash）</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              {regenRow("像素重量化", "把像素压缩到有限级数", "wm", "fast", "none", "requant", 64)}
              {regenRow("微噪声", "加性微高斯噪声", "wm", "fast", "none", "noise", 0.004)}
              {regenRow("DCT 系数扰动", "8×8 DCT 重量化并扰动中频系数", "wm", "mid", "mild", "dctStep", 12)}
              {regenRow("非整缩重采样", "先放大再回压，打散像素网格与块对齐", "dual", "fast", "mild", "nonintRatio", 0.01)}
              {regenRow("跨帧估计相减", "估计帧间固定水印并过减，会削弱静态字幕边缘", "wm", "mid", "heavy", "temporalSub", 0.6)}
              {switchRow("跨帧估计原生加速", "ffmpeg 原生链，更快且破坏力更强", "wm", "fast", "none", "nativeTemporal")}
              {fftRow}
              {regenRow("小波细节带随机化", "随机化 Haar 对角线细节子带", "wm", "fast", "none", "dwtDetail", 0.8)}
              {regenRow("光流一致性破坏", "按运动加权扰动重采样，改运动指纹", "fp", "fast", "heavy", "flowDisturb", 1.5)}
              {textureRow}
              {regenRow("多尺度特征扰动（DMFF）", "金字塔各尺度带限扰动", "fp", "slow", "heavy", "multiscale", 0.02)}
              {regenRow("人脸抗AI扰动", "仅在脸部区域注入扰动", "face", "mid", "none", "facePerturb", 0.04)}
              {regenRow("时序模糊", "帧间时域平滑", "dual", "mid", "heavy", "temporalBlur", 0.25)}
              {regenRow("LPCAA 音频攻击", "LPC 残差白化，破坏语音类指纹", "audio", "fast", "heavy", "lpcAttack", 0.5)}
              {regenRow("神经对抗（DINOv2 判重代理）", "黑盒攻击拷贝检测描述子", "fp", "slow", "heavy", "copyAttack", 0.05)}

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
