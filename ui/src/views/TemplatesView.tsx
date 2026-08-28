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
  createTemplate,
  deleteTemplate,
  listTemplates,
  updateTemplate,
  type TemplateInfo,
} from "@/lib/backend";
import {
  DEFAULT_TEMPLATE,
  payloadOf,
  type AntiLevel,
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
              {switchRow("同步处理音频指纹", "对音轨做等长频谱轻处理，不影响音画同步", "audioRemix")}
              {switchRow("音频回声扰动", "同步放慢约 3% 并保音调，平台效果需实测", "echoDefeat")}
              {switchRow("抗二次检测增强", "扰动中频 DCT 系数，破坏二次嵌入", "antiReembed")}
              {switchRow("重新构图（裁剪回缩）", "对抗内容指纹，静态缩放微模糊", "recropOn")}
              {switchRow("细节保护（人脸/字幕/纹理）", "保护区域回退原帧，保留部分水印特征", "detailProtectOn")}

              <div className="field">
                <span className="field-label">指纹对抗强度</span>
                <Select value={form.anti} onValueChange={(value) => setField("anti", value as AntiLevel)}>
                  <SelectTrigger className="h-9 w-full"><SelectValue /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="关闭">关闭 · 仅基础清洗，判重风险高</SelectItem>
                    <SelectItem value="轻度">轻度 · 低成本近无损</SelectItem>
                    <SelectItem value="标准">标准 · 均衡，哈希层打满</SelectItem>
                    <SelectItem value="强力">强力 · 轻微模糊，可能有轻微闪烁</SelectItem>
                    <SelectItem value="全兵器">全兵器 · 研究用，明显伪影</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {switchRow("锐度补偿", "抵消清除算法带来的轻微模糊", "sharpness")}
              {switchRow("色彩还原", "把处理后亮度均值校准回原片统计", "colorRestore")}
              {switchRow("空间降噪", "破坏空域扩频水印", "denoise")}
              {switchRow("伪水印注入（溯源干扰）", "注入随机干扰水印，轻微损失画质", "spoof")}

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
              {switchRow("无损输出", "极致保留画质，文件体积增大", "lossless")}

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
