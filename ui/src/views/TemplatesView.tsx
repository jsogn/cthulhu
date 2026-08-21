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

interface TemplatePayload {
  level: CleanLevel;
  restruct: number;
  perturb: number;
  denoise: boolean;
  audio: boolean;
  codec: string;
}

const LEVEL_PRESETS: Record<CleanLevel, { restruct: number; perturb: number }> = {
  轻度: { restruct: 25, perturb: 15 },
  平衡: { restruct: 30, perturb: 20 },
  深度: { restruct: 40, perturb: 30 },
};

function payloadOf(template: TemplateInfo): TemplatePayload {
  const payload = (template.payload ?? {}) as Partial<TemplatePayload>;
  return {
    level: payload.level ?? "平衡",
    restruct: payload.restruct ?? 30,
    perturb: payload.perturb ?? 20,
    denoise: payload.denoise ?? true,
    audio: payload.audio ?? true,
    codec: payload.codec ?? "H.264",
  };
}

export default function TemplatesView() {
  const [templates, setTemplates] = useState<TemplateInfo[]>([]);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [name, setName] = useState("");
  const [level, setLevel] = useState<CleanLevel>("平衡");
  const [restruct, setRestruct] = useState(30);
  const [perturb, setPerturb] = useState(20);
  const [denoise, setDenoise] = useState(true);
  const [audio, setAudio] = useState(true);
  const [codec, setCodec] = useState("H.264");
  const [editing, setEditing] = useState<TemplateInfo | null>(null);
  const [pendingDelete, setPendingDelete] = useState<TemplateInfo | null>(null);

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
    const payload: TemplatePayload = {
      level,
      restruct,
      perturb,
      denoise,
      audio,
      codec,
    };
    try {
      const finalName = name.trim() || `${level}档模板`;
      if (editing) {
        await updateTemplate(editing.id, finalName, { ...payload });
      } else {
        await createTemplate(finalName, { ...payload });
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
    setLevel("平衡");
    setRestruct(30);
    setPerturb(20);
    setDenoise(true);
    setAudio(true);
    setCodec("H.264");
    setDialogOpen(true);
  };

  const openEdit = (template: TemplateInfo) => {
    const payload = payloadOf(template);
    setEditing(template);
    setName(template.name);
    setLevel(payload.level);
    setRestruct(payload.restruct);
    setPerturb(payload.perturb);
    setDenoise(payload.denoise);
    setAudio(payload.audio);
    setCodec(payload.codec);
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

  return (
    <>
      <div className="view-head">
        <div>
          <div className="view-title">模板管理</div>
          <div className="view-desc">保存常用清洗参数（持久化保存）；套用请到素材处理的批量操作条</div>
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
                  <div className="tpl-row"><span>变速幅度</span><b>{payload.restruct}%</b></div>
                  <div className="tpl-row"><span>微扰强度</span><b>{payload.perturb}%</b></div>
                  <div className="tpl-row">
                    <span>空间降噪 / 音频重混</span>
                    <b>{payload.denoise ? "开" : "关"} / {payload.audio ? "开" : "关"}</b>
                  </div>
                  <div className="tpl-row"><span>输出编码</span><b>{payload.codec}</b></div>
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
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>{editing ? "编辑清洗模板" : "新建清洗模板"}</DialogTitle>
          </DialogHeader>
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
                value={level}
                onValueChange={(v) => {
                  const next = v as CleanLevel;
                  setLevel(next);
                  setRestruct(LEVEL_PRESETS[next].restruct);
                  setPerturb(LEVEL_PRESETS[next].perturb);
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
                变速幅度 <span className="field-value">{restruct}%</span>
              </span>
              <Slider value={[restruct]} min={10} max={70} onValueChange={([v]) => setRestruct(v)} />
            </div>
            <div className="field">
              <span className="field-label">
                微扰强度 <span className="field-value">{perturb}%</span>
              </span>
              <Slider value={[perturb]} min={5} max={50} onValueChange={([v]) => setPerturb(v)} />
            </div>
            <div className="field">
              <Label>输出编码</Label>
              <Select value={codec} onValueChange={setCodec}>
                <SelectTrigger className="h-9"><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="H.264">H.264</SelectItem>
                  <SelectItem value="H.265">H.265</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="switch">
              <div>
                <div className="switch-label">空间降噪</div>
                <div className="switch-desc">有效破坏空域扩频与小波域水印</div>
              </div>
              <Switch checked={denoise} onCheckedChange={setDenoise} />
            </div>
            <div className="switch">
              <div>
                <div className="switch-label">音频重混</div>
                <div className="switch-desc">破坏回声隐藏水印</div>
              </div>
              <Switch checked={audio} onCheckedChange={setAudio} />
            </div>
          </div>
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
