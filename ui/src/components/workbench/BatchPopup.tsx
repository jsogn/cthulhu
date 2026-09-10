import { useState } from "react";
import { X } from "lucide-react";
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
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { cleanOutputPath, collusionOutputPath, makeCleanOptions } from "@/lib/cleanOptions";
import { enqueueJob, exportOutputs, runCollusion } from "@/lib/backend";
import { useCleanPanel } from "@/stores/cleanPanel";
import { useMaterialsStore, type Material } from "@/stores/materials";
import { toast } from "@/stores/toasts";

export function BatchPopup() {
  const selected = useMaterialsStore((state) => state.selected);
  const materials = useMaterialsStore((state) => state.materials);
  const deleteSelected = useMaterialsStore((state) => state.deleteSelected);
  const toggleSelectAll = useMaterialsStore((state) => state.toggleSelectAll);
  const cleanPanel = useCleanPanel();
  const [batchOpen, setBatchOpen] = useState(false);
  const [batchSubmitting, setBatchSubmitting] = useState(false);
  const [collusionOpen, setCollusionOpen] = useState(false);
  const [collusionSubmitting, setCollusionSubmitting] = useState(false);
  const [collusionMode, setCollusionMode] = useState<"mean" | "median">("mean");
  const ids = Object.keys(selected);

  if (!ids.length) return null;

  const targets = ids
    .map((id) => materials.find((m) => m.id === id))
    .filter(
      (m): m is Material & { path: string } =>
        !!m && !!m.path && !m.missing,
    );
  const skipped = ids.length - targets.length;

  const batchExport = async () => {
    const paths = ids
      .map((id) => materials.find((m) => m.id === id))
      .filter((m) => !!m?.path)
      .map((m) => m!.path as string);
    if (!paths.length) {
      toast("所选素材缺少本地路径，无法导出");
      return;
    }
    const picker = window.appEnv?.chooseFolder;
    if (!picker) {
      toast("网页版不支持导出目录，请使用桌面版");
      return;
    }
    const chosen = await picker();
    if (!chosen) return;
    try {
      const { exported, missing } = await exportOutputs(paths, chosen);
      const parts: string[] = [];
      if (exported.length) parts.push(`已导出 ${exported.length} 个产物`);
      if (missing.length) parts.push(`${missing.length} 个素材尚无产物`);
      toast(parts.length ? parts.join(" · ") : "所选素材均未处理，请先清洗");
    } catch (error) {
      toast(error instanceof Error ? error.message : "导出失败");
    }
  };

  const remove = () => {
    deleteSelected(ids);
    toast(`已移除 ${ids.length} 个素材`);
  };

  const runBatchClean = async () => {
    if (!targets.length || batchSubmitting) return;
    setBatchSubmitting(true);
    try {
      const now = new Date();
      const pad = (value: number) => String(value).padStart(2, "0");
      const ts = `${pad(now.getMonth() + 1)}${pad(now.getDate())}_${pad(
        now.getHours(),
      )}_${pad(now.getMinutes())}_${pad(now.getSeconds())}`;
      const tasks = targets.map((target, index) => ({
        kind: "desensitize" as const,
        path: target.path,
        options: {
          output: cleanOutputPath(
            target.path,
            cleanPanel.settingsNaming,
            cleanPanel.settingsExportDir,
            ts,
            targets.length > 1 ? index + 1 : undefined,
          ),
          ...makeCleanOptions(cleanPanel),
        },
      }));
      await enqueueJob(`批量清洗 ${targets.length} 个素材`, tasks);
      toggleSelectAll(ids);
      setBatchOpen(false);
      toast(
        `已加入批量清洗，共 ${targets.length} 个任务${
          skipped > 0 ? `，跳过 ${skipped} 个（缺少本地路径）` : ""
        }`,
      );
    } catch (error) {
      toast(error instanceof Error ? error.message : "入队失败");
    } finally {
      setBatchSubmitting(false);
    }
  };

  const runCollusionAverage = async () => {
    if (targets.length < 2 || collusionSubmitting) return;
    setCollusionSubmitting(true);
    try {
      const now = new Date();
      const pad = (value: number) => String(value).padStart(2, "0");
      const ts = `${pad(now.getMonth() + 1)}${pad(now.getDate())}_${pad(
        now.getHours(),
      )}_${pad(now.getMinutes())}_${pad(now.getSeconds())}`;
      const output = collusionOutputPath(
        targets[0].path,
        cleanPanel.settingsExportDir,
        ts,
      );
      const result = await runCollusion(
        targets.map((target) => target.path),
        output,
        collusionMode,
      );
      toast(
        `共谋平均完成：${result.copies} 副本 · ${result.frames} 帧 · 预计残余 ${(
          result.estimated_watermark_reduction * 100
        ).toFixed(0)}%`,
      );
      setCollusionOpen(false);
    } catch (error) {
      toast(error instanceof Error ? error.message : "共谋平均失败");
    } finally {
      setCollusionSubmitting(false);
    }
  };

  return (
    <>
      <div className="batch-popup" role="dialog" aria-label="批量操作">
        <span className="batch-count">已选 {ids.length} 个</span>
        <Button size="sm" disabled={!targets.length} onClick={() => setBatchOpen(true)}>
          批量清洗（{targets.length}）
        </Button>
        {targets.length >= 2 ? (
          <Button
            variant="secondary"
            size="sm"
            onClick={() => setCollusionOpen(true)}
          >
            共谋平均（{targets.length}）
          </Button>
        ) : null}
        {window.appEnv?.chooseFolder ? (
          <Button variant="secondary" size="sm" onClick={() => void batchExport()}>
            导出产物
          </Button>
        ) : null}
        <Button variant="ghost" size="sm" className="text-destructive" onClick={remove}>
          移除
        </Button>
        <Button
          variant="ghost"
          size="icon"
          className="h-7 w-7"
          aria-label="关闭"
          onClick={() => toggleSelectAll(ids)}
        >
          <X />
        </Button>
      </div>
      <AlertDialog open={batchOpen} onOpenChange={setBatchOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>批量清洗 {targets.length} 个素材</AlertDialogTitle>
            <AlertDialogDescription>
              将使用当前清洗设置（重新编码 · 当前面板参数）清洗所选素材，
              开始后可在任务中心查看进度。
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>取消</AlertDialogCancel>
            <AlertDialogAction
              disabled={batchSubmitting}
              onClick={() => void runBatchClean()}
            >
              {batchSubmitting ? "入队中…" : "开始清洗"}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
      <AlertDialog open={collusionOpen} onOpenChange={setCollusionOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>共谋平均 {targets.length} 个副本</AlertDialogTitle>
            <AlertDialogDescription>
              仅对同一内容、不同水印的副本有效；分辨率不同会缩放到第一个副本，帧数不同会截断到最短。
              报告证据：8/16/32 副本 → BA 0.599/0.575/0.542。
            </AlertDialogDescription>
          </AlertDialogHeader>
          <div className="field">
            <span className="field-label">平均方式</span>
            <Select
              value={collusionMode}
              onValueChange={(value) => setCollusionMode(value as "mean" | "median")}
            >
              <SelectTrigger className="form-input h-8 w-full">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="mean">算术平均（报告口径）</SelectItem>
                <SelectItem value="median">中位数（更抗对齐误差）</SelectItem>
              </SelectContent>
            </Select>
          </div>
          {targets[0] ? (
            <div className="text-xs text-muted-foreground">
              输出：{collusionOutputPath(targets[0].path, cleanPanel.settingsExportDir, "时间戳")}
            </div>
          ) : null}
          <AlertDialogFooter>
            <AlertDialogCancel disabled={collusionSubmitting}>取消</AlertDialogCancel>
            <AlertDialogAction
              disabled={collusionSubmitting}
              onClick={(event) => {
                event.preventDefault();
                void runCollusionAverage();
              }}
            >
              {collusionSubmitting ? "处理中…" : "开始共谋平均"}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  );
}
