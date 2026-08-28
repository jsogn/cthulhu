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
import { cleanOutputPath, makeCleanOptions } from "@/lib/cleanOptions";
import { enqueueJob, exportOutputs } from "@/lib/backend";
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

  return (
    <>
      <div className="batch-popup" role="dialog" aria-label="批量操作">
        <span className="batch-count">已选 {ids.length} 个</span>
        <Button size="sm" disabled={!targets.length} onClick={() => setBatchOpen(true)}>
          批量清洗（{targets.length}）
        </Button>
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
              将使用当前清洗设置（
              {cleanPanel.outputMode === "remux"
                ? "重新封装"
                : `重新编码 · 指纹对抗 ${cleanPanel.antiLevel}`}
              ）清洗所选素材，开始后可在任务中心查看进度。
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
    </>
  );
}
