import { Play, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { enqueueJob } from "@/lib/backend";
import { useMaterialsStore } from "@/stores/materials";
import { useQueueStore } from "@/stores/queue";
import { toast } from "@/stores/toasts";

// PRD 3.3.1 要求的底部状态栏：全局任务统计、开始处理、特征库更新入口。
export default function StatusBar() {
  const materials = useMaterialsStore((state) => state.materials);
  const selected = useMaterialsStore((state) => state.selected);
  const activeId = useMaterialsStore((state) => state.activeId);
  const jobs = useQueueStore((state) => state.jobs);

  const tasks = jobs.flatMap((job) => job.tasks);
  const count = (status: string) => tasks.filter((task) => task.status === status).length;
  const active = materials.find((m) => m.id === activeId);

  const start = async () => {
    const selectedPaths = materials
      .filter((m) => selected[m.id] && m.path)
      .map((m) => ({ kind: "detect" as const, path: m.path as string }));
    const targets = selectedPaths.length
      ? selectedPaths
      : active?.path
        ? [{ kind: "detect" as const, path: active.path }]
        : [];
    if (!targets.length) {
      toast("请先通过本地路径导入或勾选素材");
      return;
    }
    try {
      await enqueueJob("开始处理", targets);
      toast(`已入队 ${targets.length} 个检测任务`);
    } catch {
      toast("入队失败，请确认引擎在线");
    }
  };

  return (
    <footer className="statusbar">
      <div className="statusbar-left">
        <span className="statusbar-item">素材 {materials.length}</span>
        <span className="statusbar-item">队列 {count("queued")}</span>
        <span className="statusbar-item">处理中 {count("running")}</span>
        <span className="statusbar-item">已完成 {count("done")}</span>
        <span className="statusbar-item">失败 {count("failed")}</span>
      </div>

      <div className="statusbar-right">
        <span className="statusbar-item">平台特征库 · 尚未接入</span>
        <Button variant="ghost" size="xs" disabled>
          <RefreshCw />
          检查更新
        </Button>
        <Button size="sm" onClick={start}>
          <Play />
          开始处理
        </Button>
      </div>
    </footer>
  );
}
