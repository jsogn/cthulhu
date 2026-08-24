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
import { Progress } from "@/components/ui/progress";
import { clearJobs, type JobInfo } from "@/lib/backend";
import { useQueueStore } from "@/stores/queue";
import { toast } from "@/stores/toasts";

const KIND_LABEL: Record<string, string> = {
  detect: "暗水印检测",
  desensitize: "清洗去重",
  repair: "可见水印修复",
};

const STATUS_LABEL: Record<string, string> = {
  queued: "队列中",
  running: "处理中",
  paused: "已暂停",
  done: "已完成",
  failed: "失败",
  canceled: "已取消",
};

function basename(path: string): string {
  return path.split(/[\\/]/).pop() || path;
}

function formatTime(timestamp: number | undefined): string {
  if (!timestamp) return "";
  return new Date(timestamp * 1000).toLocaleString();
}

export default function JobsView() {
  const jobs = useQueueStore((state) => state.jobs);
  const refresh = useQueueStore((state) => state.refresh);
  const cancel = useQueueStore((state) => state.cancel);
  const pause = useQueueStore((state) => state.pause);
  const resume = useQueueStore((state) => state.resume);
  const setPriority = useQueueStore((state) => state.setPriority);
  const retry = useQueueStore((state) => state.retry);
  const [pendingClear, setPendingClear] = useState<"finished" | "all" | null>(null);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const tasks = jobs.flatMap((job) => job.tasks);
  const count = (status: string) => tasks.filter((task) => task.status === status).length;
  const finishedCount = count("done") + count("failed");

  const confirmClear = async () => {
    if (!pendingClear) return;
    try {
      const removed = await clearJobs(pendingClear);
      await refresh();
      toast(`已清空 ${removed} 个任务`);
    } catch {
      toast("清空失败，请确认引擎在线");
    } finally {
      setPendingClear(null);
    }
  };

  return (
    <>
      <div className="view-head">
        <div>
          <div className="view-title">任务中心</div>
          <div className="view-desc">所有检测、清洗与修复任务都在这里统一管理</div>
        </div>
        <div className="flex gap-2">
          <Button
            variant="ghost"
            disabled={!finishedCount}
            onClick={() => setPendingClear("finished")}
          >
            清空已完成
          </Button>
          <Button variant="ghost" disabled={!jobs.length} onClick={() => setPendingClear("all")}>
            清空全部
          </Button>
          <Button variant="ghost" onClick={refresh}>
            刷新
          </Button>
        </div>
      </div>

      <div className="view-body">
        <div className="stats">
          <Card className="stat s-队列中 p-3.5">
            <div className="stat-num">{count("queued")}</div>
            <div className="stat-label">队列中</div>
          </Card>
          <Card className="stat s-处理中 p-3.5">
            <div className="stat-num">{count("running")}</div>
            <div className="stat-label">处理中</div>
          </Card>
          <Card className="stat s-成功 p-3.5">
            <div className="stat-num">{count("done")}</div>
            <div className="stat-label">已完成</div>
          </Card>
          <Card className="stat s-失败 p-3.5">
            <div className="stat-num">{count("failed")}</div>
            <div className="stat-label">失败</div>
          </Card>
        </div>

        {jobs.length === 0 ? (
          <Card className="p-8 text-center text-muted-foreground">
            暂无任务，回到素材处理选择素材并加入队列。
          </Card>
        ) : (
          jobs.map((job) => (
            <JobCard
              key={job.id}
              job={job}
              onCancel={cancel}
              onPause={pause}
              onResume={resume}
              onPriority={setPriority}
              onRetry={retry}
            />
          ))
        )}
      </div>

      <AlertDialog open={!!pendingClear} onOpenChange={(open) => !open && setPendingClear(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>确认清空任务</AlertDialogTitle>
            <AlertDialogDescription>
              {pendingClear === "all"
                ? "将删除全部任务记录，包括进行中的任务。此操作不可撤销。"
                : "将删除所有已完成与失败的任务记录。此操作不可撤销。"}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>取消</AlertDialogCancel>
            <AlertDialogAction onClick={confirmClear}>确认清空</AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  );
}

function JobCard({
  job,
  onCancel,
  onPause,
  onResume,
  onPriority,
  onRetry,
}: {
  job: JobInfo;
  onCancel: (id: string) => void;
  onPause: (id: string) => void;
  onResume: (id: string) => void;
  onPriority: (id: string, priority: number) => void;
  onRetry: (id: string) => void;
}) {
  const failedCount = job.tasks.filter((task) => task.status === "failed").length;

  const renderTaskDetail = (task: JobInfo["tasks"][number]) => (
    <>
      {task.status === "running" && (
        <>
          <span className="text-xs text-muted-foreground">
            {task.progress_note ?? "处理中"}
            {task.elapsed != null ? ` · 已运行 ${task.elapsed}s` : ""}
          </span>
          {task.percent === 0 ? (
            <div className="progress-track h-1 w-full overflow-hidden rounded-full bg-muted">
              <div className="progress-indeterminate h-full w-1/3 rounded-full bg-primary" />
            </div>
          ) : (
            <Progress value={task.percent} className="h-1" />
          )}
        </>
      )}
      {task.error && <span className="text-xs text-destructive">失败：{task.error}</span>}
      {task.status === "done" && task.kind === "desensitize" && (
        <>
          <span className="mono truncate text-xs text-muted-foreground">
            输出：{(task.result as { output?: string } | null)?.output ?? "—"}
          </span>
          {(() => {
            const residual = (task.result as {
              residual?: { ss: number | null } | null;
            } | null)?.residual;
            return residual?.ss != null ? (
              <span className="text-xs text-muted-foreground">
                空间水印残留 {residual.ss.toFixed(2)}（干净基线约 0.28）
              </span>
            ) : null;
          })()}
          {(() => {
            const meta = task.result as {
              transform_strategy?: string;
              preset?: string;
            } | null;
            if (!meta?.transform_strategy) return null;
            const strategyLabel = meta.transform_strategy === "fast" ? "快速" : "完整";
            return (
              <span className="text-xs text-muted-foreground">
                管线：{strategyLabel} · 编码 {meta.preset ?? "medium"}
              </span>
            );
          })()}
        </>
      )}
    </>
  );

  return (
    <div className="job-row">
      <div className="flex items-center gap-2">
        <span className="job-name min-w-0 truncate">{job.name}</span>
        <span className={`job-status ${STATUS_LABEL[job.status]}`}>{STATUS_LABEL[job.status]}</span>
      </div>
      <div className="text-xs text-muted-foreground">
        {formatTime(job.created_at)}
      </div>

      {job.tasks.length === 1 ? (
        <div className="flex flex-col gap-1">{renderTaskDetail(job.tasks[0])}</div>
      ) : (
        <div className="flex flex-col gap-1">
          {job.tasks.map((task) => (
            <div key={task.id} className="flex flex-col gap-0.5">
              <div className="flex items-center gap-2 text-xs">
                <span className="shrink-0 text-muted-foreground">{KIND_LABEL[task.kind]}</span>
                <span className="mono min-w-0 flex-1 truncate">{basename(task.path)}</span>
                <span className={`job-status ${STATUS_LABEL[task.status]}`}>
                  {STATUS_LABEL[task.status]}
                </span>
              </div>
              {renderTaskDetail(task)}
            </div>
          ))}
        </div>
      )}

      {(job.status === "queued" ||
        job.status === "running" ||
        job.status === "paused" ||
        failedCount > 0) && (
        <div className="job-actions justify-end">
          {job.status === "queued" && (
            <Button
              variant="secondary"
              size="sm"
              onClick={() => onPriority(job.id, Math.max(0, job.priority - 1))}
            >
              提前执行
            </Button>
          )}
          {job.status === "paused" ? (
            <Button variant="secondary" size="sm" onClick={() => onResume(job.id)}>
              继续
            </Button>
          ) : (
            (job.status === "queued" || job.status === "running") && (
              <Button variant="secondary" size="sm" onClick={() => onPause(job.id)}>
                暂停
              </Button>
            )
          )}
          {(job.status === "queued" || job.status === "running" || job.status === "paused") && (
            <Button variant="secondary" size="sm" onClick={() => onCancel(job.id)}>
              取消
            </Button>
          )}
          {failedCount > 0 && (
            <Button variant="secondary" size="sm" onClick={() => onRetry(job.id)}>
              重试
            </Button>
          )}
        </div>
      )}
    </div>
  );
}
