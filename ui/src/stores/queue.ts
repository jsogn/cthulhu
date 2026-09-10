import { create } from "zustand";
import {
  cancelJob,
  listJobs,
  pauseJob,
  resumeJob,
  retryJob,
  setJobPriority,
  type BackendEvent,
  type JobInfo,
  type JobTaskInfo,
} from "@/lib/backend";
import { toast } from "@/stores/toasts";

interface QueueState {
  jobs: JobInfo[];
  refresh: () => Promise<void>;
  applyEvent: (event: BackendEvent) => void;
  cancel: (jobId: string) => Promise<void>;
  pause: (jobId: string) => Promise<void>;
  resume: (jobId: string) => Promise<void>;
  setPriority: (jobId: string, priority: number) => Promise<void>;
  retry: (jobId: string) => Promise<void>;
}

export const useQueueStore = create<QueueState>((set) => ({
  jobs: [],

  refresh: async () => {
    try {
      set({ jobs: await listJobs() });
    } catch {
      // 后端未就绪时保持现状
    }
  },

  applyEvent: (event) => {
    if (event.type === "job:state" && event.job) {
      const job = event.job as JobInfo;
      set((state) => ({
        jobs: [job, ...state.jobs.filter((item) => item.id !== job.id)].sort(
          (a, b) => (b.created_at ?? 0) - (a.created_at ?? 0),
        ),
      }));
      if (job.status === "done") {
        toast(`任务完成：${job.name}`);
        try {
          new Notification(`Cthulhu · 任务完成`, { body: job.name });
        } catch {
          // 环境不支持系统通知时静默降级
        }
      }
      if (job.status === "failed") toast(`任务失败：${job.name}`);
    }
    if (event.type === "task:state" && event.task) {
      const jobId = event.job_id as string;
      const task = event.task as JobTaskInfo;
      set((state) => ({
        jobs: state.jobs.map((job) =>
          job.id === jobId
            ? { ...job, tasks: job.tasks.map((t) => (t.id === task.id ? task : t)) }
            : job,
        ),
      }));
    }
  },

  cancel: async (jobId) => {
    const job = await cancelJob(jobId);
    set((state) => ({ jobs: state.jobs.map((item) => (item.id === job.id ? job : item)) }));
    toast("已请求取消任务");
  },

  pause: async (jobId) => {
    const job = await pauseJob(jobId);
    set((state) => ({ jobs: state.jobs.map((item) => (item.id === job.id ? job : item)) }));
    toast(`已请求暂停：${job.name}`, "运行中的处理项会尽快停住，完成后不再分发新任务");
  },

  resume: async (jobId) => {
    const job = await resumeJob(jobId);
    set((state) => ({ jobs: state.jobs.map((item) => (item.id === job.id ? job : item)) }));
    toast(`已继续任务：${job.name}`);
  },

  setPriority: async (jobId, priority) => {
    const job = await setJobPriority(jobId, priority);
    set((state) => ({ jobs: state.jobs.map((item) => (item.id === job.id ? job : item)) }));
    toast(`已调整优先级：${job.name}`);
  },

  retry: async (jobId) => {
    const job = await retryJob(jobId);
    set((state) => ({
      jobs: [job, ...state.jobs.filter((item) => item.id !== job.id)].sort(
        (a, b) => (b.created_at ?? 0) - (a.created_at ?? 0),
      ),
    }));
    toast(`失败项已重新入队：${job.name}`);
  },
}));
