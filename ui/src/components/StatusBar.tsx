import { useMaterialsStore } from "@/stores/materials";
import { useQueueStore } from "@/stores/queue";

export default function StatusBar() {
  const materials = useMaterialsStore((state) => state.materials);
  const jobs = useQueueStore((state) => state.jobs);

  const tasks = jobs.flatMap((job) => job.tasks);
  const count = (status: string) => tasks.filter((task) => task.status === status).length;

  return (
    <footer className="statusbar">
      <div className="statusbar-left">
        <span className="statusbar-item">素材 {materials.length}</span>
        <span className="statusbar-item">队列 {count("queued")}</span>
        <span className="statusbar-item">处理中 {count("running")}</span>
        <span className="statusbar-item">已完成 {count("done")}</span>
        <span className="statusbar-item">失败 {count("failed")}</span>
      </div>

    </footer>
  );
}
