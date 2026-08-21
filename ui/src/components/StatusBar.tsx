import { useEffect, useState } from "react";
import { getHealth, type HealthInfo } from "@/lib/backend";
import { useMaterialsStore } from "@/stores/materials";
import { useQueueStore } from "@/stores/queue";

export default function StatusBar() {
  const materials = useMaterialsStore((state) => state.materials);
  const jobs = useQueueStore((state) => state.jobs);
  const [host, setHost] = useState<HealthInfo["host"] | null>(null);

  const tasks = jobs.flatMap((job) => job.tasks);
  const count = (status: string) => tasks.filter((task) => task.status === status).length;

  useEffect(() => {
    void getHealth().then((health) => {
      if (health.ok) setHost(health.host ?? null);
    });
  }, []);

  const hostLabel = host
    ? [
        host.os === "Darwin" ? "macOS" : host.os === "Windows" ? "Windows" : host.os,
        host.arch,
        `${host.cpu_count} 核`,
        host.memory_gb != null ? `${host.memory_gb}GB 内存` : null,
      ]
        .filter(Boolean)
        .join(" · ")
    : "";

  return (
    <footer className="statusbar">
      <div className="statusbar-left">
        {hostLabel && <span className="statusbar-item mono">{hostLabel}</span>}
      </div>
      <div className="statusbar-right">
        <span className="statusbar-item">素材 {materials.length}</span>
        <span className="statusbar-item">队列 {count("queued")}</span>
        <span className="statusbar-item">处理中 {count("running")}</span>
        <span className="statusbar-item">已完成 {count("done")}</span>
        <span className="statusbar-item">失败 {count("failed")}</span>
      </div>

    </footer>
  );
}
