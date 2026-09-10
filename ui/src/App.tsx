import { useEffect, useState } from "react";
import Rail from "@/components/Rail";
import TopBar from "@/components/TopBar";
import Toaster from "@/components/Toaster";
import ImportDialog from "@/components/ImportDialog";
import StatusBar from "@/components/StatusBar";
import JobsView from "@/views/JobsView";
import ProductsView from "@/views/ProductsView";
import SettingsView from "@/views/SettingsView";
import TemplatesView from "@/views/TemplatesView";
import Workbench from "@/views/Workbench";
import { TooltipProvider } from "@/components/ui/tooltip";
import { connectEvents, getHealth } from "@/lib/backend";
import { useAppStore } from "@/stores/app";
import { useQueueStore } from "@/stores/queue";
import { useMaterialsStore } from "@/stores/materials";
import { toast } from "@/stores/toasts";

export default function App() {
  const view = useAppStore((state) => state.view);
  const themePref = useAppStore((state) => state.themePref);
  const setBackendConnected = useAppStore((state) => state.setBackendConnected);

  const [systemDark, setSystemDark] = useState(
    () => window.matchMedia("(prefers-color-scheme: dark)").matches,
  );

  useEffect(() => {
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const onChange = (e: MediaQueryListEvent) => setSystemDark(e.matches);
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, []);

  const effective = themePref === "system" ? (systemDark ? "dark" : "light") : themePref;

  useEffect(() => {
    document.documentElement.classList.toggle("dark", effective === "dark");
    localStorage.setItem("theme", themePref);
  }, [effective, themePref]);

  useEffect(() => {
    return connectEvents((msg) => {
      if (msg.type === "open") {
        setBackendConnected(true);
        // 断线重连后补齐错过的任务事件，避免任务列表停留在旧状态。
        void useQueueStore.getState().refresh();
      }
      if (msg.type === "close") setBackendConnected(false);
      useQueueStore.getState().applyEvent(msg);
    });
  }, [setBackendConnected]);

  // 事件推送可能因断线重连窗口漏掉终态；只要还有活动任务，
  // 每 5 秒向服务端校准一次队列，保证按钮与任务列表最终一致。
  useEffect(() => {
    let wasActive = false;
    const timer = window.setInterval(() => {
      const { jobs } = useQueueStore.getState();
      const hasActive = jobs.some((job) =>
        job.tasks.some(
          (task) =>
            task.status === "queued" ||
            task.status === "running" ||
            task.status === "paused",
        ),
      );
      if (hasActive) {
        wasActive = true;
        void useQueueStore.getState().refresh();
      } else if (wasActive) {
        // 活动任务清零后最后校准一次，兜住断线期间错过的终态事件。
        wasActive = false;
        void useQueueStore.getState().refresh();
      }
    }, 5000);
    return () => window.clearInterval(timer);
  }, []);

  // 启动时等待引擎就绪后加载真实素材库；失败时由工作台展示错误空状态。
  useEffect(() => {
    let cancelled = false;
    (async () => {
      for (let attempt = 0; attempt < 90; attempt++) {
        if (cancelled) return;
        try {
          const health = await getHealth(2000);
          if (health.ok) {
            await useMaterialsStore.getState().refreshLibrary();
            if (!localStorage.getItem("onboarding-seen")) {
              localStorage.setItem("onboarding-seen", "1");
              const loaded = useMaterialsStore.getState().materials.length;
              if (loaded) {
                toast(
                  "已恢复素材库",
                  "上次导入的素材已就绪，继续清洗即可",
                );
              } else {
                toast(
                  "欢迎使用 Cthulhu",
                  "拖入或导入你自己的视频，即可开始清洗",
                );
              }
            }
            return;
          }
        } catch {
          // 引擎尚未就绪，继续等待
        }
        await new Promise((resolve) => setTimeout(resolve, 1000));
      }
      useMaterialsStore.getState().markLoadFailed();
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  // 启动时检查 FFmpeg；缺失时提前告知，避免用户操作后才遇到失败。
  useEffect(() => {
    void getHealth().then((health) => {
      if (health.ok && !health.ffmpeg) {
        toast(
          "未检测到 FFmpeg",
          "清洗与修复功能将不可用，请在「设置」查看安装指引",
        );
      }
    });
  }, []);

  return (
    <TooltipProvider>
      <div className="app">
        <TopBar />
        <div className="layout">
          <Rail />
          <main className="view">
            {view === "workbench" && <Workbench />}
            {view === "jobs" && <JobsView />}
            {view === "templates" && <TemplatesView />}
            {view === "products" && <ProductsView />}
            {view === "settings" && <SettingsView />}
          </main>
        </div>
        <StatusBar />
      </div>
      <ImportDialog />
      <Toaster />
    </TooltipProvider>
  );
}
