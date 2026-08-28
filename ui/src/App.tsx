import { lazy, Suspense, useEffect, useState } from "react";
import Rail from "@/components/Rail";
import TopBar from "@/components/TopBar";
import Toaster from "@/components/Toaster";
import ImportDialog from "@/components/ImportDialog";
import StatusBar from "@/components/StatusBar";
import Workbench from "@/views/Workbench";
import { TooltipProvider } from "@/components/ui/tooltip";
import { connectEvents, getHealth } from "@/lib/backend";
import { useAppStore } from "@/stores/app";
import { useQueueStore } from "@/stores/queue";
import { useMaterialsStore } from "@/stores/materials";
import { toast } from "@/stores/toasts";

// 非首屏视图按需加载：任务中心 / 模板 / 产物 / 设置只在切到对应页时下载。
const JobsView = lazy(() => import("@/views/JobsView"));
const TemplatesView = lazy(() => import("@/views/TemplatesView"));
const ProductsView = lazy(() => import("@/views/ProductsView"));
const SettingsView = lazy(() => import("@/views/SettingsView"));

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

  // 首屏稳定后闲时预热其余视图：启动仍保持代码分割，
  // 预热完成后点菜单即为零等待切换，不再出现懒加载占位闪烁。
  useEffect(() => {
    const timer = window.setTimeout(() => {
      void Promise.all([
        import("@/views/JobsView"),
        import("@/views/TemplatesView"),
        import("@/views/ProductsView"),
        import("@/views/SettingsView"),
      ]);
    }, 500);
    return () => window.clearTimeout(timer);
  }, []);

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
      if (hasActive) void useQueueStore.getState().refresh();
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
                  "上次导入的素材已就绪，继续检测或清洗即可",
                );
              } else {
                toast(
                  "欢迎使用 Cthulhu",
                  "拖入或导入你自己的视频，即可开始检测与清洗",
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
          "检测、清洗与修复功能将不可用，请在「设置」查看安装指引",
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
            <Suspense
              fallback={
                <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
                  加载中…
                </div>
              }
            >
              {view === "workbench" && <Workbench />}
              {view === "jobs" && <JobsView />}
              {view === "templates" && <TemplatesView />}
              {view === "products" && <ProductsView />}
              {view === "settings" && <SettingsView />}
            </Suspense>
          </main>
        </div>
        <StatusBar />
      </div>
      <ImportDialog />
      <Toaster />
    </TooltipProvider>
  );
}
