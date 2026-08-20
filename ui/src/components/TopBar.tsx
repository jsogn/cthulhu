import { Moon, Settings, Sun, Upload } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { resolveTheme, useAppStore } from "@/stores/app";

export default function TopBar() {
  const themePref = useAppStore((state) => state.themePref);
  const setThemePref = useAppStore((state) => state.setThemePref);
  const connected = useAppStore((state) => state.backendConnected);
  const setImportOpen = useAppStore((state) => state.setImportOpen);
  const setView = useAppStore((state) => state.setView);

  const effective = resolveTheme(themePref);

  return (
    <header className="topbar">
      <div className="brand">
        <span className="brand-mark" aria-hidden="true">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <rect x="2" y="4" width="20" height="16" rx="3" />
            <path d="M10 9l5 3-5 3z" />
          </svg>
        </span>
        <div>
          <div className="brand-name">暗水印清洗台</div>
          <div className="brand-sub">暗水印对抗 · 素材去重风险清洗 · 可见水印修复</div>
        </div>
      </div>

      <div className="topbar-actions">
        <Button onClick={() => setImportOpen(true)}>
          <Upload />
          导入素材
        </Button>
        <span className={`status${connected ? " ok" : " err"}`} title={connected ? "后端已连接" : "后端未连接"}>
          <span className="status-dot" />
          {connected ? "引擎就绪" : "引擎离线"}
        </span>
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant="ghost"
              size="icon"
              aria-label="切换主题"
              onClick={() => setThemePref(effective === "dark" ? "light" : "dark")}
            >
              {effective === "dark" ? <Moon /> : <Sun />}
            </Button>
          </TooltipTrigger>
          <TooltipContent>切换暗黑 / 浅色主题</TooltipContent>
        </Tooltip>
        <Tooltip>
          <TooltipTrigger asChild>
            <Button variant="ghost" size="icon" aria-label="设置" onClick={() => setView("settings")}>
              <Settings />
            </Button>
          </TooltipTrigger>
          <TooltipContent>设置</TooltipContent>
        </Tooltip>
      </div>
    </header>
  );
}
