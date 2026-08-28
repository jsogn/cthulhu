import { useTransition } from "react";
import type { LucideIcon } from "lucide-react";
import { FolderOutput, Layers, LayoutDashboard, ListTodo, Settings } from "lucide-react";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { useAppStore, type ViewKey } from "@/stores/app";

const MAIN_ITEMS: { key: ViewKey; label: string; icon: LucideIcon }[] = [
  { key: "workbench", label: "素材处理", icon: LayoutDashboard },
  { key: "jobs", label: "任务中心", icon: ListTodo },
  { key: "templates", label: "去重模板", icon: Layers },
  { key: "products", label: "产物管理", icon: FolderOutput },
];

export default function Rail() {
  const view = useAppStore((state) => state.view);
  const setView = useAppStore((state) => state.setView);
  const [, startTransition] = useTransition();

  const renderItem = ({ key, label, icon: Icon }: { key: ViewKey; label: string; icon: LucideIcon }) => (
    <Tooltip key={key}>
      <TooltipTrigger asChild>
        <button
          type="button"
          className={`rail-item${view === key ? " active" : ""}`}
          aria-current={view === key ? "page" : undefined}
          onClick={() => startTransition(() => setView(key))}
        >
          <Icon className="rail-icon" aria-hidden="true" />
          <span className="rail-label">{label}</span>
        </button>
      </TooltipTrigger>
      <TooltipContent side="right">{label}</TooltipContent>
    </Tooltip>
  );

  return (
    <nav className="rail" aria-label="主导航">
      {MAIN_ITEMS.map(renderItem)}
      <div className="rail-spacer" />
      {renderItem({ key: "settings", label: "设置", icon: Settings })}
    </nav>
  );
}
