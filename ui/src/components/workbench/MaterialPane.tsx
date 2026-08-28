import { useState } from "react";
import { Import } from "lucide-react";
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
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { clearLibrary, thumbUrl, unregisterLibrary } from "@/lib/backend";
import { cn } from "@/lib/utils";
import { useAppStore } from "@/stores/app";
import { useMaterialsStore } from "@/stores/materials";
import { toast } from "@/stores/toasts";

export function MaterialPane() {
  const materials = useMaterialsStore((state) => state.materials);
  const loadFailed = useMaterialsStore((state) => state.loadFailed);
  const activeId = useMaterialsStore((state) => state.activeId);
  const selected = useMaterialsStore((state) => state.selected);
  const search = useMaterialsStore((state) => state.search);
  const select = useMaterialsStore((state) => state.select);
  const toggleSelected = useMaterialsStore((state) => state.toggleSelected);
  const refreshLibrary = useMaterialsStore((state) => state.refreshLibrary);
  const [clearOpen, setClearOpen] = useState(false);
  const setSearch = useMaterialsStore((state) => state.setSearch);
  const toggleSelectAll = useMaterialsStore((state) => state.toggleSelectAll);
  const setImportOpen = useAppStore((state) => state.setImportOpen);
  const backendConnected = useAppStore((state) => state.backendConnected);

  const visible = materials.filter((m) => {
    const kw = search.trim().toLowerCase();
    if (kw && !m.name.toLowerCase().includes(kw) && !m.tags.join(" ").toLowerCase().includes(kw)) {
      return false;
    }
    return true;
  });
  const allSelected = visible.length > 0 && visible.every((m) => selected[m.id]);

  return (
    <aside className="pane pane-left">
      <div className="pane-head">
        <div>
          <div className="pane-title">素材库</div>
          <div className="pane-count">{visible.length} 个素材</div>
        </div>
      </div>

      <div className="pane-body">
        <div className="toolbar-row">
          <Button variant="default" size="sm" onClick={() => setImportOpen(true)}>
            <Import />
            导入
          </Button>
          <Button variant="ghost" size="sm" onClick={() => toggleSelectAll(visible.map((m) => m.id))}>
            {allSelected ? "取消全选" : "全选"}
          </Button>
          <Button
            variant="ghost"
            size="sm"
            className="text-destructive hover:text-destructive"
            disabled={materials.length === 0}
            onClick={() => setClearOpen(true)}
          >
            清空
          </Button>
        </div>

        <Input
          className="search"
          type="search"
          placeholder="搜索文件名、标签…"
          aria-label="搜索素材"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />

        {materials.length === 0 ? (
          loadFailed ? (
            <div className="mat-empty">
              <p>素材库加载失败，请确认引擎在线</p>
              <Button
                variant="secondary"
                size="sm"
                onClick={() => void useMaterialsStore.getState().refreshLibrary()}
              >
                重试
              </Button>
            </div>
          ) : !backendConnected ? (
            <div className="mat-empty">
              <p>正在启动处理引擎…</p>
              <p className="mat-empty-hint">首次启动需要一点时间，请稍候</p>
            </div>
          ) : (
            <div className="mat-empty">
              <p>素材库还是空的</p>
              <Button variant="secondary" size="sm" onClick={() => setImportOpen(true)}>
                导入第一批素材
              </Button>
            </div>
          )
        ) : visible.length === 0 ? (
          <div className="mat-empty">没有匹配的素材，试试调整搜索或筛选条件</div>
        ) : (
          <ScrollArea className="min-h-0 flex-1">
            <div className="mat-list">
              {visible.map((m) => (
                <div
                  key={m.id}
                  className={cn("mat-item", m.id === activeId && "active")}
                  role="button"
                  tabIndex={0}
                  aria-label={`选择素材 ${m.name}`}
                  onClick={() => !m.missing && select(m.id)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault();
                      if (!m.missing) select(m.id);
                    }
                  }}
                >
                  <Checkbox
                    className="mat-check"
                    checked={!!selected[m.id]}
                    onCheckedChange={() => toggleSelected(m.id)}
                    aria-label="勾选此素材"
                    onClick={(e) => e.stopPropagation()}
                    onKeyDown={(e) => e.stopPropagation()}
                  />
                  <span className="mat-thumb">
                    {m.missing ? (
                      <span className="mat-missing-icon">无文件</span>
                    ) : (
                      <img
                        src={m.path ? thumbUrl(m.path) : m.frame}
                        loading="lazy"
                        alt=""
                      />
                    )}
                  </span>
                  <span className="mat-info">
                    <span className="mat-name">
                      {m.name}
                      {m.missing && <span className="tag tag-warn">文件不存在</span>}
                    </span>
                    <span className="mat-meta mono">
                      {m.dur} · {m.res} · {m.fps} · {m.size}
                    </span>
                    {(m.outputCount ?? 0) > 0 && (
                      <span className="mat-tags">
                        <span className="tag tag-out">产物 {m.outputCount}</span>
                      </span>
                    )}
                    {m.missing && (
                      <button
                        type="button"
                        className="text-xs text-destructive underline"
                        onClick={(e) => {
                          e.stopPropagation();
                          void unregisterLibrary(m.path!)
                            .then(() => {
                              toast(`已移除记录：${m.name}`);
                              void refreshLibrary();
                            })
                            .catch(() => toast("移除记录失败"));
                        }}
                      >
                        移除记录
                      </button>
                    )}
                  </span>
                </div>
              ))}
            </div>
          </ScrollArea>
        )}
      </div>

      <AlertDialog open={clearOpen} onOpenChange={setClearOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>确认清空素材库</AlertDialogTitle>
            <AlertDialogDescription>
              将删除全部 {materials.length} 条素材记录及其对应的产物文件（含产物记录）。
              源视频文件不会删除，此操作不可撤销。
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>取消</AlertDialogCancel>
            <AlertDialogAction
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
              onClick={() => {
                void clearLibrary()
                  .then((result) => {
                    toast(
                      `已清空：${result.removed_materials} 条素材、${result.removed_products} 个产物`,
                    );
                    void refreshLibrary();
                  })
                  .catch(() => toast("清空素材库失败"));
                setClearOpen(false);
              }}
            >
              确认清空
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </aside>
  );
}
