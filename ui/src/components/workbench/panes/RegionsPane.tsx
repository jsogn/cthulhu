import { Plus, Redo2, Trash2, Undo2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { TabsContent } from "@/components/ui/tabs";
import { cn } from "@/lib/utils";
import { useRegionsStore } from "@/stores/regions";
import { toast } from "@/stores/toasts";

interface RegionsPaneProps {
  onRunRepair: () => void;
}

export function RegionsPane({ onRunRepair }: RegionsPaneProps) {
  const regions = useRegionsStore((state) => state.regions);
  const activeRegionId = useRegionsStore((state) => state.activeId);
  const undoStack = useRegionsStore((state) => state.undoStack);
  const redoStack = useRegionsStore((state) => state.redoStack);
  const selectRegion = useRegionsStore((state) => state.selectRegion);
  const addRegion = useRegionsStore((state) => state.addRegion);
  const deleteRegion = useRegionsStore((state) => state.deleteRegion);
  const patchRegion = useRegionsStore((state) => state.patchRegion);
  const undo = useRegionsStore((state) => state.undo);
  const redo = useRegionsStore((state) => state.redo);

  return (
    <TabsContent value="水印区域" className="tab-pane">
      <ScrollArea className="h-full">
        <div className="flex flex-col gap-2.5">
          <div className="undo-row">
            <Button variant="ghost" size="sm" disabled={undoStack.length === 0} onClick={undo}>
              <Undo2 />
              撤销
            </Button>
            <Button variant="ghost" size="sm" disabled={redoStack.length === 0} onClick={redo}>
              <Redo2 />
              重做
            </Button>
          </div>

          {regions.length === 0 ? (
            <div className="mat-empty">还没有水印区域，点击下方按钮添加</div>
          ) : (
            regions.map((r) => (
              <div
                key={r.id}
                className={cn("region-card", r.id === activeRegionId && "active")}
                onClick={() => selectRegion(r.id)}
              >
                <div className="region-head">
                  <span className="region-name">{r.name}</span>
                  <span className="region-tools">
                    <Button
                      variant="ghost"
                      size="icon"
                      className="size-6"
                      aria-label={`删除${r.name}`}
                      onClick={(e) => {
                        e.stopPropagation();
                        deleteRegion(r.id);
                      }}
                    >
                      <Trash2 />
                    </Button>
                  </span>
                </div>
                <div className="field">
                  <span className="field-label">生效时间（秒，留空为全程）</span>
                  <div className="flex gap-2">
                    <Input
                      className="h-8"
                      type="number"
                      min={0}
                      placeholder="开始"
                      value={r.start ?? ""}
                      onChange={(e) =>
                        patchRegion(r.id, {
                          start: e.target.value === "" ? null : Number(e.target.value),
                        })
                      }
                    />
                    <Input
                      className="h-8"
                      type="number"
                      min={0}
                      placeholder="结束"
                      value={r.end ?? ""}
                      onChange={(e) =>
                        patchRegion(r.id, {
                          end: e.target.value === "" ? null : Number(e.target.value),
                        })
                      }
                    />
                  </div>
                </div>
              </div>
            ))
          )}

          <Button
            variant="secondary"
            size="sm"
            className="w-full"
            onClick={() => {
              addRegion();
              toast("已添加水印区域，可在预览中拖动调整");
            }}
          >
            <Plus />
            添加水印区域
          </Button>
          <Button variant="secondary" size="sm" className="w-full" onClick={onRunRepair}>
            执行修复
          </Button>
          <p className="note">
            在预览中拖动选框圈住水印位置，可调整大小与参数。
          </p>
        </div>
      </ScrollArea>
    </TabsContent>
  );
}
