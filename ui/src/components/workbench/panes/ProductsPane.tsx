import { GitCompareArrows } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { TabsContent } from "@/components/ui/tabs";
import type { OutputInfo } from "@/lib/backend";
import { fmtSize } from "@/lib/format";
import { OUTPUT_KIND_LABEL, outputTimeLabel } from "@/lib/workbenchOptions";
import { openInFolder } from "@/stores/toasts";

interface ProductsPaneProps {
  outputs: OutputInfo[];
  compareOriginal: (output: OutputInfo) => void;
  showVariant: (outputPath: string) => void;
  setPlayerPath: (path: string | null) => void;
  setPendingDelete: (output: OutputInfo | null) => void;
}

export function ProductsPane({
  outputs,
  compareOriginal,
  showVariant,
  setPlayerPath,
  setPendingDelete,
}: ProductsPaneProps) {
  return (
    <TabsContent value="处理产物" className="tab-pane">
      <ScrollArea className="h-full">
        <div className="flex min-w-0 flex-col gap-2.5">
          {outputs.length === 0 ? (
            <p className="note">
              还没有处理产物，清洗或修复后会自动出现。
            </p>
          ) : (
            <>
              <p className="note">与原片对比，或播放查看。</p>
              {outputs.map((output) => (
                <div key={output.path} className="out-item">
                  <div className="out-item-head">
                    <span className="out-kind">{OUTPUT_KIND_LABEL[output.kind]}</span>
                    <span className="out-name" title={output.name}>{output.name}</span>
                  </div>
                  <div className="out-meta">
                    {fmtSize(output.size)} · {outputTimeLabel(output.mtime)}
                  </div>
                  <div className="out-item-actions">
                    <Button
                      variant="secondary"
                      size="sm"
                      className="w-full"
                      onClick={() => compareOriginal(output)}
                    >
                      <GitCompareArrows /> 对比原片
                    </Button>
                    <div className="out-item-secondary">
                      <Button
                        variant="ghost"
                        size="sm"
                        className="flex-1"
                        onClick={() => setPlayerPath(output.path)}
                      >
                        播放
                      </Button>
                      <Button
                        variant="ghost"
                        size="sm"
                        className="flex-1"
                        onClick={() => openInFolder(output.path)}
                      >
                        打开
                      </Button>
                      <Button
                        variant="ghost"
                        size="sm"
                        className="flex-1"
                        onClick={() => showVariant(output.path)}
                      >
                        参数
                      </Button>
                      <Button
                        variant="ghost"
                        size="sm"
                        className="flex-1 text-destructive"
                        onClick={() => setPendingDelete(output)}
                      >
                        删除
                      </Button>
                    </div>
                  </div>
                </div>
              ))}
            </>
          )}
        </div>
      </ScrollArea>
    </TabsContent>
  );
}
