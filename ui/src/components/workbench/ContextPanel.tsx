import { Tabs } from "@/components/ui/tabs";
import {
  ContextTabs,
  type WorkbenchTab,
} from "@/components/workbench/ContextTabs";
import { DeleteProductDialog } from "@/components/workbench/dialogs/DeleteProductDialog";
import { ProductPlayerDialog } from "@/components/workbench/dialogs/ProductPlayerDialog";
import { VariantDetailDialog } from "@/components/workbench/dialogs/VariantDetailDialog";
import { CleanPane } from "@/components/workbench/panes/CleanPane";
import { ProductsPane } from "@/components/workbench/panes/ProductsPane";
import { useContextPanel } from "@/components/workbench/useContextPanel";

export interface ContextProps {
  tab: WorkbenchTab;
  setTab: (tab: WorkbenchTab) => void;
  onStartCompare: (left: string, right: string, leftLabel: string, rightLabel: string) => void;
}

/** 右栏上下文面板：编排状态来自 useContextPanel，这里只做 JSX 装配。 */
export function ContextPanel({ tab, setTab, onStartCompare }: ContextProps) {
  const ctx = useContextPanel(onStartCompare);

  return (
    <aside className="pane pane-right">
      <Tabs value={tab} onValueChange={(value) => setTab(value as WorkbenchTab)} className="min-h-0 flex-1">
        <ContextTabs setTab={setTab} />

        <CleanPane
          material={ctx.material ?? null}
          templateList={ctx.templateList}
          applyTemplateById={ctx.applyTemplateById}
          cleanSubmitting={ctx.cleanSubmitting}
          runClean={() => void ctx.runClean()}
        />
        <ProductsPane
          outputs={ctx.outputs}
          compareOriginal={ctx.compareOriginal}
          showVariant={ctx.showVariant}
          setPlayerPath={ctx.setPlayerPath}
          setPendingDelete={ctx.setPendingDelete}
        />
      </Tabs>

      <VariantDetailDialog
        variantDetail={ctx.variantDetail}
        onClose={() => ctx.setVariantDetail(null)}
      />
      <ProductPlayerDialog
        playerPath={ctx.playerPath}
        onClose={() => ctx.setPlayerPath(null)}
      />
      <DeleteProductDialog
        pendingDelete={ctx.pendingDelete}
        onClose={() => ctx.setPendingDelete(null)}
        onConfirm={() => void ctx.confirmDelete()}
      />
    </aside>
  );
}
