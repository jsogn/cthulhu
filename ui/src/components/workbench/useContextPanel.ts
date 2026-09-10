import { useEffect } from "react";
import { useCleanActions } from "@/components/workbench/useCleanActions";
import { useOutputsFlow } from "@/components/workbench/useOutputsFlow";
import { useTemplateFlow } from "@/components/workbench/useTemplateFlow";
import type { DesensitizeJobResult, OutputInfo } from "@/lib/backend";
import { useCleanPanel } from "@/stores/cleanPanel";
import { useMaterialsStore } from "@/stores/materials";
import { useQueueStore } from "@/stores/queue";
import { toast } from "@/stores/toasts";

const HANDLED_TASKS_KEY = "cthulhu-handled-task-ids";

function loadHandledTaskIds(): Set<string> {
  try {
    return new Set(JSON.parse(localStorage.getItem(HANDLED_TASKS_KEY) ?? "[]"));
  } catch {
    return new Set();
  }
}

const handledTaskIds = loadHandledTaskIds();

function rememberHandledTask(id: string): void {
  handledTaskIds.add(id);
  try {
    localStorage.setItem(
      HANDLED_TASKS_KEY,
      JSON.stringify([...handledTaskIds].slice(-50)),
    );
  } catch {
    // 本地存储不可用时仅在会话内去重
  }
}

/** 右栏上下文面板：按 模板/产物/清洗动作 三条流组合。 */
export function useContextPanel(
  onStartCompare: (left: string, right: string, leftLabel: string, rightLabel: string) => void,
) {
  const material = useMaterialsStore((state) =>
    state.materials.find((m) => m.id === state.activeId),
  );
  const jobs = useQueueStore((state) => state.jobs);

  useEffect(() => {
    void useCleanPanel.getState().loadSettings();
  }, []);

  const templateFlow = useTemplateFlow();
  const outputsFlow = useOutputsFlow(material);
  const cleanActions = useCleanActions(material);

  // 清洗任务完成后：刷新产物列表并提示。
  useEffect(() => {
    const task = jobs
      .flatMap((job) => job.tasks)
      .find(
        (item) =>
          item.kind === "desensitize" &&
          item.path === material?.path &&
          item.status === "done" &&
          item.result != null,
      );
    if (!task || handledTaskIds.has(task.id)) return;
    const result = task.result as DesensitizeJobResult | null;
    if (!result?.output) return;
    rememberHandledTask(task.id);
    void outputsFlow.refreshOutputs();
    const outName = result.output.split(/[\\/]/).pop() ?? result.output;
    toast("清洗完成", outName);
  }, [jobs, material?.path, outputsFlow.refreshOutputs]);

  const runComparePair = (a: string, b: string, leftLabel: string, rightLabel: string) => {
    onStartCompare(a, b, leftLabel, rightLabel);
  };

  const compareOriginal = (output: OutputInfo) => {
    if (material?.path) runComparePair(material.path, output.path, "原片", "处理后");
  };

  return {
    material,
    templateList: templateFlow.templateList,
    cleanSubmitting: cleanActions.cleanSubmitting,
    outputs: outputsFlow.outputs,
    variantDetail: outputsFlow.variantDetail,
    setVariantDetail: outputsFlow.setVariantDetail,
    pendingDelete: outputsFlow.pendingDelete,
    setPendingDelete: outputsFlow.setPendingDelete,
    playerPath: outputsFlow.playerPath,
    setPlayerPath: outputsFlow.setPlayerPath,
    runClean: cleanActions.runClean,
    showVariant: outputsFlow.showVariant,
    compareOriginal,
    confirmDelete: outputsFlow.confirmDelete,
    applyTemplateById: templateFlow.applyTemplateById,
  };
}
