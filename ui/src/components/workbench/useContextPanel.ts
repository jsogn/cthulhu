import { useEffect } from "react";
import { useCleanActions } from "@/components/workbench/useCleanActions";
import { useDetectionFlow } from "@/components/workbench/useDetectionFlow";
import { useOutputsFlow } from "@/components/workbench/useOutputsFlow";
import { useTemplateFlow } from "@/components/workbench/useTemplateFlow";
import type {
  DesensitizeJobResult,
  DetectReport,
  JobInfo,
  OutputInfo,
} from "@/lib/backend";
import { useCleanPanel } from "@/stores/cleanPanel";
import { useMaterialsStore, type RiskLevel } from "@/stores/materials";
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

/** 正在排队 / 执行 / 暂停中的检测任务所覆盖的路径（用于防止重复提交）。 */
function pendingDetectPaths(jobs: JobInfo[]): Set<string> {
  return new Set(
    jobs
      .flatMap((job) => job.tasks)
      .filter(
        (task) =>
          task.kind === "detect" &&
          (task.status === "queued" || task.status === "running" || task.status === "paused"),
      )
      .map((task) => task.path),
  );
}

/** 右栏上下文面板：按 模板/检测/产物/清洗动作 四条流组合。 */
export function useContextPanel(
  onStartCompare: (left: string, right: string, leftLabel: string, rightLabel: string) => void,
) {
  const material = useMaterialsStore((state) =>
    state.materials.find((m) => m.id === state.activeId),
  );
  const jobs = useQueueStore((state) => state.jobs);
  const hashOn = useCleanPanel((state) => state.hashOn);
  const setHashOn = useCleanPanel((state) => state.setHashOn);
  const setDctOn = useCleanPanel((state) => state.setDctOn);

  const risk = (material?.risk ?? "待检测") as RiskLevel;
  const score = material?.score ?? 0;
  const report = (material as { report?: DetectReport } | undefined)?.report ?? null;
  const detecting = !!material?.path && pendingDetectPaths(jobs).has(material.path);

  useEffect(() => {
    void useCleanPanel.getState().loadSettings();
  }, []);

  const templateFlow = useTemplateFlow();
  const detectionFlow = useDetectionFlow({ material, report, detecting });
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
    risk,
    score,
    report,
    detecting,
    detectSubmitting: detectionFlow.detectSubmitting,
    audio: detectionFlow.audio,
    audioMissing: detectionFlow.audioMissing,
    templateList: templateFlow.templateList,
    cleanSubmitting: cleanActions.cleanSubmitting,
    outputs: outputsFlow.outputs,
    variantDetail: outputsFlow.variantDetail,
    setVariantDetail: outputsFlow.setVariantDetail,
    pendingDelete: outputsFlow.pendingDelete,
    setPendingDelete: outputsFlow.setPendingDelete,
    playerPath: outputsFlow.playerPath,
    setPlayerPath: outputsFlow.setPlayerPath,
    hashOn,
    setHashOn,
    setDctOn,
    detectNow: detectionFlow.detectNow,
    runClean: cleanActions.runClean,
    showVariant: outputsFlow.showVariant,
    compareOriginal,
    confirmDelete: outputsFlow.confirmDelete,
    runRepairJob: cleanActions.runRepairJob,
    applyTemplateById: templateFlow.applyTemplateById,
  };
}
