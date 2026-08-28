import { useEffect, useRef, useState } from "react";
import {
  analyzeAudio,
  deleteOutput,
  enqueueJob,
  fetchOutputs,
  listVariants,
  listTemplates,
  type AudioAnalysis,
  type DesensitizeJobResult,
  type DetectReport,
  type JobInfo,
  type OutputInfo,
  type TemplateInfo,
  type VariantInfo,
} from "@/lib/backend";
import { cleanOutputPath, makeCleanOptions } from "@/lib/cleanOptions";
import { payloadOf } from "@/lib/templates";
import { useAppStore } from "@/stores/app";
import { useCleanPanel } from "@/stores/cleanPanel";
import { useMaterialsStore, type Material, type RiskLevel } from "@/stores/materials";
import { useQueueStore } from "@/stores/queue";
import { useRegionsStore } from "@/stores/regions";
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

/** 右栏上下文面板的编排逻辑：状态、副作用与业务处理，供 ContextPanel 装配。 */
export function useContextPanel(
  onStartCompare: (left: string, right: string, leftLabel: string, rightLabel: string) => void,
) {
  const material = useMaterialsStore((state) =>
    state.materials.find((m) => m.id === state.activeId),
  );
  const jobs = useQueueStore((state) => state.jobs);
  const {
    antiLevel,
    setAntiLevel,
    settingsExportDir,
    settingsNaming,
    templateId,
    setTemplateId,
    applyTemplatePayload,
    resetCleanDefaults,
  } = useCleanPanel();
  const [templateList, setTemplateList] = useState<TemplateInfo[]>([]);
  const [templatesLoaded, setTemplatesLoaded] = useState(false);
  const [cleanSubmitting, setCleanSubmitting] = useState(false);
  const cleanDebounceRef = useRef<number | null>(null);
  const [detectSubmitting, setDetectSubmitting] = useState(false);
  const [audio, setAudio] = useState<AudioAnalysis | null>(null);
  const [audioMissing, setAudioMissing] = useState(false);
  const [outputs, setOutputs] = useState<OutputInfo[]>([]);
  const [variantDetail, setVariantDetail] = useState<VariantInfo | null>(null);
  const [pendingDelete, setPendingDelete] = useState<OutputInfo | null>(null);
  const [playerPath, setPlayerPath] = useState<string | null>(null);

  const risk = (material?.risk ?? "待检测") as RiskLevel;
  const score = material?.score ?? 0;
  const report = (material as { report?: DetectReport } | undefined)?.report ?? null;
  const detecting = !!material?.path && pendingDetectPaths(jobs).has(material.path);

  useEffect(
    () => () => {
      if (cleanDebounceRef.current !== null) window.clearTimeout(cleanDebounceRef.current);
    },
    [],
  );

  useEffect(() => {
    setAudio(null);
    setAudioMissing(false);
    if (!material?.path || !report) return;
    let cancelled = false;
    void analyzeAudio(material.path)
      .then((result) => {
        if (!cancelled) setAudio(result);
      })
      .catch(() => {
        if (!cancelled) setAudioMissing(true);
      });
    return () => {
      cancelled = true;
    };
  }, [material?.path, report]);

  useEffect(() => {
    void useCleanPanel.getState().loadSettings();
  }, []);

  const applyTemplateById = (id: string) => {
    if (id === "manual") {
      resetCleanDefaults();
      setTemplateId("manual");
      return;
    }
    const template = templateList.find((item) => item.id === id);
    if (!template) return;
    applyTemplatePayload(payloadOf(template));
    setTemplateId(id);
    toast(`已套用模板：${template.name}`);
  };

  // 模板列表加载完成前不做“孤儿模板”检查，避免切回页面瞬间误清参数。
  useEffect(() => {
    let cancelled = false;
    void listTemplates()
      .then((list) => {
        if (cancelled) return;
        setTemplateList(list);
        setTemplatesLoaded(true);
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, []);

  // 模板页点击「使用」后，切回工作台时把参数回填到右侧面板。
  useEffect(() => {
    const pending = useAppStore.getState().consumePendingTemplate();
    if (pending) {
      applyTemplatePayload(pending.payload);
      setTemplateId(pending.id);
    }
  }, []);

  // 已套用的模板被删除或列表刷新后，回退为手动状态，避免下拉框悬空。
  useEffect(() => {
    if (!templatesLoaded) return;
    if (templateId !== "manual" && !templateList.some((item) => item.id === templateId)) {
      resetCleanDefaults();
      setTemplateId("manual");
    }
  }, [templatesLoaded, templateList, templateId]);

  // 产物列表绑定当前素材上下文，切换素材即刷新。
  useEffect(() => {
    if (!material?.path) {
      setOutputs([]);
      return;
    }
    void fetchOutputs(material.path)
      .then((result) => {
        setOutputs(result.outputs);
        // 外部手动删除产物文件后记录会被剪除，同步刷新数量标签避免残留。
        void useMaterialsStore.getState().refreshOutputCounts();
      })
      .catch(() => setOutputs([]));
  }, [material?.path]);

  const runClean = async () => {
    const target = material as (Material & { path?: string }) | null;
    if (!target?.path || !material) {
      toast("该素材缺少本地路径，无法执行");
      return;
    }
    const now = new Date();
    const pad = (value: number) => String(value).padStart(2, "0");
    const ts = `${pad(now.getMonth() + 1)}${pad(now.getDate())}_${pad(
      now.getHours(),
    )}_${pad(now.getMinutes())}_${pad(now.getSeconds())}`;
    const cleanPanel = useCleanPanel.getState();
    const output = cleanOutputPath(
      target.path,
      cleanPanel.settingsNaming,
      cleanPanel.settingsExportDir,
      ts,
    );
    try {
      await enqueueJob(`${material.name} · 清洗去重`, [
        {
          kind: "desensitize",
          path: target.path,
          options: { output, ...makeCleanOptions(cleanPanel) },
        },
      ]);
      setCleanSubmitting(true);
      if (cleanDebounceRef.current !== null) window.clearTimeout(cleanDebounceRef.current);
      cleanDebounceRef.current = window.setTimeout(() => setCleanSubmitting(false), 1800);
      toast("已加入处理队列，完成后自动刷新校验与预览");
    } catch (error) {
      toast(error instanceof Error ? error.message : "入队失败");
    }
  };

  const detectNow = async () => {
    if (!material?.path) {
      toast("该素材缺少本地路径，无法检测");
      return;
    }
    if (detecting || detectSubmitting) {
      toast("该素材正在检测中，请等待完成");
      return;
    }
    setDetectSubmitting(true);
    try {
      const job = await enqueueJob(`${material.name} · 检测`, [
        { kind: "detect", path: material.path },
      ]);
      // 立即写入队列状态，避免按钮在事件到达前仍可重复点击。
      useQueueStore.getState().applyEvent({ type: "job:state", job });
      toast("已加入检测队列，完成后结果自动更新");
    } catch {
      toast("入队失败，请确认引擎在线");
    } finally {
      setDetectSubmitting(false);
    }
  };

  const showVariant = (outputPath: string) => {
    if (!material?.path) return;
    void listVariants(material.path)
      .then((list) => {
        const found = list.find((item) => item.output === outputPath);
        if (found) {
          setVariantDetail(found);
        } else {
          toast("该产物没有参数记录", "旧版本产物可能没有记录");
        }
      })
      .catch(() => toast("读取产物记录失败"));
  };

  const runComparePair = (a: string, b: string, leftLabel: string, rightLabel: string) => {
    onStartCompare(a, b, leftLabel, rightLabel);
  };

  const compareOriginal = (output: OutputInfo) => {
    if (material?.path) runComparePair(material.path, output.path, "原片", "处理后");
  };

  const confirmDelete = async () => {
    if (!pendingDelete) return;
    try {
      await deleteOutput(pendingDelete.path);
      await fetchOutputs(material?.path ?? "")
        .then((report) => setOutputs(report.outputs))
        .catch(() => undefined);
      void useMaterialsStore.getState().refreshOutputCounts();
      toast(`已删除产物：${pendingDelete.name}`);
    } catch (error) {
      toast(error instanceof Error ? error.message : "删除失败");
    } finally {
      setPendingDelete(null);
    }
  };

  // 清洗任务完成后：刷新产物列表并提示；判重等补充指标留在任务结果里，不打扰用户。
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
    if (!task) return;
    if (handledTaskIds.has(task.id)) return;
    const result = task.result as DesensitizeJobResult | null;
    if (!result?.output) return;
    rememberHandledTask(task.id);
    void useMaterialsStore.getState().refreshOutputCounts();
    void fetchOutputs(material?.path ?? "")
      .then((report) => setOutputs(report.outputs))
      .catch(() => undefined);
    const outName = result.output.split(/[\\/]/).pop() ?? result.output;
    toast("清洗完成", outName);
  }, [jobs, material?.path]);

  const runRepairJob = async () => {
    const target = material as (Material & { path?: string }) | null;
    if (!target?.path) {
      toast("该素材缺少本地路径，无法执行");
      return;
    }
    const regions = useRegionsStore.getState().regions;
    if (!regions.length) {
      toast("请先添加水印区域");
      return;
    }
    const now = new Date();
    const pad = (value: number) => String(value).padStart(2, "0");
    const ts = `${pad(now.getMonth() + 1)}${pad(now.getDate())}_${pad(
      now.getHours(),
    )}_${pad(now.getMinutes())}_${pad(now.getSeconds())}`;
    const srcName = target.path.split(/[\\/]/).pop()?.replace(/\.(mp4|mov|mkv|avi|flv|ts)$/i, "") ?? "素材";
    const fileName =
      settingsNaming === "时间戳 + 原文件名"
        ? `${ts}_${srcName}_修复.mp4`
        : `${srcName}_修复_${ts}.mp4`;
    const baseDir = settingsExportDir || "~/Documents/Cthulhu";
    const output = `${baseDir.replace(/\/+$/, "")}/${fileName}`;
    try {
      await enqueueJob(`${target.name} · 修复`, [
        {
          kind: "repair",
          path: target.path,
          options: {
            output,
            crf: 23,
            regions: regions.map((region) => ({
              x: region.left / 100,
              y: region.top / 100,
              w: region.w / 100,
              h: region.h / 100,
              ...(region.start != null ? { start: region.start } : {}),
              ...(region.end != null ? { end: region.end } : {}),
            })),
          },
        },
      ]);
      toast("修复任务已加入队列");
    } catch {
      toast("入队失败，请确认引擎在线");
    }
  };

  return {
    material,
    risk,
    score,
    report,
    detecting,
    detectSubmitting,
    audio,
    audioMissing,
    templateList,
    cleanSubmitting,
    outputs,
    variantDetail,
    setVariantDetail,
    pendingDelete,
    setPendingDelete,
    playerPath,
    setPlayerPath,
    antiLevel,
    setAntiLevel,
    detectNow,
    runClean,
    showVariant,
    compareOriginal,
    confirmDelete,
    runRepairJob,
    applyTemplateById,
  };
}
