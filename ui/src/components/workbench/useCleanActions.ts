import { useEffect, useRef, useState } from "react";
import { enqueueJob } from "@/lib/backend";
import { cleanOutputPath, makeCleanOptions } from "@/lib/cleanOptions";
import { useCleanPanel } from "@/stores/cleanPanel";
import type { Material } from "@/stores/materials";
import { toast } from "@/stores/toasts";

/** 清洗动作流：入队、防重复提交与输出路径组装。 */
export function useCleanActions(material: Material | undefined) {
  const [cleanSubmitting, setCleanSubmitting] = useState(false);
  const cleanDebounceRef = useRef<number | null>(null);

  useEffect(
    () => () => {
      if (cleanDebounceRef.current !== null) window.clearTimeout(cleanDebounceRef.current);
    },
    [],
  );

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

  return { cleanSubmitting, runClean };
}
