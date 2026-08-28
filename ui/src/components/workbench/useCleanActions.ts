import { useEffect, useRef, useState } from "react";
import { enqueueJob } from "@/lib/backend";
import { cleanOutputPath, makeCleanOptions } from "@/lib/cleanOptions";
import { useCleanPanel } from "@/stores/cleanPanel";
import type { Material } from "@/stores/materials";
import { useRegionsStore } from "@/stores/regions";
import { toast } from "@/stores/toasts";

/** 清洗/修复动作流：入队、防重复提交与输出路径组装。 */
export function useCleanActions(material: Material | undefined) {
  const settingsNaming = useCleanPanel((state) => state.settingsNaming);
  const settingsExportDir = useCleanPanel((state) => state.settingsExportDir);
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

  return { cleanSubmitting, runClean, runRepairJob };
}
