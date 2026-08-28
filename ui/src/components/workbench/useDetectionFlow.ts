import { useEffect, useState } from "react";
import {
  analyzeAudio,
  enqueueJob,
  type AudioAnalysis,
  type DetectReport,
} from "@/lib/backend";
import type { Material } from "@/stores/materials";
import { useQueueStore } from "@/stores/queue";
import { toast } from "@/stores/toasts";

interface DetectionFlowArgs {
  material: Material | undefined;
  report: DetectReport | null;
  detecting: boolean;
}

/** 检测流：检测任务入队与当前素材的音轨分析。 */
export function useDetectionFlow({ material, report, detecting }: DetectionFlowArgs) {
  const [detectSubmitting, setDetectSubmitting] = useState(false);
  const [audio, setAudio] = useState<AudioAnalysis | null>(null);
  const [audioMissing, setAudioMissing] = useState(false);

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

  return { detectSubmitting, audio, audioMissing, detectNow };
}
