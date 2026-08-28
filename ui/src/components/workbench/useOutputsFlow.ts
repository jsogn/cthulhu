import { useCallback, useEffect, useState } from "react";
import {
  deleteOutput,
  fetchOutputs,
  listVariants,
  type OutputInfo,
  type VariantInfo,
} from "@/lib/backend";
import type { Material } from "@/stores/materials";
import { useMaterialsStore } from "@/stores/materials";
import { toast } from "@/stores/toasts";

/** 产物流：当前素材的产物清单、参数弹窗、播放器与删除。 */
export function useOutputsFlow(material: Material | undefined) {
  const [outputs, setOutputs] = useState<OutputInfo[]>([]);
  const [variantDetail, setVariantDetail] = useState<VariantInfo | null>(null);
  const [pendingDelete, setPendingDelete] = useState<OutputInfo | null>(null);
  const [playerPath, setPlayerPath] = useState<string | null>(null);

  const refreshOutputs = useCallback(async () => {
    if (!material?.path) {
      setOutputs([]);
      return;
    }
    await fetchOutputs(material.path)
      .then((result) => {
        setOutputs(result.outputs);
        // 外部手动删除产物文件后记录会被剪除，同步刷新数量标签避免残留。
        void useMaterialsStore.getState().refreshOutputCounts();
      })
      .catch(() => setOutputs([]));
  }, [material?.path]);

  // 产物列表绑定当前素材上下文，切换素材即刷新。
  useEffect(() => {
    void refreshOutputs();
  }, [refreshOutputs]);

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

  const confirmDelete = async () => {
    if (!pendingDelete) return;
    try {
      await deleteOutput(pendingDelete.path);
      await refreshOutputs();
      toast(`已删除产物：${pendingDelete.name}`);
    } catch (error) {
      toast(error instanceof Error ? error.message : "删除失败");
    } finally {
      setPendingDelete(null);
    }
  };

  return {
    outputs,
    variantDetail,
    setVariantDetail,
    pendingDelete,
    setPendingDelete,
    playerPath,
    setPlayerPath,
    showVariant,
    confirmDelete,
    refreshOutputs,
  };
}
