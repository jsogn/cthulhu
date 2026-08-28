import { useEffect, useState } from "react";
import { listTemplates, type TemplateInfo } from "@/lib/backend";
import { payloadOf } from "@/lib/templates";
import { useAppStore } from "@/stores/app";
import { useCleanPanel } from "@/stores/cleanPanel";
import { toast } from "@/stores/toasts";

/** 模板流：列表加载、跨页回填、孤儿回退与应用模板。 */
export function useTemplateFlow() {
  const templateId = useCleanPanel((state) => state.templateId);
  const setTemplateId = useCleanPanel((state) => state.setTemplateId);
  const applyTemplatePayload = useCleanPanel((state) => state.applyTemplatePayload);
  const resetCleanDefaults = useCleanPanel((state) => state.resetCleanDefaults);
  const [templateList, setTemplateList] = useState<TemplateInfo[]>([]);
  const [templatesLoaded, setTemplatesLoaded] = useState(false);

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

  return { templateList, applyTemplateById };
}
