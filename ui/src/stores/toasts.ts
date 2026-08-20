import { toast as sonnerToast } from "sonner";

/** 统一 toast 入口，内部使用 shadcn 推荐的 sonner。 */
export const toast = (
  message: string,
  description?: string,
  action?: { label: string; onClick: () => void },
) => {
  if (action) {
    sonnerToast(message, { description, action });
  } else if (description) {
    sonnerToast(message, { description });
  } else {
    sonnerToast(message);
  }
};

/** 在系统文件管理器中定位产物；浏览器开发环境降级为路径提示。 */
export const openInFolder = (filePath: string) => {
  const opener = window.appEnv?.showInFolder;
  if (!opener) {
    toast("桌面版可打开所在文件夹", filePath);
    return;
  }
  void opener(filePath).then((ok) => {
    if (!ok) toast("打开文件夹失败", filePath);
  });
};
