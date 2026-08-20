// preload：以最小面暴露环境信息，渲染进程不直接接触 Node。
const { contextBridge, ipcRenderer, webUtils } = require("electron");

// 主进程经 additionalArguments 注入后端地址与一次性令牌。
function readArg(name, fallback) {
  const prefix = `--${name}=`;
  const found = process.argv.find((arg) => arg.startsWith(prefix));
  return found ? found.slice(prefix.length) : fallback;
}

contextBridge.exposeInMainWorld("appEnv", {
  backendUrl: readArg("backend-url", "http://127.0.0.1:57173"),
  authToken: readArg("auth-token", ""),
  platform: process.platform,
  chooseFolder: () => ipcRenderer.invoke("dialog:open-directory"),
  chooseFile: () => ipcRenderer.invoke("dialog:open-file"),
  showInFolder: (filePath) => ipcRenderer.invoke("shell:show-in-folder", filePath),
  // 拖入的 File 对象拿不到完整路径，经 webUtils 桥接给检测引擎。
  getPathForFile: (file) => {
    try {
      return webUtils.getPathForFile(file);
    } catch {
      return null;
    }
  },
});
