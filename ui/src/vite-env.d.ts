/// <reference types="vite/client" />

// Electron preload 通过 contextBridge 注入的最小环境信息。
interface Window {
  appEnv?: {
    backendUrl: string;
    authToken?: string;
    platform: string;
    getPathForFile?: (file: File) => string | null;
    chooseFolder?: () => Promise<string | null>;
    chooseFile?: () => Promise<string | null>;
    showInFolder?: (filePath: string) => Promise<boolean>;
  };
}
