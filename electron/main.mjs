// Electron 主进程：负责窗口生命周期，并在开发模式下拉起 Python 后端 sidecar。
import { app, BrowserWindow, dialog, ipcMain, shell } from "electron";
import { spawn } from "node:child_process";
import { randomBytes } from "node:crypto";
import fs from "node:fs";
import net from "node:net";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const IS_DEV = !app.isPackaged;
const BACKEND_PORT = 57173;
const DEV_AUTH_TOKEN = "dev-local";
const VITE_URL = process.env.VITE_DEV_SERVER_URL ?? "http://localhost:5173";

/** @type {import('node:child_process').ChildProcess | null} */
let backendProcess = null;

/** 从系统分配一个空闲回环端口，用于生产模式随机化后端监听地址。 */
function getFreePort() {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.unref();
    server.on("error", reject);
    server.listen(0, "127.0.0.1", () => {
      const { port } = server.address();
      server.close(() => resolve(port));
    });
  });
}

async function startBackend() {
  console.log("[backend] 启动模式：", IS_DEV ? "dev" : "packaged", "resources:", process.resourcesPath);
  if (IS_DEV) {
    const backendDir = path.resolve(__dirname, "..", "backend");
    backendProcess = spawn(
      "uv",
      ["run", "uvicorn", "cthulhu_backend.main:app", "--port", String(BACKEND_PORT)],
      {
        cwd: backendDir,
        stdio: "inherit",
        env: {
          ...process.env,
          CTHULHU_AUTH_TOKEN: DEV_AUTH_TOKEN,
          CTHULHU_FFMPEG_INSTALL_DIR: path.join(app.getPath("userData"), "ffmpeg"),
        },
      },
    );
    return { port: BACKEND_PORT, authToken: DEV_AUTH_TOKEN };
  } else {
    // 生产模式：启动打包后的 Python sidecar，并由其后端托管前端静态资源。
    // extraResources 拷贝的是目录内容，可执行文件即 backend/<name>。
    const port = await getFreePort();
    const authToken = randomBytes(32).toString("hex");
    const backendExeName = process.platform === "win32" ? "cthulhu-backend.exe" : "cthulhu-backend";
    const backendExe = path.join(process.resourcesPath, "backend", backendExeName);
    const staticDir = path.join(process.resourcesPath, "backend", "ui-dist");
    const ffmpegDir = path.join(process.resourcesPath, "ffmpeg");
    const ffmpegBinary = path.join(ffmpegDir, process.platform === "win32" ? "ffmpeg.exe" : "ffmpeg");
    const userData = app.getPath("userData");
    const env = {
      ...process.env,
      CTHULHU_STATIC_DIR: staticDir,
      // 数据写入用户目录，避免向只读的 .app 包内写库与演示素材。
      CTHULHU_DB: path.join(userData, "cthulhu.db"),
      CTHULHU_DEMO_LIBRARY: path.join(userData, "demo-library"),
      CTHULHU_AUTH_TOKEN: authToken,
      CTHULHU_PORT: String(port),
      CTHULHU_FFMPEG_INSTALL_DIR: path.join(userData, "ffmpeg"),
      CTHULHU_SKIP_DEMO: "1",
    };
    if (fs.existsSync(ffmpegBinary)) {
      env.CTHULHU_FFMPEG_DIR = ffmpegDir;
    }
    backendProcess = spawn(backendExe, [], {
      env,
      stdio: "inherit",
    });
    console.log("[backend] pid:", backendProcess.pid, "exe:", backendExe);
    return { port, authToken };
  }
  backendProcess.on("exit", (code) => {
    console.log(`[backend] 退出，code=${code}`);
    backendProcess = null;
  });
  backendProcess.on("error", (error) => {
    console.error("[backend] 启动错误：", error.message);
  });
}

async function waitForBackend(port, timeoutMs = 30000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const res = await fetch(`http://127.0.0.1:${port}/api/health`);
      if (res.ok) return;
    } catch {
      // 后端尚未就绪，继续轮询
    }
    await new Promise((resolve) => setTimeout(resolve, 300));
  }
  throw new Error("Python 后端启动超时");
}

async function createWindow() {
  const { port, authToken } = await startBackend();
  await waitForBackend(port);
  const backendUrl = `http://127.0.0.1:${port}`;

  const win = new BrowserWindow({
    width: 1440,
    height: 900,
    minWidth: 1280,
    minHeight: 720,
    title: "暗水印清洗台",
    backgroundColor: "#0b0d12",
    webPreferences: {
      preload: path.join(__dirname, "preload.cjs"),
      contextIsolation: true,
      nodeIntegration: false,
      additionalArguments: [`--backend-url=${backendUrl}`, `--auth-token=${authToken}`],
    },
  });

  // 把渲染进程 console 转发到终端，便于桌面端联调。
  win.webContents.on("console-message", (event) => {
    console.log(`[renderer:${event.level}] ${event.message}`);
  });

  // 开发模式加载 Vite；生产模式加载后端托管的静态资源（打包阶段接入）。
  if (IS_DEV) {
    await win.loadURL(VITE_URL);
  } else {
    await win.loadURL(`${backendUrl}/`);
  }
}

app.whenReady().then(() => {
  ipcMain.handle("dialog:open-directory", async () => {
    const result = await dialog.showOpenDialog({ properties: ["openDirectory"] });
    return result.canceled ? null : result.filePaths[0];
  });

  ipcMain.handle("dialog:open-file", async () => {
    const result = await dialog.showOpenDialog({ properties: ["openFile"] });
    return result.canceled ? null : result.filePaths[0];
  });

  ipcMain.handle("shell:show-in-folder", async (_event, filePath) => {
    try {
      if (filePath && fs.existsSync(filePath)) {
        shell.showItemInFolder(filePath);
      } else if (filePath) {
        const directory = path.dirname(filePath);
        await shell.openPath(directory);
      }
      return true;
    } catch (error) {
      console.error("[shell] 打开文件夹失败：", error);
      return false;
    }
  });

  createWindow().catch((err) => {
    console.error("[electron] 启动失败：", err);
    app.quit();
  });

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});

app.on("before-quit", () => {
  backendProcess?.kill();
});
