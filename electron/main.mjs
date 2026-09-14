// Electron 主进程：负责窗口生命周期，并在开发模式下拉起 Python 后端 sidecar。
import { app, BrowserWindow, dialog, ipcMain, shell } from "electron";
import { spawn } from "node:child_process";
import { randomBytes } from "node:crypto";
import fs from "node:fs";
import net from "node:net";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  decideEngineRestart,
  FORCE_KILL_DELAY_MS,
  HEALTH_PROBE_TIMEOUT_MS,
  isEngineWorking,
  parseHeartbeat,
  WATCHDOG_FAILURE_THRESHOLD,
  WATCHDOG_INTERVAL_MS,
  restartDelayMs,
} from "./supervisor.mjs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const IS_DEV = !app.isPackaged;
const BACKEND_PORT = 57173;
const DEV_AUTH_TOKEN = "dev-local";
const VITE_URL = process.env.VITE_DEV_SERVER_URL ?? "http://localhost:5173";

/** 系统缓存目录：Chromium 会话数据与后端可重建缓存统一放这里，
 *  Application Support 只保留数据库、素材库等用户数据。 */
function getCacheDir() {
  if (process.platform === "darwin") {
    return path.join(os.homedir(), "Library", "Caches", "cthulhu-electron");
  }
  if (process.platform === "win32") {
    const local = process.env.LOCALAPPDATA ?? path.join(os.homedir(), "AppData", "Local");
    return path.join(local, "cthulhu-electron", "Cache");
  }
  return path.join(os.homedir(), ".cache", "cthulhu-electron");
}

// 必须在 ready 前设置，Chromium 的 Cache/Code Cache/GPUCache 等
// 才会落到系统缓存目录而不是污染 Application Support。
app.setPath("sessionData", getCacheDir());

/** @type {import('node:child_process').ChildProcess | null} */
let backendProcess = null;

// 后端地址与令牌在本次应用会话内固定：引擎意外退出并自动重启后，
// 渲染进程无需重载也能用同一地址/令牌重新连上恢复的引擎。
let backendPort = null;
let backendAuthToken = null;
let quitting = false;
let restartAttempts = 0;
let restartTimer = null;
let watchdogTimer = null;
let backendFailures = 0;
/** 「探针失败且没有任务进展」的起始时刻：观察期从这里算。 */
let suspiciousSince = null;
/** 下次拉起引擎时告诉它「上次为什么重启」，后端会写进续跑任务的说明里。 */
let restartCause = null;
/** 上次是不是原生崩溃（信号/非零退出）：后端据此把推理设备降到 CPU。 */
let restartAbnormal = false;
/** 是否是守护进程自己发起的重启（此时退出信号不能算「原生崩溃」）。 */
let engineKillRequested = false;
/** 诊断文件位置：崩溃日志与心跳文件都落在用户数据目录（开发模式在缓存目录）。 */
let diagnosticsDir = null;

/** 诊断目录与心跳/崩溃日志路径（首次调用时创建目录）。 */
function getDiagnostics() {
  if (diagnosticsDir === null) {
    diagnosticsDir = IS_DEV ? getCacheDir() : app.getPath("userData");
    fs.mkdirSync(diagnosticsDir, { recursive: true });
  }
  return {
    dir: diagnosticsDir,
    crashLog: path.join(diagnosticsDir, "backend-crash.log"),
    heartbeat: path.join(diagnosticsDir, "engine-heartbeat.json"),
  };
}

/** 把引擎生命周期事件追加进崩溃日志，现场出问题时能直接看原因与时间。 */
function logEngineEvent(message) {
  try {
    fs.appendFileSync(
      getDiagnostics().crashLog,
      `\n----- ${new Date().toISOString()} ${message} -----\n`,
    );
  } catch (error) {
    console.error("[backend] 诊断日志写入失败：", error.message);
  }
}

/** 读后端写的心跳文件（由独立线程维护，事件循环卡住也能反映真实状态）。 */
function readHeartbeat(nowMs) {
  try {
    const { heartbeat } = getDiagnostics();
    const stat = fs.statSync(heartbeat);
    return parseHeartbeat(fs.readFileSync(heartbeat, "utf8"), stat.mtimeMs, nowMs);
  } catch {
    return null;
  }
}

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

/** 本次应用会话内固定后端的端口与鉴权令牌（首次启动时生成）。 */
async function resolveBackendConfig() {
  if (backendPort !== null && backendAuthToken !== null) {
    return { port: backendPort, authToken: backendAuthToken };
  }
  if (IS_DEV) {
    backendPort = BACKEND_PORT;
    backendAuthToken = DEV_AUTH_TOKEN;
  } else {
    backendPort = await getFreePort();
    backendAuthToken = randomBytes(32).toString("hex");
  }
  return { port: backendPort, authToken: backendAuthToken };
}

/** 以独立进程组拉起后端：守护重启或强杀时连同 ffmpeg 子进程一起回收。 */
function spawnBackend(command, args, options) {
  return spawn(command, args, {
    ...options,
    detached: process.platform !== "win32",
  });
}

function onBackendExit() {
  const exitedPid = backendProcess?.pid;
  const exitCode = backendProcess?.exitCode;
  const signalCode = backendProcess?.signalCode;
  console.log(`[backend] 退出，code=${exitCode ?? "unknown"} signal=${signalCode ?? "-"}`);
  backendProcess = null;
  // 原生崩溃时 ffmpeg 子进程可能残留，连同旧进程组一并回收，避免孤儿编码。
  if (process.platform !== "win32" && exitedPid) {
    try {
      process.kill(-exitedPid, "SIGKILL");
    } catch {
      // 进程组已不存在，忽略
    }
  }
  if (!quitting) {
    const detail = signalCode ? `signal=${signalCode}` : `code=${exitCode ?? "unknown"}`;
    if (!engineKillRequested) {
      // 自己崩掉的（信号/非零退出码）才可能是原生崩溃，才值得降级推理设备。
      restartCause = `引擎崩溃退出（${detail}）`;
      restartAbnormal = signalCode != null || (exitCode ?? 0) !== 0;
      logEngineEvent(`引擎进程意外退出：${detail}`);
    } else {
      restartAbnormal = false;
      logEngineEvent(`引擎被强制重启后退出：${detail}`);
    }
    engineKillRequested = false;
  }
  scheduleBackendRestart();
}

function onBackendError(error) {
  console.error("[backend] 启动错误：", error.message);
  backendProcess = null;
  if (!quitting) {
    restartCause = "引擎启动失败";
    logEngineEvent(`引擎启动失败：${error.message}`);
  }
  scheduleBackendRestart();
}

/** 后端意外退出后自动拉起：指数退避，恢复健康后由看门狗归零重试计数。 */
function scheduleBackendRestart() {
  if (quitting || restartTimer !== null) return;
  const delay = restartDelayMs(restartAttempts);
  restartAttempts += 1;
  console.log(`[backend] ${Math.round(delay / 1000)} 秒后自动重启引擎`);
  restartTimer = setTimeout(async () => {
    restartTimer = null;
    if (quitting) return;
    try {
      await startBackend();
    } catch (error) {
      console.error("[backend] 自动重启失败：", error.message);
      scheduleBackendRestart();
    }
  }, delay);
}

function terminateBackend(signal = "SIGTERM") {
  const proc = backendProcess;
  if (!proc || proc.exitCode !== null) return;
  try {
    if (process.platform === "win32") {
      proc.kill(signal);
    } else {
      process.kill(-proc.pid, signal);
    }
  } catch {
    // 进程/进程组可能刚好退出，忽略
  }
}

function killBackend() {
  terminateBackend("SIGTERM");
  const proc = backendProcess;
  if (!proc || proc.exitCode !== null) return;
  // 兜底：SIGTERM 后仍未退出则强杀，避免僵尸进程占用端口。
  const forceKillTimer = setTimeout(() => {
    if (backendProcess === proc && proc.exitCode === null) {
      terminateBackend("SIGKILL");
    }
  }, FORCE_KILL_DELAY_MS);
  forceKillTimer.unref?.();
}

/** 周期性健康检查：只有「长期无响应且确实没有任务进展」才强制重启。 */
function startWatchdog(port) {
  if (watchdogTimer !== null) return;
  watchdogTimer = setInterval(async () => {
    if (quitting || backendProcess === null || backendProcess.exitCode !== null) return;
    let healthy = false;
    try {
      const res = await fetch(`http://127.0.0.1:${port}/api/health`, {
        signal: AbortSignal.timeout(HEALTH_PROBE_TIMEOUT_MS),
      });
      healthy = res.ok;
    } catch {
      healthy = false;
    }
    if (healthy) {
      backendFailures = 0;
      restartAttempts = 0;
      suspiciousSince = null;
      return;
    }
    backendFailures += 1;
    const nowMs = Date.now();
    const heartbeat = readHeartbeat(nowMs);
    const working = isEngineWorking(heartbeat, nowMs);
    // 清洗一条视频要几十分钟：只要还在推进就绝不打扰，哪怕探针全超时。
    if (working) {
      suspiciousSince = null;
      console.warn(
        `[backend] 健康检查超时（第 ${backendFailures} 次），但任务仍在推进，继续等待`,
      );
      return;
    }
    if (suspiciousSince === null) suspiciousSince = nowMs;
    const verdict = decideEngineRestart({
      probeFailures: backendFailures,
      suspiciousSinceMs: suspiciousSince,
      working: false,
      heartbeatAgeMs: heartbeat?.ageMs ?? null,
      nowMs,
    });
    console.warn(
      `[backend] 健康检查连续失败 ${backendFailures}/${WATCHDOG_FAILURE_THRESHOLD}：${verdict.reason || "观察中"}`,
    );
    if (verdict.restart) {
      backendFailures = 0;
      suspiciousSince = null;
      console.error("[backend] 引擎长时间无响应，强制重启以恢复任务调度");
      restartCause = verdict.reason;
      engineKillRequested = true;
      logEngineEvent(`强制重启引擎：${verdict.reason}`);
      killBackend();
    }
  }, WATCHDOG_INTERVAL_MS);
}

async function startBackend() {
  const { port, authToken } = await resolveBackendConfig();
  console.log("[backend] 启动模式：", IS_DEV ? "dev" : "packaged", "resources:", process.resourcesPath);
  const { crashLog, heartbeat } = getDiagnostics();
  // 心跳与「上次重启原因」都交给后端：看门狗据此区分「忙」与「死」，
  // 续跑任务也能把原因写进自己的说明里。
  const diagnosticsEnv = {
    CTHULHU_CRASH_LOG: crashLog,
    CTHULHU_ENGINE_HEARTBEAT: heartbeat,
    CTHULHU_RESTART_CAUSE: restartCause ?? "",
    CTHULHU_RESTART_ABNORMAL: restartAbnormal ? "1" : "",
  };
  if (IS_DEV) {
    const backendDir = path.resolve(__dirname, "..", "backend");
    backendProcess = spawnBackend(
      "uv",
      ["run", "--extra", "purify", "uvicorn", "cthulhu_backend.main:app", "--port", String(port)],
      {
        cwd: backendDir,
        stdio: "inherit",
        env: {
          ...process.env,
          ...diagnosticsEnv,
          CTHULHU_AUTH_TOKEN: authToken,
          CTHULHU_FFMPEG_INSTALL_DIR: path.join(getCacheDir(), "ffmpeg"),
          ...(fs.existsSync(path.resolve(__dirname, "..", "packaging", "models"))
            ? {
                CTHULHU_PURIFY_MODEL_DIR: path.resolve(
                  __dirname,
                  "..",
                  "packaging",
                  "models",
                ),
              }
            : {}),
        },
      },
    );
  } else {
    // 生产模式：启动打包后的 Python sidecar，并由其后端托管前端静态资源。
    // extraResources 拷贝的是目录内容，可执行文件即 backend/<name>。
    const backendExeName = process.platform === "win32" ? "cthulhu-backend.exe" : "cthulhu-backend";
    const backendExe = path.join(process.resourcesPath, "backend", backendExeName);
    const staticDir = path.join(process.resourcesPath, "backend", "ui-dist");
    const ffmpegDir = path.join(process.resourcesPath, "ffmpeg");
    const ffmpegBinary = path.join(ffmpegDir, process.platform === "win32" ? "ffmpeg.exe" : "ffmpeg");
    const deepModel = path.join(process.resourcesPath, "backend", "models", "clip-vit-b32-vision-fp16.onnx");
    const userData = app.getPath("userData");
    const cacheDir = getCacheDir();
    const env = {
      ...process.env,
      ...diagnosticsEnv,
      CTHULHU_STATIC_DIR: staticDir,
      // 数据写入用户目录，避免向只读的 .app 包内写库与演示素材。
      CTHULHU_DB: path.join(userData, "cthulhu.db"),
      CTHULHU_THUMB_CACHE: path.join(cacheDir, "thumb-cache"),
      CTHULHU_LIBRARY_DIR: path.join(userData, "library"),
      CTHULHU_DEEP_MODEL: deepModel,
      CTHULHU_AUTH_TOKEN: authToken,
      CTHULHU_PORT: String(port),
      CTHULHU_FFMPEG_INSTALL_DIR: path.join(cacheDir, "ffmpeg"),
    };
    if (fs.existsSync(ffmpegBinary)) {
      env.CTHULHU_FFMPEG_DIR = ffmpegDir;
    }
    backendProcess = spawnBackend(backendExe, [], {
      env,
      stdio: "inherit",
    });
    console.log("[backend] pid:", backendProcess.pid, "exe:", backendExe);
  }
  // 原因只对紧随其后的这一次启动有意义，避免长期挂在进程里。
  restartCause = null;
  restartAbnormal = false;
  backendProcess.once("exit", onBackendExit);
  backendProcess.once("error", onBackendError);
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
  const { port, authToken } = await resolveBackendConfig();
  if (backendProcess === null) {
    await startBackend();
  }
  await waitForBackend(port);
  startWatchdog(port);
  const backendUrl = `http://127.0.0.1:${port}`;

  const win = new BrowserWindow({
    width: 1440,
    height: 900,
    minWidth: 1280,
    minHeight: 720,
    title: "Cthulhu · 推广素材处理台",
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
  quitting = true;
  if (restartTimer !== null) {
    clearTimeout(restartTimer);
    restartTimer = null;
  }
  if (watchdogTimer !== null) {
    clearInterval(watchdogTimer);
    watchdogTimer = null;
  }
  terminateBackend();
});
