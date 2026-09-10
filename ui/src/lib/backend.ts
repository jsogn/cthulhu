// Python 后端的连接封装：REST 用于命令，WebSocket 用于进度事件。

import type { components } from "@/lib/api-types";

export const backendUrl = window.appEnv?.backendUrl ?? "http://127.0.0.1:57173";

/** 本地引擎访问令牌：Electron 注入优先，开发环境回退 Vite 注入的固定值。 */
const authToken = window.appEnv?.authToken ?? import.meta.env.VITE_AUTH_TOKEN ?? "";

/** 统一附带鉴权头的本地 API 请求。 */
async function apiFetch(path: string, init: RequestInit = {}): Promise<Response> {
  const headers = new Headers(init.headers);
  headers.set("X-CTHULHU-Token", authToken);
  try {
    return await fetch(`${backendUrl}${path}`, { ...init, headers });
  } catch {
    throw new Error("无法连接处理引擎，请稍后重试");
  }
}

/** 视频指定时间点的取帧地址。 */
export const frameUrl = (path: string, t: number) =>
  `${backendUrl}/api/frame?path=${encodeURIComponent(path)}&t=${t}&token=${encodeURIComponent(authToken)}`;

/** 视频文件的流式播放地址（支持 Range 拖动定位）。 */
export const mediaUrl = (path: string) =>
  `${backendUrl}/api/media?path=${encodeURIComponent(path)}&token=${encodeURIComponent(authToken)}`;

/** 素材封面缩略图地址（等比缩放为 JPEG，避免整帧 PNG 拖慢列表）。 */
export const thumbUrl = (path: string, width = 320) =>
  `${backendUrl}/api/thumb?path=${encodeURIComponent(path)}&width=${width}&token=${encodeURIComponent(authToken)}`;

export interface ImportedVideo {
  path: string;
  name: string;
  size: number;
  duplicate?: boolean;
  video?: LibraryVideoInfo | null;
}

/** 把浏览器拖入的 File 导入本地引擎素材库，返回可供播放/处理的路径。 */
export function importFile(
  file: File,
  onProgress?: (percent: number) => void,
): Promise<ImportedVideo> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${backendUrl}/api/import`);
    xhr.setRequestHeader("X-CTHULHU-Token", authToken);
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) {
        onProgress?.(Math.round((event.loaded / event.total) * 100));
      }
    };
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(JSON.parse(xhr.responseText) as ImportedVideo);
        } catch {
          reject(new Error("导入结果解析失败"));
        }
        return;
      }
      let detail = "导入失败";
      try {
        const data = JSON.parse(xhr.responseText) as { detail?: string };
        if (data.detail) detail = data.detail;
      } catch {
        // 保留默认提示
      }
      reject(new Error(`${detail}（${xhr.status}）`));
    };
    xhr.onerror = () => reject(new Error("导入失败，请检查引擎连接"));
    const form = new FormData();
    form.append("file", file);
    xhr.send(form);
  });
}

export interface HealthInfo {
  ok: boolean;
  version: string;
  ffmpeg: boolean;
  host?: {
    os: string;
    arch: string;
    cpu_count: number;
    memory_gb: number | null;
  };
}

/** 读取后端健康状态（含 FFmpeg 可用性）。 */
export async function getHealth(timeoutMs = 3000): Promise<HealthInfo> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await apiFetch("/api/health", { signal: controller.signal });
    const data = (await res.json()) as HealthInfo;
    return { ...data, ok: res.ok };
  } catch {
    return { ok: false, version: "", ffmpeg: false };
  } finally {
    clearTimeout(timer);
  }
}

export async function checkHealth(timeoutMs = 3000): Promise<boolean> {
  const health = await getHealth(timeoutMs);
  return health.ok;
}

export interface FfmpegInstallState {
  status: "idle" | "downloading" | "installed" | "failed";
  detail: string;
}

/** 用户手动指定本机已有的视频处理引擎可执行文件。 */
export async function selectFfmpeg(
  path: string,
): Promise<{ ffmpeg_dir: string; available: boolean }> {
  const res = await apiFetch("/api/ffmpeg/select", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path }),
  });
  if (!res.ok) {
    const data = await res.json().catch(() => null);
    throw new Error((data as { detail?: string } | null)?.detail ?? `指定失败（${res.status}）`);
  }
  return res.json();
}

/** 启动视频处理引擎的一键下载安装。 */
export async function installFfmpeg(): Promise<FfmpegInstallState> {
  const res = await apiFetch("/api/ffmpeg/install", { method: "POST" });
  if (!res.ok) throw new Error(`安装启动失败（${res.status}）`);
  return res.json();
}

/** 查询视频处理引擎一键安装进度。 */
export async function ffmpegInstallState(): Promise<FfmpegInstallState> {
  const res = await apiFetch("/api/ffmpeg/install");
  if (!res.ok) throw new Error(`查询安装状态失败（${res.status}）`);
  return res.json();
}

export interface PurifyStatus {
  available: boolean;
  reason: string;
  model_id: string;
  model_cached: boolean;
  bundled: boolean;
  variant: string | null;
  allow_download: boolean;
  device: string;
  state: "unavailable" | "missing" | "downloading" | "ready" | "error" | "unknown";
  progress: number;
  downloaded_bytes: number;
  total_bytes: number;
  error: string | null;
  model_dir: string;
  /** 潜空间引擎（10MB TAESD）：默认快档，内置即就绪。 */
  latent?: {
    engine: "latent";
    model_id: string;
    cached: boolean;
    bundled: boolean;
    path: string;
  };
}

/** 查询潜空间净化运行时与权重状态（依赖/权重缓存/设备）。 */
export async function purifyStatus(): Promise<PurifyStatus> {
  const res = await apiFetch("/api/purify/status");
  if (!res.ok) throw new Error(`查询净化状态失败（${res.status}）`);
  return res.json();
}

/** 触发净化权重准备（10MB TAESD）；返回当前状态，前端轮询直到 ready。 */
export async function installPurify(): Promise<PurifyStatus> {
  const res = await apiFetch("/api/purify/install", { method: "POST" });
  if (!res.ok) {
    const data = await res.json().catch(() => null);
    throw new Error(
      (data as { detail?: string } | null)?.detail ?? `净化模型安装启动失败（${res.status}）`,
    );
  }
  return res.json();
}

export interface CollusionResult {
  paths: string[];
  output: string;
  copies: number;
  frames: number;
  mode: "mean" | "median";
  fps: number | null;
  resolution: number[];
  quality: { psnr_db: number; ssim: number }[];
  estimated_watermark_reduction: number;
}

/** 共谋平均：同一内容的多份不同水印副本对齐后平均。 */
export async function runCollusion(
  paths: string[],
  output: string,
  mode: "mean" | "median" = "mean",
): Promise<CollusionResult> {
  const res = await apiFetch("/api/collusion", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ paths, output, mode }),
  });
  if (!res.ok) {
    const data = await res.json().catch(() => null);
    throw new Error(
      (data as { detail?: string } | null)?.detail ?? `共谋平均失败（${res.status}）`,
    );
  }
  return res.json();
}

export type BackendEvent = { type: string } & Record<string, unknown>;

// 检测引擎返回的报告结构（与后端 /api/detect 对齐）。
export interface DetectReport {
  probe: {
    codec?: string;
    width: number;
    height: number;
    fps: number;
    duration: number;
    format?: string;
  };
  container: {
    udta_boxes?: number;
    meta_boxes?: number;
    has_xmp?: boolean;
    n_trak?: number;
    suspicious?: string[];
  };
  sei_count: number | null;
  bitstream: {
    score: number;
    level: string;
    flags: string[];
    heuristic?: boolean;
  };
  blind?: {
    ss: number;
    qim: number;
    lsb: number;
    temporal?: number;
    dwt?: number;
    chroma?: number;
    echo?: number;
  } | null;
  blind_structural?: {
    dctmod: number;
    svd: number;
  };
  blind_confidence?: Record<string, number>;
  thresholds?: Record<string, number>;
  hits?: string[];
}

/** 清洗选项：由 OpenAPI 生成的请求契约派生（path/output 之外的字段）。 */
export type DesensitizeOptions = Partial<
  Omit<components["schemas"]["DesensitizeRequest"], "path" | "output">
>;

export interface AudioAnalysis {
  sample_rate: number;
  duration: number;
  echo_score: number;
  waveform: number[];
  spectrum: number[];
}

/** 分析视频音轨：波形、频谱与回声隐藏置信度。 */
export async function analyzeAudio(path: string): Promise<AudioAnalysis> {
  const res = await apiFetch("/api/audio/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path }),
  });
  if (!res.ok) {
    const data = await res.json().catch(() => null);
    throw new Error((data as { detail?: string } | null)?.detail ?? `音频分析失败（${res.status}）`);
  }
  return res.json();
}

export interface DesensitizeReport {
  input: string;
  output: string;
  frames: number;
  /** 成片帧数：回声清除等同步变速会改变时长，与输入帧数可能不同。 */
  out_frames?: number;
  similarity_before: { content_cosine: number; motion_cosine: number };
  similarity_after: { content_cosine: number; motion_cosine: number };
  vmaf: number | null;
  psnr_db?: number;
  ssim?: number;
  residual?: {
    ss: number | null;
    qim: number | null;
    lsb: number | null;
    echo: number | null;
  } | null;
}

/** 清洗任务完成后的完整结果（含画质校验指标）。 */
export interface DesensitizeJobResult {
  output: string;
  similarity_after?: { content_cosine: number };
  psnr_db?: number;
  ssim?: number;
  vmaf?: number | null;
  vmaf_aligned?: number | null;
  /** 指标遍转入后台，任务先完成、指标稍后回写。 */
  metrics_pending?: boolean;
}

export type JobKind = "detect" | "desensitize";

export interface JobTaskSpec {
  kind: JobKind;
  path: string;
  options?: Record<string, unknown>;
}

export interface JobTaskInfo {
  id: string;
  kind: JobKind;
  path: string;
  options?: Record<string, unknown>;
  status: "queued" | "running" | "paused" | "done" | "failed" | "canceled";
  percent: number;
  progress_note: string | null;
  elapsed: number | null;
  error: string | null;
  result: DetectReport | DesensitizeReport | null;
}

export interface JobInfo {
  id: string;
  name: string;
  parallelism: number;
  priority: number;
  status: "queued" | "running" | "paused" | "done" | "failed" | "canceled";
  created_at: number;
  tasks: JobTaskInfo[];
}

export interface ScanImportResult {
  directory: boolean;
  files: {
    path: string;
    name: string;
    size: number;
    video: {
      codec?: string;
      width: number;
      height: number;
      fps: number;
      duration: number;
      format?: string;
    } | null;
    error: string | null;
  }[];
  valid: number;
  invalid: number;
}

export interface TemplateInfo {
  id: string;
  name: string;
  payload: Record<string, unknown>;
  created_at: number;
}

/** 素材库文件的视频元信息（与后端 ffprobe 结果对齐）。 */
export interface LibraryVideoInfo {
  codec?: string;
  width: number;
  height: number;
  fps: number;
  duration: number;
}

/** 持久化素材库清单项。 */
export interface LibraryFile {
  path: string;
  name: string;
  size: number;
  video: LibraryVideoInfo | null;
  error: string | null;
  duplicate?: boolean;
  report?: DetectReport | null;
}

/** 建立事件连接，返回断开函数。 */
export function connectEvents(onMessage: (msg: BackendEvent) => void): () => void {
  const wsUrl =
    backendUrl.replace(/^http/, "ws") + `/ws/events?token=${encodeURIComponent(authToken)}`;
  let ws: WebSocket | null = null;
  let stopped = false;
  let retry = 0;
  let timer: number | undefined;

  // 后端重启或网络抖动导致断线时自动重连，避免任务进度永久停止刷新。
  const open = () => {
    if (stopped) return;
    ws = new WebSocket(wsUrl);
    ws.onopen = () => {
      retry = 0;
      onMessage({ type: "open" });
    };
    ws.onmessage = (event) => {
      try {
        onMessage(JSON.parse(event.data as string));
      } catch {
        // 忽略非 JSON 消息
      }
    };
    ws.onclose = () => {
      onMessage({ type: "close" });
      if (stopped) return;
      retry += 1;
      const delay = Math.min(8000, 500 * 2 ** (retry - 1));
      timer = window.setTimeout(open, delay);
    };
    ws.onerror = () => ws?.close();
  };

  open();
  return () => {
    stopped = true;
    if (timer !== undefined) window.clearTimeout(timer);
    ws?.close();
  };
}

export interface OutputInfo {
  kind: "cleaned" | "repaired";
  path: string;
  name: string;
  size: number;
  mtime: number;
}

export interface ProductInfo {
  path: string;
  name: string;
  kind: "cleaned" | "repaired";
  size: number;
  mtime: number;
  source: string;
  exists: boolean;
}

/** 列出源素材的全部处理产物，最新在前。 */
export async function fetchOutputs(path: string): Promise<{ source: string; outputs: OutputInfo[] }> {
  const res = await apiFetch(`/api/outputs?path=${encodeURIComponent(path)}`);
  if (!res.ok) throw new Error(`读取产物失败（${res.status}）`);
  return res.json();
}

/** 素材库各源素材的产物数量汇总。 */
export async function fetchOutputCounts(): Promise<Record<string, number>> {
  const res = await apiFetch("/api/outputs/counts");
  if (!res.ok) throw new Error(`读取产物数量失败（${res.status}）`);
  return res.json();
}

/** 删除指定处理产物（仅限清洗/修复产物）。 */
export async function deleteOutput(path: string): Promise<{ removed: number }> {
  const res = await apiFetch(`/api/outputs?path=${encodeURIComponent(path)}`, {
    method: "DELETE",
  });
  if (!res.ok) {
    const data = await res.json().catch(() => null);
    throw new Error((data as { detail?: string } | null)?.detail ?? `删除失败（${res.status}）`);
  }
  return res.json();
}

/** 全局产物清单：跨全部源素材汇总清洗/修复产物，含总数与总占用。 */
export async function listAllProducts(): Promise<{
  products: ProductInfo[];
  count: number;
  total_size: number;
}> {
  const res = await apiFetch("/api/products");
  if (!res.ok) throw new Error(`读取产物失败（${res.status}）`);
  return res.json();
}

/** 批量删除产物文件与记录；仅允许清洗/修复/候选命名，不触碰源视频。 */
export async function deleteProducts(paths: string[]): Promise<{ removed: number }> {
  const res = await apiFetch("/api/products/delete", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ paths }),
  });
  if (!res.ok) {
    const data = await res.json().catch(() => null);
    throw new Error((data as { detail?: string } | null)?.detail ?? `删除失败（${res.status}）`);
  }
  return res.json();
}

export async function enqueueJob(
  name: string,
  tasks: JobTaskSpec[],
  parallelism?: number,
): Promise<JobInfo> {
  const res = await apiFetch("/api/jobs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, tasks, ...(parallelism ? { parallelism } : {}) }),
  });
  if (!res.ok) throw new Error(`入队失败（${res.status}）`);
  return res.json();
}

/** 把指定源视频已生成的清洗/修复产物导出到设置中的导出目录。 */
export async function exportOutputs(
  paths: string[],
  exportDir?: string,
): Promise<{ exported: { source: string; dest: string }[]; missing: string[] }> {
  const res = await apiFetch("/api/export", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ paths, ...(exportDir ? { export_dir: exportDir } : {}) }),
  });
  if (!res.ok) throw new Error(`导出失败（${res.status}）`);
  return res.json();
}

export async function listJobs(): Promise<JobInfo[]> {
  const res = await apiFetch("/api/jobs");
  if (!res.ok) throw new Error(`获取任务失败（${res.status}）`);
  return res.json();
}

export async function cancelJob(jobId: string): Promise<JobInfo> {
  const res = await apiFetch(`/api/jobs/${jobId}/cancel`, { method: "POST" });
  if (!res.ok) throw new Error(`取消失败（${res.status}）`);
  return res.json();
}

export async function retryJob(jobId: string): Promise<JobInfo> {
  const res = await apiFetch(`/api/jobs/${jobId}/retry`, { method: "POST" });
  if (!res.ok) throw new Error(`重试失败（${res.status}）`);
  return res.json();
}

/** 暂停任务（运行中的任务完成当前项后停止分发）。 */
export async function pauseJob(jobId: string): Promise<JobInfo> {
  const res = await apiFetch(`/api/jobs/${jobId}/pause`, { method: "POST" });
  if (!res.ok) throw new Error(`暂停失败（${res.status}）`);
  return res.json();
}

/** 继续已暂停的任务。 */
export async function resumeJob(jobId: string): Promise<JobInfo> {
  const res = await apiFetch(`/api/jobs/${jobId}/resume`, { method: "POST" });
  if (!res.ok) throw new Error(`继续失败（${res.status}）`);
  return res.json();
}

/** 调整排队任务的优先级，数字越小越先执行。 */
export async function setJobPriority(jobId: string, priority: number): Promise<JobInfo> {
  const res = await apiFetch(`/api/jobs/${jobId}/priority`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ priority }),
  });
  if (!res.ok) throw new Error(`调整优先级失败（${res.status}）`);
  return res.json();
}

/** 递归扫描导入目标（文件或目录）。 */
export async function scanImportPath(path: string): Promise<ScanImportResult> {
  const res = await apiFetch("/api/import/scan", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path }),
  });
  if (!res.ok) {
    const data = await res.json().catch(() => null);
    throw new Error((data as { detail?: string } | null)?.detail ?? `扫描失败（${res.status}）`);
  }
  return res.json();
}

export async function getSettings(): Promise<Record<string, unknown>> {
  const res = await apiFetch("/api/settings");
  if (!res.ok) throw new Error(`读取设置失败（${res.status}）`);
  return res.json();
}

export async function saveSettings(settings: Record<string, unknown>): Promise<void> {
  const res = await apiFetch("/api/settings", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(settings),
  });
  if (!res.ok) throw new Error(`保存设置失败（${res.status}）`);
}

export async function listTemplates(): Promise<TemplateInfo[]> {
  const res = await apiFetch("/api/templates");
  if (!res.ok) throw new Error(`读取模板失败（${res.status}）`);
  return res.json();
}

export async function createTemplate(
  name: string,
  payload: Record<string, unknown>,
): Promise<TemplateInfo> {
  const res = await apiFetch("/api/templates", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, payload }),
  });
  if (!res.ok) throw new Error(`创建模板失败（${res.status}）`);
  return res.json();
}

/** 更新参数模板。 */
export async function updateTemplate(
  templateId: string,
  name: string,
  payload: Record<string, unknown>,
): Promise<TemplateInfo> {
  const res = await apiFetch(`/api/templates/${templateId}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, payload }),
  });
  if (!res.ok) throw new Error(`更新模板失败（${res.status}）`);
  return res.json();
}

/** 删除参数模板。 */
export async function deleteTemplate(templateId: string): Promise<void> {
  const res = await apiFetch(`/api/templates/${templateId}`, { method: "DELETE" });
  if (!res.ok) throw new Error(`删除模板失败（${res.status}）`);
}

export interface VariantInfo {
  id: string;
  source: string;
  output: string;
  options: Record<string, unknown>;
  seed: number;
  template_id: string | null;
  metrics: Record<string, unknown>;
  created_at: number;
}

/** 产物记录清单：按源文件过滤。 */
export async function listVariants(
  source?: string,
): Promise<VariantInfo[]> {
  const params = new URLSearchParams();
  if (source) params.set("source", source);
  const res = await apiFetch(`/api/variants?${params.toString()}`);
  if (!res.ok) throw new Error(`读取产物记录失败（${res.status}）`);
  return res.json();
}

/** 清空任务（scope: finished | all）。 */
export async function clearJobs(scope: "finished" | "all" = "finished"): Promise<number> {
  const res = await apiFetch(`/api/jobs?scope=${scope}`, { method: "DELETE" });
  if (!res.ok) throw new Error(`清空任务失败（${res.status}）`);
  const data = (await res.json()) as { removed: number };
  return data.removed;
}

/** 读取持久化素材库（含缓存的素材元数据，最新导入在前）。 */
export async function fetchLibrary(): Promise<LibraryFile[]> {
  const res = await apiFetch("/api/library");
  if (!res.ok) throw new Error(`读取素材库失败（${res.status}）`);
  const data = (await res.json()) as { files: LibraryFile[] };
  return data.files;
}

/** 把桌面端导入的本地视频登记进素材库，应用重启后仍会恢复显示。 */
export async function registerLibrary(path: string): Promise<LibraryFile> {
  const res = await apiFetch("/api/library", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path }),
  });
  if (!res.ok) {
    const data = await res.json().catch(() => null);
    throw new Error(
      (data as { detail?: string } | null)?.detail ?? `登记素材失败（${res.status}）`,
    );
  }
  return res.json();
}

/** 从素材库移除记录；源视频文件不受影响，由用户手动管理。 */
export async function unregisterLibrary(path: string): Promise<void> {
  const res = await apiFetch(`/api/library?path=${encodeURIComponent(path)}`, {
    method: "DELETE",
  });
  if (!res.ok) throw new Error(`移除素材失败（${res.status}）`);
}

/** 清空素材库：移除全部素材记录并删除对应产物（源视频不受影响）。 */
export async function clearLibrary(): Promise<{
  removed_materials: number;
  removed_products: number;
  removed_bytes: number;
}> {
  const res = await apiFetch("/api/library/all", { method: "DELETE" });
  if (!res.ok) throw new Error(`清空素材库失败（${res.status}）`);
  return res.json();
}
