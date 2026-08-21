// Python 后端的连接封装：REST 用于命令，WebSocket 用于进度事件。

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

export interface UploadedVideo {
  path: string;
  name: string;
  size: number;
  video?: LibraryVideoInfo | null;
}

/** 把浏览器拖入的 File 上传到本地引擎素材库，返回可供播放/检测的路径。 */
export function uploadFile(
  file: File,
  onProgress?: (percent: number) => void,
): Promise<UploadedVideo> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${backendUrl}/api/upload`);
    xhr.setRequestHeader("X-CTHULHU-Token", authToken);
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) {
        onProgress?.(Math.round((event.loaded / event.total) * 100));
      }
    };
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(JSON.parse(xhr.responseText) as UploadedVideo);
        } catch {
          reject(new Error("上传结果解析失败"));
        }
        return;
      }
      let detail = "上传失败";
      try {
        const data = JSON.parse(xhr.responseText) as { detail?: string };
        if (data.detail) detail = data.detail;
      } catch {
        // 保留默认提示
      }
      reject(new Error(`${detail}（${xhr.status}）`));
    };
    xhr.onerror = () => reject(new Error("上传失败，请检查引擎连接"));
    const form = new FormData();
    form.append("file", file);
    xhr.send(form);
  });
}

export interface HealthInfo {
  ok: boolean;
  version: string;
  ffmpeg: boolean;
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
    echo?: number;
  } | null;
}

export interface DesensitizeOptions {
  reorder: boolean;
  speed: number;
  recrop: number;
  perturb: number;
  regrade: boolean;
  audioRemix: boolean;
  sharpness: boolean;
  colorRestore: boolean;
  denoise: boolean;
  antiReembed: boolean;
  seed: number;
  codec?: "libx264" | "libx265";
  lossless?: boolean;
  spoof?: boolean;
  bitrateKbps?: number;
  gop?: number;
  resolution?: string;
  fpsOut?: number;
  rotate?: number;
  phashAttack?: boolean;
  phashEpsilon?: number;
  phashIters?: number;
  multiHashAttack?: boolean;
  median?: number;
  noise?: number;
  requant?: number;
  dctStep?: number;
  dropEvery?: number;
  jitter?: number;
  perspective?: number;
  warp?: number;
  mirror?: boolean;
  chromaLevels?: number;
  subtractBeta?: number;
  transcodeChain?: boolean;
  saliency?: number;
  nativeFilters?: boolean;
  detailProtect?: number;
}

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
  order_disruption?: number;
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

export type JobKind = "detect" | "desensitize" | "repair";

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

export interface AuditEntry {
  id?: number;
  name: string;
  time: string;
  action: string;
  params: string;
  out: string;
  result: string;
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

/** 对本地视频文件执行检测（容器 / SEI / 压缩域联合分析）。 */
export async function runDetect(path: string): Promise<DetectReport> {
  const res = await apiFetch("/api/detect", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path }),
  });
  if (!res.ok) {
    const data = await res.json().catch(() => null);
    throw new Error((data as { detail?: string } | null)?.detail ?? `检测失败（${res.status}）`);
  }
  return res.json();
}

/** 对本地视频执行内容脱敏（清洗）变换。 */
export async function runDesensitize(
  path: string,
  output: string,
  options: DesensitizeOptions,
): Promise<DesensitizeReport> {
  const res = await apiFetch("/api/desensitize", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path, output, ...options }),
  });
  if (!res.ok) {
    const data = await res.json().catch(() => null);
    throw new Error((data as { detail?: string } | null)?.detail ?? `处理失败（${res.status}）`);
  }
  return res.json();
}

/** 生成原片与产物的并排预览拼图，返回图片本地路径。 */
export async function makePreview(
  path: string,
  output: string,
): Promise<{ image_path: string }> {
  const res = await apiFetch("/api/preview", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path, output }),
  });
  if (!res.ok) {
    const data = await res.json().catch(() => null);
    throw new Error((data as { detail?: string } | null)?.detail ?? `生成预览失败（${res.status}）`);
  }
  return res.json();
}

/** 预览拼图的鉴权图片地址。 */
export function previewImageUrl(path: string): string {
  return `${backendUrl}/api/preview-image?path=${encodeURIComponent(path)}&token=${encodeURIComponent(authToken)}`;
}

export interface CandidateInfo {
  index: number;
  output: string;
  seed: number;
  content_cosine: number;
  stability_ratio: number;
  dhash_reduction: number;
  order_disruption: number;
  export_health: {
    video_codec?: string;
    audio_codec?: string;
    width?: number;
    height?: number;
    size_bytes?: number;
  };
  score: number;
}

export interface CandidatesReport {
  source: string;
  output_dir: string;
  candidates: CandidateInfo[];
}

/** 同一素材生成多个差异化候选并按低损优选评分排序。 */
export async function generateCandidates(
  path: string,
  outputDir: string,
  count: number,
  options: DesensitizeOptions,
): Promise<CandidatesReport> {
  const res = await apiFetch("/api/candidates", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path, output_dir: outputDir, count, options }),
  });
  if (!res.ok) {
    const data = await res.json().catch(() => null);
    throw new Error((data as { detail?: string } | null)?.detail ?? `生成候选失败（${res.status}）`);
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
): Promise<{ exported: { source: string; dest: string }[]; missing: string[] }> {
  const res = await apiFetch("/api/export", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ paths }),
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

export async function listAudit(): Promise<AuditEntry[]> {
  const res = await apiFetch("/api/audit");
  if (!res.ok) throw new Error(`读取审计失败（${res.status}）`);
  return res.json();
}

export async function appendAudit(entry: AuditEntry): Promise<AuditEntry> {
  const res = await apiFetch("/api/audit", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(entry),
  });
  if (!res.ok) throw new Error(`写入审计失败（${res.status}）`);
  return res.json();
}

/** 更新审计记录的结果状态（如排队中 → 成功/失败）。 */
export async function updateAudit(auditId: number, result: string): Promise<void> {
  const res = await apiFetch(`/api/audit/${auditId}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ result }),
  });
  if (!res.ok) throw new Error(`更新审计失败（${res.status}）`);
}

/** 清空任务（scope: finished | all）。 */
export async function clearJobs(scope: "finished" | "all" = "finished"): Promise<number> {
  const res = await apiFetch(`/api/jobs?scope=${scope}`, { method: "DELETE" });
  if (!res.ok) throw new Error(`清空任务失败（${res.status}）`);
  const data = (await res.json()) as { removed: number };
  return data.removed;
}

/** 清空审计（处理历史）。 */
export async function clearAudit(): Promise<number> {
  const res = await apiFetch("/api/audit", { method: "DELETE" });
  if (!res.ok) throw new Error(`清空历史失败（${res.status}）`);
  const data = (await res.json()) as { removed: number };
  return data.removed;
}

/** 读取持久化素材库（含缓存的检测报告，最新导入在前）。 */
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

/** 从素材库移除记录；网页上传的库内副本会删除，桌面源文件不受影响。 */
export async function unregisterLibrary(path: string): Promise<void> {
  const res = await apiFetch(`/api/library?path=${encodeURIComponent(path)}`, {
    method: "DELETE",
  });
  if (!res.ok) throw new Error(`移除素材失败（${res.status}）`);
}
