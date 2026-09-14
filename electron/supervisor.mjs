// 后端守护策略的纯决策逻辑：与 Electron 生命周期解耦，便于单元测试。
//
// 策略约定：
// - 崩溃重启采用指数退避，1 秒起步、30 秒封顶；
// - 每 15 秒探测一次 /api/health，连续 3 次失败只算「可疑」，不是「该死」；
// - 判定强杀还要看后端自己写的心跳：只要还有任务在跑、且最长 PROGRESS_FRESH_MS
//   内有进展，就一律不动手——清洗一条视频要几十分钟，误杀一次就得从头再来；
// - 只有「探针持续失败 + 任务确实没进展」并持续 STALL_GRACE_MS 之后才强杀；
// - 强杀兜底在 SIGTERM 后 3 秒触发，避免僵尸进程占用端口。

export const RESTART_BASE_DELAY_MS = 1000;
export const RESTART_MAX_DELAY_MS = 30000;
export const HEALTH_PROBE_TIMEOUT_MS = 5000;
export const WATCHDOG_INTERVAL_MS = 15000;
export const WATCHDOG_FAILURE_THRESHOLD = 3;
export const FORCE_KILL_DELAY_MS = 3000;

/** 心跳文件超过这个时长没更新，说明整个进程（含工作线程）都卡住了。 */
export const HEARTBEAT_STALE_MS = 30000;
/** 「探针失败 + 无进展」持续到这个时长才允许强杀。 */
export const STALL_GRACE_MS = 180000;
/** 距上次任务进展不超过这个时长，就算「在干活」，不许打扰。 */
export const PROGRESS_FRESH_MS = 180000;

/** 第 attempts 次重启前的等待时长：指数退避并封顶。 */
export function restartDelayMs(
  attempts,
  base = RESTART_BASE_DELAY_MS,
  max = RESTART_MAX_DELAY_MS,
) {
  return Math.min(max, base * 2 ** attempts);
}

/**
 * 解析后端心跳文件：坏 JSON / 字段缺失一律当成「没有心跳」。
 *
 * @returns {{ageMs: number, activeTasks: number, progressAgeMs: number|null}|null}
 */
export function parseHeartbeat(text, mtimeMs, nowMs) {
  let payload;
  try {
    payload = JSON.parse(text);
  } catch {
    return null;
  }
  if (typeof payload !== "object" || payload === null || Array.isArray(payload)) return null;
  const activeTasks = Number.isFinite(payload.active_tasks) ? payload.active_tasks : 0;
  const progressAt = Number.isFinite(payload.progress_at) ? payload.progress_at : null;
  return {
    ageMs: Math.max(0, nowMs - mtimeMs),
    activeTasks: Math.max(0, activeTasks),
    progressAgeMs: progressAt === null ? null : Math.max(0, nowMs - progressAt * 1000),
  };
}

/** 引擎是否正在干活：有任务在跑且最近有进展。 */
export function isEngineWorking(heartbeat, nowMs, freshMs = PROGRESS_FRESH_MS) {
  if (!heartbeat || heartbeat.activeTasks <= 0) return false;
  if (heartbeat.progressAgeMs === null) return false;
  return heartbeat.progressAgeMs < freshMs;
}

/**
 * 是否该强制重启引擎，附带回给人看的原因。
 *
 * 三条硬约束（缺一不可）：探针连续失败达阈值、引擎没有在干活、
 * 这种「可疑」状态已经持续 STALL_GRACE_MS。
 *
 * @param {object} state
 * @param {number} state.probeFailures 连续失败次数
 * @param {number|null} state.suspiciousSinceMs 首次「失败且没在干活」的时间戳
 * @param {boolean} state.working 引擎是否正在推进任务
 * @param {number|null} state.heartbeatAgeMs 心跳文件距今毫秒（null=没有心跳）
 * @param {number} state.nowMs
 * @returns {{restart: boolean, reason: string, suspicious: boolean}}
 */
export function decideEngineRestart({
  probeFailures,
  suspiciousSinceMs,
  working,
  heartbeatAgeMs,
  nowMs,
  threshold = WATCHDOG_FAILURE_THRESHOLD,
  graceMs = STALL_GRACE_MS,
  heartbeatStaleMs = HEARTBEAT_STALE_MS,
}) {
  if (probeFailures < threshold) {
    return { restart: false, reason: "", suspicious: false };
  }
  if (working) {
    return { restart: false, reason: "任务仍在推进，继续等待", suspicious: false };
  }
  if (suspiciousSinceMs === null) {
    return { restart: false, reason: "首次无响应，开始观察", suspicious: true };
  }
  const stalledMs = nowMs - suspiciousSinceMs;
  if (stalledMs < graceMs) {
    return {
      restart: false,
      reason: `无进展 ${Math.round(stalledMs / 1000)} 秒，未达 ${Math.round(graceMs / 1000)} 秒观察期`,
      suspicious: true,
    };
  }
  const seconds = Math.round(stalledMs / 1000);
  const heartbeatGone = heartbeatAgeMs === null || heartbeatAgeMs > heartbeatStaleMs;
  return {
    restart: true,
    reason: heartbeatGone
      ? `引擎无响应且心跳停止 ${seconds} 秒`
      : `引擎无响应 ${seconds} 秒且任务无进展`,
    suspicious: true,
  };
}
