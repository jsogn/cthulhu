// 后端守护策略的纯决策逻辑：与 Electron 生命周期解耦，便于单元测试。
//
// 策略约定：
// - 崩溃重启采用指数退避，1 秒起步、30 秒封顶；
// - 每 15 秒探测一次 /api/health，连续 3 次失败视为事件循环卡死，强制重启；
// - 强杀兜底在 SIGTERM 后 3 秒触发，避免僵尸进程占用端口。

export const RESTART_BASE_DELAY_MS = 1000;
export const RESTART_MAX_DELAY_MS = 30000;
export const HEALTH_PROBE_TIMEOUT_MS = 5000;
export const WATCHDOG_INTERVAL_MS = 15000;
export const WATCHDOG_FAILURE_THRESHOLD = 3;
export const FORCE_KILL_DELAY_MS = 3000;

/** 第 attempts 次重启前的等待时长：指数退避并封顶。 */
export function restartDelayMs(
  attempts,
  base = RESTART_BASE_DELAY_MS,
  max = RESTART_MAX_DELAY_MS,
) {
  return Math.min(max, base * 2 ** attempts);
}

/** 健康检查连续失败 failures 次时，是否应当强制重启引擎。 */
export function shouldForceRestart(failures, threshold = WATCHDOG_FAILURE_THRESHOLD) {
  return failures >= threshold;
}
