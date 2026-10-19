// 时间与体积等展示格式化工具。

export function fmtSize(bytes: number): string {
  if (bytes < 1024) return `${bytes}B`;
  if (bytes < 1048576) return `${Math.round(bytes / 1024)}KB`;
  if (bytes < 1073741824) return `${(bytes / 1048576).toFixed(1)}MB`;
  return `${(bytes / 1073741824).toFixed(2)}GB`;
}

/** 帧率展示：整数帧率去掉浮点尾差，非整数保留三位内的有效小数。 */
export function fmtFps(fps: number | null | undefined): string {
  if (fps == null || !Number.isFinite(fps) || fps <= 0) return "—";
  const rounded = Math.round(fps);
  if (Math.abs(fps - rounded) < 0.01) return String(rounded);
  return fps.toFixed(3).replace(/\.?0+$/, "");
}

/** 播放时间展示：不足一小时用 MM:SS，超过一小时用 HH:MM:SS。 */
export function fmtClock(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return "00:00";
  const totalSec = Math.floor(seconds);
  const ss = totalSec % 60;
  const mm = Math.floor(totalSec / 60) % 60;
  const hh = Math.floor(totalSec / 3600);
  const p = (n: number) => String(n).padStart(2, "0");
  return hh > 0 ? `${p(hh)}:${p(mm)}:${p(ss)}` : `${p(mm)}:${p(ss)}`;
}
