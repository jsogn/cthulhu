import { useEffect, useRef, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import {
  ffmpegInstallState,
  getHealth,
  getSettings,
  installFfmpeg,
  saveSettings,
  selectFfmpeg,
  type FfmpegInstallState,
} from "@/lib/backend";
import { resolveTheme, useAppStore } from "@/stores/app";
import { toast } from "@/stores/toasts";

const DEFAULTS: Record<string, string> = {
  export_dir: "~/导出/Cthulhu",
  naming: "原文件名 + 时间戳",
  parallelism: "2",
  gpu: "仅 CPU（软件编码）",
  transform_strategy: "fast",
  preset: "veryfast",
  temp_dir: "/tmp/watermark-cleaner",
};

export default function SettingsView() {
  const themePref = useAppStore((state) => state.themePref);
  const setThemePref = useAppStore((state) => state.setThemePref);
  const followSystem = themePref === "system";
  const [form, setForm] = useState<Record<string, string>>({ ...DEFAULTS });
  const [ffmpegOk, setFfmpegOk] = useState<boolean | null>(null);
  const [install, setInstall] = useState<FfmpegInstallState | null>(null);
  const pollRef = useRef<number | null>(null);

  useEffect(() => {
    void getSettings()
      .then((saved) =>
        setForm((current) => {
          const merged = {
            ...current,
            ...Object.fromEntries(
              Object.entries(saved).map(([key, value]) => [key, String(value)]),
            ),
          };
          // 旧版本的 GPU 选项名已废弃，统一归一到当前两个语义。
          if (!["自动（H.265 硬编加速）", "仅 CPU（软件编码）"].includes(merged.gpu)) {
            merged.gpu = "自动（H.265 硬编加速）";
          }
          return merged;
        }),
      )
      .catch(() => undefined);
  }, []);

  const refreshEngine = async () => {
    try {
      const health = await getHealth();
      setFfmpegOk(health.ok && health.ffmpeg);
    } catch {
      setFfmpegOk(false);
    }
  };

  useEffect(() => {
    void refreshEngine();
    return () => {
      if (pollRef.current !== null) window.clearInterval(pollRef.current);
    };
  }, []);

  const setValue = (key: string, value: string) => setForm((current) => ({ ...current, [key]: value }));

  const save = async () => {
    try {
      await saveSettings(form);
      toast("设置已保存，重启后仍生效");
    } catch {
      toast("保存失败，请确认引擎在线");
    }
  };

  const startInstall = async () => {
    try {
      const state = await installFfmpeg();
      setInstall(state);
      if (state.status === "downloading") {
        pollRef.current = window.setInterval(() => {
          void ffmpegInstallState()
            .then((latest) => {
              setInstall(latest);
              if (latest.status === "installed" || latest.status === "failed") {
                if (pollRef.current !== null) window.clearInterval(pollRef.current);
                pollRef.current = null;
                void refreshEngine();
                if (latest.status === "installed") {
                  toast("视频处理引擎已安装完成");
                } else {
                  toast("安装失败，请检查网络后重试", latest.detail);
                }
              }
            })
            .catch(() => undefined);
        }, 2000);
      }
    } catch (error) {
      toast(error instanceof Error ? error.message : "启动安装失败");
    }
  };

  const chooseLocal = async () => {
    const picker = window.appEnv?.chooseFile;
    if (!picker) {
      toast("选择本机文件仅在桌面版可用");
      return;
    }
    const path = await picker();
    if (!path) return;
    try {
      const result = await selectFfmpeg(path);
      await refreshEngine();
      toast(
        result.available
          ? "视频处理引擎已启用"
          : "该文件缺少配套组件，请选择名为 ffmpeg 的可执行文件",
      );
    } catch (error) {
      toast(error instanceof Error ? error.message : "启用失败");
    }
  };

  const chooseExportDir = async () => {
    const picker = window.appEnv?.chooseFolder;
    if (!picker) {
      toast("选择目录仅在桌面版可用");
      return;
    }
    const path = await picker();
    if (!path) return;
    setValue("export_dir", path);
  };

  const installing = install?.status === "downloading";
  const engineDesc =
    ffmpegOk === null
      ? "正在检查…"
      : ffmpegOk
      ? "已就绪，视频检测、清洗与修复均可正常使用"
        : "尚未安装，可自动安装或指定本机已有组件";

  return (
    <>
      <div className="view-head">
        <div>
          <div className="view-title">设置</div>
          <div className="view-desc">默认行为与硬件资源策略（持久化保存）</div>
        </div>
        <Button onClick={save}>保存设置</Button>
      </div>

      <div className="view-body">
        <div className="form-grid">
          <div className="form-field full">
            <Label>默认导出目录</Label>
            <div className="flex gap-2">
              <Input
                className="form-input flex-1"
                value={form.export_dir}
                onChange={(e) => setValue("export_dir", e.target.value)}
              />
              <Button
                variant="secondary"
                size="sm"
                style={{ height: 34 }}
                onClick={() => void chooseExportDir()}
              >
                选择目录
              </Button>
            </div>
            <div className="form-help">清洗产物默认存放目录；批量导出也复制到这里</div>
          </div>
          <div className="form-field">
            <Label>产物命名规则</Label>
            <Select value={form.naming} onValueChange={(v) => setValue("naming", v)}>
              <SelectTrigger className="form-input h-9 w-full">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="原文件名 + 时间戳">原文件名 + 时间戳（不覆盖）</SelectItem>
                <SelectItem value="时间戳 + 原文件名">时间戳 + 原文件名（不覆盖）</SelectItem>
              </SelectContent>
            </Select>
            <div className="form-help">清洗产物文件名格式，两种规则都不会覆盖已有产物</div>
          </div>
          <div className="form-field">
            <Label>批量并行度上限</Label>
            <Select value={form.parallelism} onValueChange={(v) => setValue("parallelism", v)}>
              <SelectTrigger className="form-input h-9 w-full">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {["1", "2", "3", "4"].map((n) => (
                  <SelectItem key={n} value={n}>
                    {n}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <div className="form-help">
              数值越高处理越快但占用资源越多；内存预算会按并行度自动均分
            </div>
          </div>
          <div className="form-field">
            <Label>处理管线</Label>
            <Select
              value={form.transform_strategy}
              onValueChange={(v) => setValue("transform_strategy", v)}
            >
              <SelectTrigger className="form-input h-9 w-full">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="fast">快速（推荐）</SelectItem>
                <SelectItem value="thorough">完整（原重方案，兜底）</SelectItem>
              </SelectContent>
            </Select>
            <div className="form-help">
              与完整管线效果相当，约快 2 倍
            </div>
          </div>
          <div className="form-field">
            <Label>编码速度档</Label>
            <Select value={form.preset} onValueChange={(v) => setValue("preset", v)}>
              <SelectTrigger className="form-input h-9 w-full">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="medium">标准（画质优先）</SelectItem>
                <SelectItem value="veryfast">快速（体积更小，检测效果相当）</SelectItem>
              </SelectContent>
            </Select>
            <div className="form-help">仅作用于 H.264 软件编码，H.265 硬编不受影响</div>
          </div>
          <div className="form-field">
            <Label>编码加速（H.265）</Label>
            <Select value={form.gpu} onValueChange={(v) => setValue("gpu", v)}>
              <SelectTrigger className="form-input h-9 w-full">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="自动（H.265 硬编加速）">
                  自动（H.265 硬编加速）
                </SelectItem>
                <SelectItem value="仅 CPU（软件编码）">仅 CPU（软件编码）</SelectItem>
              </SelectContent>
            </Select>
            <div className="form-help">
              实测 H.265 硬编约快 1.5 倍但体积约大 2 倍；H.264 软件编码综合更优
            </div>
          </div>
          <div className="form-field full">
            <Label>临时目录</Label>
            <Input
              className="form-input"
              value={form.temp_dir}
              onChange={(e) => setValue("temp_dir", e.target.value)}
              disabled
            />
            <div className="form-help">当前使用系统临时目录</div>
          </div>
          <div className="form-field full">
            <Label>界面主题</Label>
            <div className="switch switch-inline">
              <div>
                <div className="switch-label">{followSystem ? "跟随系统" : "手动模式"}</div>
                <div className="switch-desc">
                  {followSystem
                    ? "自动匹配系统外观（浅色 / 深色）"
                    : "已关闭跟随，可在顶部右侧手动切换主题"}
                </div>
              </div>
              <Switch
                checked={followSystem}
                onCheckedChange={(checked) => {
                  if (checked) {
                    setThemePref("system");
                    toast("主题已跟随系统");
                  } else {
                    setThemePref(resolveTheme(themePref));
                    toast("已切换为手动模式，可在右上角切换主题");
                  }
                }}
              />
            </div>
          </div>
          <div className="form-field full">
            <Label>运行环境</Label>
            <div className="env-card">
              <span
                className={ffmpegOk === null ? "env-dot" : ffmpegOk ? "env-dot ok" : "env-dot bad"}
              />
              <div className="env-main">
                <div className="env-title">视频处理引擎</div>
                <div className="env-desc">{engineDesc}</div>
              </div>
              <div className="env-actions">
                {ffmpegOk ? (
                  <Badge className="env-badge">已就绪</Badge>
                ) : (
                  <>
                    <Button
                      size="sm"
                      variant="secondary"
                      disabled={installing || ffmpegOk === null}
                      onClick={() => void startInstall()}
                    >
                      {installing ? "安装中…" : "自动安装"}
                    </Button>
                    <Button size="sm" variant="ghost" onClick={() => void chooseLocal()}>
                      选择本机文件
                    </Button>
                  </>
                )}
              </div>
            </div>
            {installing && (
              <div className="form-help">正在下载视频处理组件，完成后自动生效，请保持网络连接…</div>
            )}
            {install?.status === "failed" && (
              <div className="form-help text-destructive">
                安装失败：{install.detail || "网络异常"}，可重试或选择本机已有文件
              </div>
            )}
          </div>
        </div>
        <p className="settings-footnote">本工具仅用于处理你拥有或有使用权的视频，检测结果仅供参考。</p>
      </div>
    </>
  );
}
