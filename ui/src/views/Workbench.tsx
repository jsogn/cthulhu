import { useEffect, useRef, useState } from "react";
import type { MutableRefObject } from "react";
import type { KeyboardEvent as ReactKeyboardEvent, PointerEvent as ReactPointerEvent } from "react";
import {
  Code2,
  GitCompareArrows,
  MoreVertical,
  Pause,
  Play,
  Plus,
  Redo2,
  ScanSearch,
  SkipBack,
  SkipForward,
  Trash2,
  Undo2,
  Upload,
  X,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Switch } from "@/components/ui/switch";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { fmtFrames, fmtSize, nowStr } from "@/lib/format";
import {
  analyzeAudio,
  deleteOutput,
  enqueueJob,
  exportOutputs,
  fetchOutputs,
  frameUrl,
  generateCandidates,
  getSettings,
  listVariants,
  listTemplates,
  mediaUrl,
  thumbUrl,
  type AudioAnalysis,
  type CandidateInfo,
  type DetectReport,
  type DesensitizeOptions,
  type JobInfo,
  type OutputInfo,
  type TemplateInfo,
  type VariantInfo,
} from "@/lib/backend";
import { payloadOf, type TemplatePayload } from "@/lib/templates";
import { cn } from "@/lib/utils";
import { useAppStore } from "@/stores/app";
import { useHistoryStore } from "@/stores/history";
import { useMaterialsStore, type Material, type RiskLevel } from "@/stores/materials";
import { useQueueStore } from "@/stores/queue";
import { useRegionsStore } from "@/stores/regions";
import { openInFolder, toast } from "@/stores/toasts";

const RISK_OPTIONS: ("全部" | RiskLevel)[] = [
  "全部",
  "有疑似特征",
  "未检出异常",
  "待检测",
];
const TAB_KEYS = ["清洗去重", "检测参考", "处理产物", "水印区域"] as const;
const FRAME_MAX = 540;

/** 正在排队 / 执行 / 暂停中的检测任务所覆盖的路径（用于防止重复提交）。 */
function pendingDetectPaths(jobs: JobInfo[]): Set<string> {
  return new Set(
    jobs
      .flatMap((job) => job.tasks)
      .filter(
        (task) =>
          task.kind === "detect" &&
          (task.status === "queued" || task.status === "running" || task.status === "paused"),
      )
      .map((task) => task.path),
  );
}

function pendingCleanPaths(jobs: JobInfo[]): Set<string> {
  return new Set(
    jobs
      .flatMap((job) => job.tasks)
      .filter(
        (task) =>
          task.kind === "desensitize" &&
          (task.status === "queued" || task.status === "running" || task.status === "paused"),
      )
      .map((task) => task.path),
  );
}

const SNAKE_OPTION_KEYS: Record<string, string> = {
  audioRemix: "audio_remix",
  colorRestore: "color_restore",
  antiReembed: "anti_reembed",
  bitrateKbps: "bitrate_kbps",
  fpsOut: "fps_out",
  phashAttack: "phash_attack",
  phashEpsilon: "phash_epsilon",
  phashIters: "phash_iters",
  multiHashAttack: "multi_hash_attack",
  dctStep: "dct_step",
  dropEvery: "drop_every",
  chromaLevels: "chroma_levels",
  subtractBeta: "subtract_beta",
  transcodeChain: "transcode_chain",
  nativeFilters: "native_filters",
  detailProtect: "detail_protect",
  shotRetime: "shot_retime",
  cutJitter: "cut_jitter",
  audioStrong: "audio_strong",
};

function toSnakeOptions(options: DesensitizeOptions): Record<string, unknown> {
  return Object.fromEntries(
    Object.entries(options)
      .filter(([, value]) => value !== undefined && value !== "")
      .map(([key, value]) => [SNAKE_OPTION_KEYS[key] ?? key, value]),
  );
}

const CLEAN_LEVELS = ["轻度", "平衡", "深度"] as const;
type CleanLevel = (typeof CLEAN_LEVELS)[number];

const LEVEL_PRESETS: Record<CleanLevel, { retime: number; perturb: number }> = {
  轻度: { retime: 25, perturb: 15 },
  平衡: { retime: 30, perturb: 20 },
  深度: { retime: 40, perturb: 30 },
};

/** 三档清洗的真实行为映射（与后端参数一一对应，不展示虚假指标）。 */
const levelDisplay = (level: CleanLevel, recropOn: boolean) => {
  const { retime, perturb } = LEVEL_PRESETS[level];
  const speed = Math.max(0.85, 1 - 0.15 * (retime / 100));
  const gamma = 0.03 + 0.2 * (perturb / 100);
  const brightness = 0.02 + 0.06 * (perturb / 100);
  const crop = recropOn ? 0.015 + 0.075 * (perturb / 100) : 0;
  return { speed, gamma, brightness, crop, retime };
};

const OUTPUT_KIND_LABEL: Record<OutputInfo["kind"], string> = {
  cleaned: "清洗",
  repaired: "修复",
  candidate: "候选",
};

function outputTimeLabel(mtime: number): string {
  const date = new Date(mtime * 1000);
  const pad = (value: number) => String(value).padStart(2, "0");
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${pad(
    date.getHours(),
  )}:${pad(date.getMinutes())}`;
}

// 指纹对抗档：几何去同步 + pHash 签名扰动 + 底层载荷攻击原语组合。
type AntiPreset = {
  rotate: number;
  epsilon: number;
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
  jointAttack?: boolean;
  saliency?: number;
  nativeFilters?: boolean;
  detailProtect?: number;
  shotRetime?: boolean;
  cutJitter?: number;
  audioStrong?: boolean;
};

const ANTI_PRESETS: Record<string, AntiPreset | undefined> = {
  关闭: undefined,
  轻度: { rotate: 1.0, epsilon: 0.03 },
  标准: {
    rotate: 1.4,
    epsilon: 0.03,
    dctStep: 4,
    requant: 96,
    chromaLevels: 128,
    dropEvery: 13,
    noise: 0.006,
    jitter: 0.006,
    nativeFilters: true,
    shotRetime: true,
    cutJitter: 2,
    audioStrong: true,
  },
  强力: {
    rotate: 1.8,
    epsilon: 0.04,
    dctStep: 6,
    requant: 64,
    chromaLevels: 96,
    dropEvery: 9,
    noise: 0.01,
    jitter: 0.01,
    perspective: 0.004,
    saliency: 1,
    jointAttack: true,
    nativeFilters: true,
    shotRetime: true,
    cutJitter: 3,
    audioStrong: true,
  },
  全兵器: {
    rotate: 2.2,
    epsilon: 0.05,
    dctStep: 12,
    requant: 32,
    chromaLevels: 32,
    dropEvery: 7,
    noise: 0.02,
    jitter: 0.01,
    perspective: 0.005,
    warp: 0.003,
    subtractBeta: 1.2,
    transcodeChain: true,
    jointAttack: true,
    saliency: 3,
    nativeFilters: true,
    shotRetime: true,
    cutJitter: 4,
    audioStrong: true,
  },
};

function riskLabel(risk: RiskLevel): string {
  return risk;
}

function riskClass(risk: RiskLevel): string {
  return risk === "有疑似特征" ? "suspect" : risk === "未检出异常" ? "clear" : "pending";
}

export default function Workbench() {
  const materials = useMaterialsStore((state) => state.materials);
  const activeId = useMaterialsStore((state) => state.activeId);

  const active = materials.find((m) => m.id === activeId) ?? null;
  const [tab, setTab] = useState<(typeof TAB_KEYS)[number]>("清洗去重");

  const [frame, setFrame] = useState(132);
  const [playing, setPlaying] = useState(false);
  const [comparePct, setComparePct] = useState(50);
  const [compareLeft, setCompareLeft] = useState<string>("");
  const [compareRight, setCompareRight] = useState<string>("");
  const [compareLeftLabel, setCompareLeftLabel] = useState("原片");
  const [compareRightLabel, setCompareRightLabel] = useState("产物");
  const cleanOptionsRef = useRef<DesensitizeOptions | null>(null);
  const frameRef = useRef(frame);
  frameRef.current = frame;

  const startCompare = (
    left: string,
    right: string,
    leftLabel: string,
    rightLabel: string,
  ) => {
    setCompareLeft(left);
    setCompareRight(right);
    setCompareLeftLabel(leftLabel);
    setCompareRightLabel(rightLabel);
  };

  const exitCompare = () => {
    setCompareLeft("");
    setCompareRight("");
  };

  return (
    <section className="workbench">
      <div className="workbench-main">
        <MaterialPane />
        <PreviewPane
          material={active}
          frame={frame}
          setFrame={setFrame}
          playing={playing}
          setPlaying={setPlaying}
          comparePct={comparePct}
          setComparePct={setComparePct}
          compareLeft={compareLeft}
          compareRight={compareRight}
          compareLeftLabel={compareLeftLabel}
          compareRightLabel={compareRightLabel}
          onExitCompare={exitCompare}
        />
        <ContextPanel
          tab={tab}
          setTab={setTab}
          onStartCompare={startCompare}
          cleanOptionsRef={cleanOptionsRef}
        />
      </div>
      <BatchPopup />
    </section>
  );
}

/* ============ 左栏：素材库 ============ */
function MaterialPane() {
  const materials = useMaterialsStore((state) => state.materials);
  const loadFailed = useMaterialsStore((state) => state.loadFailed);
  const activeId = useMaterialsStore((state) => state.activeId);
  const selected = useMaterialsStore((state) => state.selected);
  const riskFilter = useMaterialsStore((state) => state.riskFilter);
  const search = useMaterialsStore((state) => state.search);
  const select = useMaterialsStore((state) => state.select);
  const toggleSelected = useMaterialsStore((state) => state.toggleSelected);
  const setRiskFilter = useMaterialsStore((state) => state.setRiskFilter);
  const setSearch = useMaterialsStore((state) => state.setSearch);
  const toggleSelectAll = useMaterialsStore((state) => state.toggleSelectAll);
  const setImportOpen = useAppStore((state) => state.setImportOpen);
  const backendConnected = useAppStore((state) => state.backendConnected);
  const jobs = useQueueStore((state) => state.jobs);
  const [scanBusy, setScanBusy] = useState(false);

  const visible = materials.filter((m) => {
    if (riskFilter !== "全部" && m.risk !== riskFilter) return false;
    const kw = search.trim().toLowerCase();
    if (kw && !m.name.toLowerCase().includes(kw) && !m.tags.join(" ").toLowerCase().includes(kw)) {
      return false;
    }
    return true;
  });
  const allSelected = visible.length > 0 && visible.every((m) => selected[m.id]);
  const pendingPaths = pendingDetectPaths(jobs);

  const runScan = async () => {
    if (scanBusy) return;
    const targets = visible
      .filter(
        (m): m is Material & { path: string } => !!m.path && !pendingPaths.has(m.path),
      )
      .map((m) => ({ kind: "detect" as const, path: m.path }));
    if (!targets.length) {
      toast(
        visible.some((m) => m.path)
          ? "这些素材均已在检测中，请等待完成"
          : "没有可检测的素材，请先导入本地视频",
      );
      return;
    }
    setScanBusy(true);
    try {
      const job = await enqueueJob("批量检测", targets);
      useQueueStore.getState().applyEvent({ type: "job:state", job });
      toast(`已入队 ${targets.length} 个检测任务`);
    } catch {
      toast("入队失败，请确认引擎在线");
    } finally {
      setScanBusy(false);
    }
  };

  return (
    <aside className="pane pane-left">
      <div className="pane-head">
        <div>
          <div className="pane-title">素材库</div>
          <div className="pane-count">{visible.length} 个素材</div>
        </div>
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant="ghost"
              size="icon"
              aria-label="批量扫描暗水印"
              disabled={
                scanBusy ||
                visible.some((m) => m.path) &&
                visible.every((m) => !m.path || pendingPaths.has(m.path))
              }
              onClick={() => runScan()}
            >
              <ScanSearch />
            </Button>
          </TooltipTrigger>
          <TooltipContent>批量扫描暗水印</TooltipContent>
        </Tooltip>
      </div>

      <div className="pane-body">
        <div className="toolbar-row">
          <Button variant="default" size="sm" onClick={() => setImportOpen(true)}>
            <Upload />
            导入
          </Button>
          <Button variant="ghost" size="sm" onClick={() => toggleSelectAll(visible.map((m) => m.id))}>
            {allSelected ? "取消全选" : "全选"}
          </Button>
        </div>

        <Input
          className="search"
          type="search"
          placeholder="搜索文件名、标签…"
          aria-label="搜索素材"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />

        <div className="chips">
          {RISK_OPTIONS.map((option) => (
            <Badge
              key={option}
              asChild
              variant={riskFilter === option ? "secondary" : "outline"}
            >
              <button
                type="button"
                className={cn(
                  "cursor-pointer",
                  riskFilter === option &&
                    "border-primary bg-primary/10 text-primary hover:bg-primary/15",
                  !(riskFilter === option) && "hover:bg-muted/60 hover:text-foreground",
                )}
                onClick={() => setRiskFilter(option)}
              >
                {option === "全部" ? "全部" : riskLabel(option)}
              </button>
            </Badge>
          ))}
        </div>

        {materials.length === 0 ? (
          loadFailed ? (
            <div className="mat-empty">
              <p>素材库加载失败，请确认引擎在线</p>
              <Button
                variant="secondary"
                size="sm"
                onClick={() => void useMaterialsStore.getState().refreshLibrary()}
              >
                重试
              </Button>
            </div>
          ) : !backendConnected ? (
            <div className="mat-empty">
              <p>正在启动处理引擎…</p>
              <p className="mat-empty-hint">首次启动需要一点时间，请稍候</p>
            </div>
          ) : (
            <div className="mat-empty">
              <p>素材库还是空的</p>
              <Button variant="secondary" size="sm" onClick={() => setImportOpen(true)}>
                导入第一批素材
              </Button>
            </div>
          )
        ) : visible.length === 0 ? (
          <div className="mat-empty">没有匹配的素材，试试调整搜索或筛选条件</div>
        ) : (
          <ScrollArea className="min-h-0 flex-1">
            <div className="mat-list">
              {visible.map((m) => (
                <div
                  key={m.id}
                  className={cn("mat-item", m.id === activeId && "active")}
                  role="button"
                  tabIndex={0}
                  aria-label={`选择素材 ${m.name}`}
                  onClick={() => select(m.id)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault();
                      select(m.id);
                    }
                  }}
                >
                  <Checkbox
                    className="mat-check"
                    checked={!!selected[m.id]}
                    onCheckedChange={() => toggleSelected(m.id)}
                    aria-label="勾选此素材"
                    onClick={(e) => e.stopPropagation()}
                    onKeyDown={(e) => e.stopPropagation()}
                  />
                  <span className="mat-thumb">
                    <img
                      src={m.path ? thumbUrl(m.path) : m.frame}
                      loading="lazy"
                      alt=""
                    />
                  </span>
                  <span className="mat-info">
                    <span className="mat-name">{m.name}</span>
                    <span className="mat-meta mono">
                      {m.dur} · {m.res} · {m.fps} · {m.size}
                    </span>
                    <span className="mat-tags">
                      <span className={`tag ${riskClass(m.risk)}`}>{riskLabel(m.risk)}</span>
                      {(m.outputCount ?? 0) > 0 && (
                        <span className="tag tag-out">产物 {m.outputCount}</span>
                      )}
                    </span>
                  </span>
                </div>
              ))}
            </div>
          </ScrollArea>
        )}
      </div>
    </aside>
  );
}

/* ============ 批量操作弹窗 ============ */
function BatchPopup() {
  const selected = useMaterialsStore((state) => state.selected);
  const materials = useMaterialsStore((state) => state.materials);
  const deleteSelected = useMaterialsStore((state) => state.deleteSelected);
  const toggleSelectAll = useMaterialsStore((state) => state.toggleSelectAll);
  const [exportDir, setExportDir] = useState("");
  const ids = Object.keys(selected);

  useEffect(() => {
    void getSettings()
      .then((settings) => setExportDir((settings.export_dir as string) ?? ""))
      .catch(() => undefined);
  }, []);

  if (!ids.length) return null;

  const batchExport = async () => {
    const paths = ids
      .map((id) => materials.find((m) => m.id === id))
      .filter((m) => !!m?.path)
      .map((m) => m!.path as string);
    if (!paths.length) {
      toast("所选素材缺少本地路径，无法导出");
      return;
    }
    let destDir = exportDir || "~/导出/Cthulhu";
    const picker = window.appEnv?.chooseFolder;
    if (picker) {
      const chosen = await picker();
      if (!chosen) return;
      destDir = chosen;
    }
    try {
      const { exported, missing } = await exportOutputs(paths, destDir);
      const parts: string[] = [];
      if (exported.length) parts.push(`已导出 ${exported.length} 个产物`);
      if (missing.length) parts.push(`${missing.length} 个素材尚无产物`);
      toast(parts.length ? parts.join(" · ") : "所选素材均未处理，请先清洗");
    } catch (error) {
      toast(error instanceof Error ? error.message : "导出失败");
    }
  };

  const remove = () => {
    deleteSelected(ids);
    toast(`已移除 ${ids.length} 个素材`);
  };

  return (
    <div className="batch-popup" role="dialog" aria-label="批量操作">
      <span className="batch-count">已选 {ids.length} 个</span>
      <Button variant="secondary" size="sm" onClick={() => void batchExport()}>
        导出产物
      </Button>
      <Button variant="ghost" size="sm" className="text-destructive" onClick={remove}>
        移除
      </Button>
      <Button
        variant="ghost"
        size="icon"
        className="h-7 w-7"
        aria-label="关闭"
        onClick={() => toggleSelectAll(ids)}
      >
        <X />
      </Button>
    </div>
  );
}

/* ============ 中栏：预览 ============ */
interface PreviewProps {
  material: Material | null;
  frame: number;
  setFrame: (value: number) => void;
  playing: boolean;
  setPlaying: (value: boolean) => void;
  comparePct: number;
  setComparePct: (value: number) => void;
  compareLeft: string;
  compareRight: string;
  compareLeftLabel: string;
  compareRightLabel: string;
  onExitCompare: () => void;
}

function PreviewPane({
  material,
  frame,
  setFrame,
  playing,
  setPlaying,
  comparePct,
  setComparePct,
  compareLeft,
  compareRight,
  compareLeftLabel,
  compareRightLabel,
  onExitCompare,
}: PreviewProps) {
  const regions = useRegionsStore((state) => state.regions);
  const activeRegionId = useRegionsStore((state) => state.activeId);
  const patchRegion = useRegionsStore((state) => state.patchRegion);
  const beginMove = useRegionsStore((state) => state.beginMove);

  const stageRef = useRef<HTMLDivElement>(null);
  const mediaBoxRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const compareLeftRef = useRef<HTMLVideoElement>(null);
  const compareMode = !!compareLeft && !!compareRight;
  const lastEmitRef = useRef(0);
  const [zoomPos, setZoomPos] = useState<{ x: number; y: number } | null>(null);
  const [zoomSrc, setZoomSrc] = useState<string | null>(null);
  const [mediaDuration, setMediaDuration] = useState<number | null>(null);
  const [mediaSize, setMediaSize] = useState<{ w: number; h: number } | null>(null);
  const [fitBox, setFitBox] = useState<{ w: number; h: number } | null>(null);

  const activeRegion = regions.find((r) => r.id === activeRegionId) ?? null;
  const fps = Number.parseFloat(material?.fps ?? "30") || 30;
  const hasRealFrame = !!material?.path;
  const durationSec = mediaDuration ?? material?.duration ?? null;
  const maxFrame = durationSec
    ? Math.max(1, Math.round(durationSec * fps) - 1)
    : FRAME_MAX;
  const frameSrc = hasRealFrame
    ? frameUrl(material.path as string, frame / fps)
    : (material?.frame ?? "/assets/frame-drama.svg");

  // 素材切换：重置真实时长/尺寸与放大镜快照，避免串台。
  useEffect(() => {
    setMediaDuration(null);
    setMediaSize(null);
    setFitBox(null);
    setZoomPos(null);
    setZoomSrc(null);
  }, [material?.id]);

  // 素材切换或元数据就绪后，把时间轴收敛到真实时长内。
  useEffect(() => {
    if (durationSec && frame > maxFrame) {
      setFrame(0);
    }
  }, [durationSec, frame, maxFrame, setFrame]);

  // 依据视频真实宽高与舞台尺寸计算等比显示框，保证水印框选与画面一一对应。
  useEffect(() => {
    if (!hasRealFrame) {
      setFitBox(null);
      return;
    }
    const stage = stageRef.current;
    if (!stage) return;
    const compute = () => {
      if (!mediaSize) return;
      const scale = Math.min(
        stage.clientWidth / mediaSize.w,
        stage.clientHeight / mediaSize.h,
      );
      setFitBox({
        w: Math.floor(mediaSize.w * scale),
        h: Math.floor(mediaSize.h * scale),
      });
    };
    compute();
    const observer = new ResizeObserver(compute);
    observer.observe(stage);
    return () => observer.disconnect();
  }, [hasRealFrame, mediaSize, material?.id]);

  // 播放/暂停状态与真实 <video> 同步（静音播放，避免浏览器自动播放限制）。
  useEffect(() => {
    const video = videoRef.current;
    const compare = compareLeftRef.current;
    if (!video || !hasRealFrame) return;
    if (playing) {
      if (video.ended) video.currentTime = 0;
      video.play().catch(() => setPlaying(false));
      if (compare) compare.play().catch(() => undefined);
    } else {
      video.pause();
      if (compare) compare.pause();
      setFrame(Math.min(maxFrame, Math.round(video.currentTime * fps)));
    }
  }, [playing, hasRealFrame, setPlaying, setFrame, fps, maxFrame, material?.id, compareLeft]);

  // 暂停状态下，时间轴/逐帧/快捷键改变帧号时把视频 seek 到对应时间；
  // 播放中由视频自身推进，避免帧号回灌造成卡顿。
  useEffect(() => {
    if (playing) return;
    const video = videoRef.current;
    const compare = compareLeftRef.current;
    if (!video || !hasRealFrame || video.readyState < 1) return;
    const target = frame / fps;
    if (Math.abs(video.currentTime - target) > 0.5 / fps) {
      video.currentTime = target;
    }
    if (compare && Math.abs(compare.currentTime - target) > 0.5 / fps) {
      compare.currentTime = target;
    }
  }, [frame, fps, playing, hasRealFrame, material?.id, compareLeft]);

  // 把当前视频画面快照成放大镜底图（跨源播放时需匿名 CORS）。
  const captureZoom = () => {
    const video = videoRef.current;
    const box = mediaBoxRef.current;
    if (!video || !box || video.readyState < 2 || !hasRealFrame) return;
    const canvas = document.createElement("canvas");
    canvas.width = box.clientWidth || 640;
    canvas.height = box.clientHeight || 360;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    setZoomSrc(canvas.toDataURL("image/jpeg", 0.92));
  };

  const onTimeUpdate = () => {
    const video = videoRef.current;
    if (!video) return;
    const compare = compareLeftRef.current;
    if (compare && Math.abs(compare.currentTime - video.currentTime) > 0.08) {
      compare.currentTime = Math.min(video.currentTime, compare.duration || video.currentTime);
    }
    // 帧号只以约 10Hz 回写，降低整页渲染频率；视频画面本身保持流畅。
    const now = performance.now();
    if (now - lastEmitRef.current >= 100) {
      lastEmitRef.current = now;
      setFrame(Math.min(maxFrame, Math.round(video.currentTime * fps)));
    }
    if (zoomPos) captureZoom();
  };

  const onLoadedMetadata = () => {
    const video = videoRef.current;
    if (!video) return;
    if (Number.isFinite(video.duration) && video.duration > 0) {
      setMediaDuration(video.duration);
    }
    if (video.videoWidth && video.videoHeight) {
      setMediaSize({ w: video.videoWidth, h: video.videoHeight });
    }
    if (Number.isFinite(video.duration) && frame / fps > video.duration) {
      setFrame(0);
      video.currentTime = 0;
    }
  };

  const percentFrom = (e: ReactPointerEvent, el: HTMLElement) => {
    const rect = el.getBoundingClientRect();
    return {
      x: ((e.clientX - rect.left) / rect.width) * 100,
      y: ((e.clientY - rect.top) / rect.height) * 100,
    };
  };

  const onComparePointerDown = (e: ReactPointerEvent<HTMLDivElement>) => {
    if ((e.target as HTMLElement).closest(".sel-box")) return;
    if (zoomPos) return;
    const el = e.currentTarget;
    el.setPointerCapture(e.pointerId);
    setComparePct(Math.max(0, Math.min(100, percentFrom(e, el).x)));
  };

  const onComparePointerMove = (e: ReactPointerEvent<HTMLDivElement>) => {
    if (zoomPos) {
      setZoomPos((current) => (current ? percentFrom(e, e.currentTarget) : current));
      return;
    }
    if (e.buttons & 1) {
      setComparePct(Math.max(0, Math.min(100, percentFrom(e, e.currentTarget).x)));
    }
  };

  const onCompareDoubleClick = (e: ReactPointerEvent<HTMLDivElement>) => {
    if ((e.target as HTMLElement).closest(".sel-box")) return;
    const point = percentFrom(e, e.currentTarget);
    setZoomPos((current) => (current ? null : point));
    if (!zoomPos && hasRealFrame) captureZoom();
  };

  const onBoxPointerDown = (e: ReactPointerEvent<HTMLDivElement>) => {
    if (!activeRegion) return;
    const box = mediaBoxRef.current;
    if (!box) return;
    beginMove();
    const el = e.currentTarget;
    el.setPointerCapture(e.pointerId);
    const start = percentFrom(e, box);
    const startLeft = activeRegion.left;
    const startTop = activeRegion.top;
    const move = (ev: PointerEvent) => {
      const rect = box.getBoundingClientRect();
      const p = {
        x: ((ev.clientX - rect.left) / rect.width) * 100,
        y: ((ev.clientY - rect.top) / rect.height) * 100,
      };
      patchRegion(activeRegion.id, {
        left: Math.max(0, Math.min(100 - activeRegion.w, startLeft + (p.x - start.x))),
        top: Math.max(0, Math.min(100 - activeRegion.h, startTop + (p.y - start.y))),
      });
    };
    const up = () => {
      box.removeEventListener("pointermove", move);
      box.removeEventListener("pointerup", up);
    };
    box.addEventListener("pointermove", move);
    box.addEventListener("pointerup", up);
  };

  const onResizePointerDown = (e: ReactPointerEvent<HTMLSpanElement>) => {
    e.stopPropagation();
    if (!activeRegion) return;
    const box = mediaBoxRef.current;
    if (!box) return;
    beginMove();
    const el = e.currentTarget;
    el.setPointerCapture(e.pointerId);
    const start = percentFrom(e, box);
    const startW = activeRegion.w;
    const startH = activeRegion.h;
    const move = (ev: PointerEvent) => {
      const rect = box.getBoundingClientRect();
      const p = {
        x: ((ev.clientX - rect.left) / rect.width) * 100,
        y: ((ev.clientY - rect.top) / rect.height) * 100,
      };
      patchRegion(activeRegion.id, {
        w: Math.max(4, Math.min(100 - activeRegion.left, startW + (p.x - start.x))),
        h: Math.max(3, Math.min(100 - activeRegion.top, startH + (p.y - start.y))),
      });
    };
    const up = () => {
      box.removeEventListener("pointermove", move);
      box.removeEventListener("pointerup", up);
    };
    box.addEventListener("pointermove", move);
    box.addEventListener("pointerup", up);
  };

  const onBoxKeyDown = (e: ReactKeyboardEvent<HTMLDivElement>) => {
    if (!activeRegion) return;
    const step = e.shiftKey ? 5 : 1;
    let left = activeRegion.left;
    let top = activeRegion.top;
    if (e.key === "ArrowLeft") left = Math.max(0, left - step);
    else if (e.key === "ArrowRight") left = Math.min(100 - activeRegion.w, left + step);
    else if (e.key === "ArrowUp") top = Math.max(0, top - step);
    else if (e.key === "ArrowDown") top = Math.min(100 - activeRegion.h, top + step);
    else return;
    e.preventDefault();
    patchRegion(activeRegion.id, { left, top });
  };

  const beforeClip = `inset(0 ${100 - comparePct}% 0 0)`;

  return (
    <section className="pane pane-center">
      <div className="preview-head">
        <span className="preview-name">{material?.name ?? "未选择素材"}</span>
        {material && <span className="preview-meta mono">{material.res} · {material.fps}</span>}
        {compareMode && (
          <Button variant="ghost" size="sm" onClick={onExitCompare}>
            退出对比
          </Button>
        )}
      </div>

      <div className="preview-stage" ref={stageRef}>
        <div
          ref={mediaBoxRef}
          className={hasRealFrame ? "media-box" : "media-box fill"}
          style={hasRealFrame && fitBox ? { width: fitBox.w, height: fitBox.h } : undefined}
          onDoubleClick={onCompareDoubleClick}
        >
          {hasRealFrame ? (
            compareMode ? (
              <div
                className="compare compare-live"
                onPointerDown={onComparePointerDown}
                onPointerMove={onComparePointerMove}
              >
                <video
                  key={`right-${compareRight}`}
                  ref={videoRef}
                  className="preview-video"
                  src={mediaUrl(compareRight)}
                  preload="metadata"
                  muted
                  playsInline
                  crossOrigin="anonymous"
                  onTimeUpdate={onTimeUpdate}
                  onLoadedMetadata={onLoadedMetadata}
                  onEnded={() => setPlaying(false)}
                />
                <div className="compare-before" style={{ clipPath: beforeClip }}>
                  <video
                    key={`left-${compareLeft}`}
                    ref={compareLeftRef}
                    className="preview-video"
                    src={mediaUrl(compareLeft)}
                    preload="metadata"
                    muted
                    playsInline
                    crossOrigin="anonymous"
                  />
                </div>
                <div
                  className="compare-handle"
                  style={{ left: `${comparePct}%` }}
                  role="presentation"
                >
                  <span className="compare-handle-knob">
                    <Code2 />
                  </span>
                </div>
                <span className="compare-badge compare-badge-left">{compareLeftLabel}</span>
                <span className="compare-badge compare-badge-right">{compareRightLabel}</span>
              </div>
            ) : (
              <video
                key={material?.id}
                ref={videoRef}
                className="preview-video"
                src={material?.path ? mediaUrl(material.path) : undefined}
                preload="metadata"
                muted
                playsInline
                crossOrigin="anonymous"
                onTimeUpdate={onTimeUpdate}
                onLoadedMetadata={onLoadedMetadata}
                onEnded={() => setPlaying(false)}
              />
            )
          ) : (
            <div
              className="compare"
              onPointerDown={onComparePointerDown}
              onPointerMove={onComparePointerMove}
            >
              <img src={frameSrc} alt="预览视频帧" />
              <div className="compare-before" style={{ clipPath: beforeClip }}>
                <img src={frameSrc} alt="" aria-hidden="true" />
              </div>
              <div
                className="compare-handle"
                style={{ left: `${comparePct}%` }}
                role="presentation"
              />
            </div>
          )}

          {activeRegion && (
            <div
              className="repair-veil"
              style={{
                left: `${activeRegion.left}%`,
                top: `${activeRegion.top}%`,
                width: `${activeRegion.w}%`,
                height: `${activeRegion.h}%`,
                opacity: 0.25,
              }}
            >
              <span>{activeRegion.name} · 修复区域预览</span>
            </div>
          )}

          {zoomPos && (() => {
            const box = mediaBoxRef.current;
            const width = box?.clientWidth ?? 640;
            const height = box?.clientHeight ?? 360;
            const scale = 3;
            const lens = 120;
            const centerX = (zoomPos.x / 100) * width;
            const centerY = (zoomPos.y / 100) * height;
            const translateX = centerX * scale - lens / 2;
            const translateY = centerY * scale - lens / 2;
            const left = Math.max(0, Math.min(width - lens, centerX - lens / 2));
            const top = Math.max(0, Math.min(height - lens, centerY - lens / 2));
            const imageStyle = {
              width: width * scale,
              height: height * scale,
              transform: `translate(${-translateX}px, ${-translateY}px)`,
            };
            if (hasRealFrame && !zoomSrc) return null;
            return (
              <div className="zoom-lens" style={{ left, top }}>
                {hasRealFrame ? (
                  <img src={zoomSrc ?? ""} alt="" aria-hidden="true" style={imageStyle} />
                ) : (
                  <>
                    <img src={frameSrc} alt="" aria-hidden="true" style={imageStyle} />
                    <div className="zoom-before" style={{ clipPath: beforeClip }}>
                      <img src={frameSrc} alt="" aria-hidden="true" style={imageStyle} />
                    </div>
                  </>
                )}
              </div>
            );
          })()}

          {activeRegion && (
            <div
              className="sel-box"
              role="group"
              tabIndex={0}
              aria-label="水印区域选框，可用方向键微调位置"
              style={{
                left: `${activeRegion.left}%`,
                top: `${activeRegion.top}%`,
                width: `${activeRegion.w}%`,
                height: `${activeRegion.h}%`,
              }}
              onPointerDown={onBoxPointerDown}
              onKeyDown={onBoxKeyDown}
            >
              <span className="sel-label">{activeRegion.name}</span>
              <span className="sel-resize" onPointerDown={onResizePointerDown} />
            </div>
          )}
        </div>
      </div>

      <div className="preview-foot">
        <Button
          variant="ghost"
          size="icon"
          aria-label="播放 / 暂停"
          onClick={() => setPlaying(!playing)}
        >
          {playing ? <Pause /> : <Play />}
        </Button>
        <Button
          variant="ghost"
          size="icon"
          aria-label="上一帧"
          onClick={() => {
            setPlaying(false);
            setFrame(Math.max(0, frame - 1));
          }}
        >
          <SkipBack />
        </Button>
        <Button
          variant="ghost"
          size="icon"
          aria-label="下一帧"
          onClick={() => {
            setPlaying(false);
            setFrame(Math.min(maxFrame, frame + 1));
          }}
        >
          <SkipForward />
        </Button>
        <span className="timecode mono">{fmtFrames(frame, fps)}</span>
        <div className="timeline-wrap">
          <input
            className="timeline"
            type="range"
            min={0}
            max={maxFrame}
            value={frame}
            aria-label="时间轴"
            onChange={(e) => {
              const value = Number(e.target.value);
              setFrame(value);
              const video = videoRef.current;
              if (video && hasRealFrame) {
                video.currentTime = value / fps;
              }
            }}
          />
        </div>
        <span className="timecode mono">{fmtFrames(maxFrame, fps)}</span>
      </div>

    </section>
  );
}

/* ============ 右栏：上下文面板 ============ */
interface ContextProps {
  tab: (typeof TAB_KEYS)[number];
  setTab: (tab: (typeof TAB_KEYS)[number]) => void;
  onStartCompare: (left: string, right: string, leftLabel: string, rightLabel: string) => void;
  cleanOptionsRef: MutableRefObject<DesensitizeOptions | null>;
}

function ContextPanel({ tab, setTab, onStartCompare, cleanOptionsRef }: ContextProps) {
  const material = useMaterialsStore((state) =>
    state.materials.find((m) => m.id === state.activeId),
  );
  const regions = useRegionsStore((state) => state.regions);
  const activeRegionId = useRegionsStore((state) => state.activeId);
  const undoStack = useRegionsStore((state) => state.undoStack);
  const redoStack = useRegionsStore((state) => state.redoStack);
  const selectRegion = useRegionsStore((state) => state.selectRegion);
  const addRegion = useRegionsStore((state) => state.addRegion);
  const deleteRegion = useRegionsStore((state) => state.deleteRegion);
  const patchRegion = useRegionsStore((state) => state.patchRegion);
  const undo = useRegionsStore((state) => state.undo);
  const redo = useRegionsStore((state) => state.redo);
  const addHistory = useHistoryStore((state) => state.add);
  const jobs = useQueueStore((state) => state.jobs);

  const [level, setLevel] = useState<CleanLevel>("平衡");
  const [retime, setRetime] = useState(30);
  const [perturb, setPerturb] = useState(20);
  const [audioClean, setAudioClean] = useState(true);
  const [antiReembed, setAntiReembed] = useState(false);
  const [antiLevel, setAntiLevel] = useState("关闭");
  const [recropOn, setRecropOn] = useState(false);
  const [detailProtectOn, setDetailProtectOn] = useState(false);
  const [sharpness, setSharpness] = useState(true);
  const [colorFix, setColorFix] = useState(true);
  const [aiDenoise, setAiDenoise] = useState(true);
  const [spoof, setSpoof] = useState(false);
  const [codec, setCodec] = useState("H.264");
  const [lossless, setLossless] = useState(false);
  const [resolution, setResolution] = useState("保持原始分辨率");
  const [bitrate, setBitrate] = useState("");
  const [gop, setGop] = useState("");
  const [fpsOut, setFpsOut] = useState("");
  const [settingsExportDir, setSettingsExportDir] = useState("");
  const [settingsNaming, setSettingsNaming] = useState("原文件名 + 时间戳");
  const [templateList, setTemplateList] = useState<TemplateInfo[]>([]);
  const [templateValue, setTemplateValue] = useState("manual");
  const [lastOutput, setLastOutput] = useState<string | null>(null);
  const cleaning = !!material?.path && pendingCleanPaths(jobs).has(material.path);
  const [detectSubmitting, setDetectSubmitting] = useState(false);
  const [audio, setAudio] = useState<AudioAnalysis | null>(null);
  const [audioMissing, setAudioMissing] = useState(false);
  const [lastClean, setLastClean] = useState<{
    psnr_db?: number;
    ssim?: number;
    vmaf?: number | null;
    vmaf_aligned?: number | null;
  } | null>(null);
  const [dedupRisk, setDedupRisk] = useState<{
    duplicate_risk: number;
    risk_level: string;
  } | null>(null);
  const [candidates, setCandidates] = useState<CandidateInfo[] | null>(null);
  const [outputs, setOutputs] = useState<OutputInfo[]>([]);
  const [variantDetail, setVariantDetail] = useState<VariantInfo | null>(null);
  const [pendingDelete, setPendingDelete] = useState<OutputInfo | null>(null);
  const [playerPath, setPlayerPath] = useState<string | null>(null);
  const [candidatesBusy, setCandidatesBusy] = useState(false);
  const tabRowRef = useRef<HTMLDivElement | null>(null);
  const tabItemRefs = useRef<(HTMLButtonElement | null)[]>([]);
  const [visibleTabCount, setVisibleTabCount] = useState<number>(TAB_KEYS.length);

  const risk = material?.risk ?? "待检测";
  const score = material?.score ?? 0;
  const report = (material as { report?: DetectReport } | undefined)?.report ?? null;
  const detecting = !!material?.path && pendingDetectPaths(jobs).has(material.path);
  const pct = risk === "待检测" ? 100 : Math.max(0, Math.min(100, score));
  const ringColor =
    risk === "待检测"
      ? "var(--border)"
      : risk === "有疑似特征"
        ? "var(--warn)"
        : "var(--ok)";
  const scoreSub = !report
    ? "尚未扫描暗水印，点击素材库右上角「批量扫描」开始检测"
    : report.bitstream.flags.length > 0
      ? `码流评分 ${score} · 命中 ${report.bitstream.flags.length} 项疑似特征 · 建议差分复核`
      : `码流评分 ${score} · 未命中明显码流异常 · 建议差分复核`;

  useEffect(() => {
    setAudio(null);
    setAudioMissing(false);
    if (!material?.path || !report) return;
    let cancelled = false;
    void analyzeAudio(material.path)
      .then((result) => {
        if (!cancelled) setAudio(result);
      })
      .catch(() => {
        if (!cancelled) setAudioMissing(true);
      });
    return () => {
      cancelled = true;
    };
  }, [material?.path, report]);

  useEffect(() => {
    void getSettings()
      .then((settings) => {
        setSettingsExportDir((settings.export_dir as string) ?? "");
        setSettingsNaming((settings.naming as string) ?? "原文件名 + 时间戳");
      })
      .catch(() => undefined);
  }, []);

  const applyTemplatePayload = (payload: TemplatePayload) => {
    setLevel(payload.level);
    setRetime(payload.retime);
    setPerturb(payload.perturb);
    setAudioClean(payload.audioRemix);
    setAntiReembed(payload.antiReembed);
    setAntiLevel(payload.anti);
    setRecropOn(payload.recropOn);
    setDetailProtectOn(payload.detailProtectOn);
    setSharpness(payload.sharpness);
    setColorFix(payload.colorRestore);
    setAiDenoise(payload.denoise);
    setSpoof(payload.spoof);
    setCodec(payload.codec);
    setLossless(payload.lossless);
    setResolution(payload.resolution);
    setBitrate(payload.bitrate != null ? String(payload.bitrate) : "");
    setGop(payload.gop != null ? String(payload.gop) : "");
    setFpsOut(payload.fpsOut != null ? String(payload.fpsOut) : "");
  };

  const applyTemplateById = (id: string) => {
    if (id === "manual") {
      setTemplateValue("manual");
      return;
    }
    const template = templateList.find((item) => item.id === id);
    if (!template) return;
    applyTemplatePayload(payloadOf(template));
    setTemplateValue(id);
    toast(`已套用模板：${template.name}`);
  };

  useEffect(() => {
    void listTemplates()
      .then(setTemplateList)
      .catch(() => undefined);
  }, []);

  // 模板页点击「使用」后，切回工作台时把参数回填到右侧面板。
  useEffect(() => {
    const pending = useAppStore.getState().consumePendingTemplate();
    if (pending) {
      applyTemplatePayload(pending.payload);
      setTemplateValue(pending.id);
    }
  }, []);

  // 已套用的模板被删除或列表刷新后，回退为手动状态，避免下拉框悬空。
  useEffect(() => {
    if (templateValue !== "manual" && !templateList.some((item) => item.id === templateValue)) {
      setTemplateValue("manual");
    }
  }, [templateList, templateValue]);

  // 产物列表绑定当前素材上下文，切换素材即刷新。
  useEffect(() => {
    if (!material?.path) {
      setOutputs([]);
      return;
    }
    void fetchOutputs(material.path)
      .then((result) => {
        setOutputs(result.outputs);
      })
      .catch(() => setOutputs([]));
  }, [material?.path]);

  // 标签页自适应：放得下就全部展示，放不下时收起尾部标签并在右侧提供“更多”下拉。
  useEffect(() => {
    const compute = () => {
      const row = tabRowRef.current;
      if (!row) return;
      const widths = TAB_KEYS.map((_, index) => tabItemRefs.current[index]?.offsetWidth ?? 0);
      const total = widths.reduce((sum, width) => sum + width + 4, 0);
      // 全部放得下时不显示“更多”按钮，也无需预留宽度。
      if (total <= row.clientWidth) {
        setVisibleTabCount(TAB_KEYS.length);
        return;
      }
      const available = row.clientWidth - 48; // 溢出时预留“更多”按钮宽度
      let used = 0;
      let count = 0;
      for (let index = 0; index < TAB_KEYS.length; index++) {
        const item = tabItemRefs.current[index];
        if (!item) break;
        used += item.offsetWidth + 4; // 加上相邻标签的间距
        if (used > available) break;
        count = index + 1;
      }
      setVisibleTabCount(Math.max(1, count));
    };
    compute();
    const observer = new ResizeObserver(compute);
    if (tabRowRef.current) observer.observe(tabRowRef.current);
    return () => observer.disconnect();
  }, []);

  const runClean = async () => {
    const target = material as (Material & { path?: string }) | null;
    if (!target?.path || !material) {
      toast("该素材缺少本地路径，无法执行");
      return;
    }
    const srcStem = target.path.replace(/\.(mp4|mov|mkv|avi|flv|ts)$/i, "");
    const srcName = srcStem.split(/[\\/]/).pop() ?? srcStem;
    const now = new Date();
    const pad = (value: number) => String(value).padStart(2, "0");
    const ts = `${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}_${pad(
      now.getHours(),
    )}${pad(now.getMinutes())}${pad(now.getSeconds())}`;
    const fileName =
      settingsNaming === "时间戳 + 原文件名"
        ? `${ts}_${srcName}_cleaned.mp4`
        : `${srcName}_cleaned_${ts}.mp4`;
    const baseDir = settingsExportDir;
    const output = baseDir
      ? `${baseDir.replace(/\/+$/, "")}/${fileName}`
      : `${srcStem}_cleaned_${ts}.mp4`;
    const anti = ANTI_PRESETS[antiLevel];
    try {
      const cleanOptions: DesensitizeOptions = {
        reorder: false,
        speed: Math.max(0.85, 1 - 0.15 * (retime / 100)),
        recrop: recropOn ? 0.015 + 0.075 * (perturb / 100) : 0,
        perturb: perturb / 100,
        regrade: true,
        audioRemix: audioClean,
        sharpness,
        colorRestore: colorFix,
        denoise: aiDenoise,
        antiReembed,
        detailProtect: detailProtectOn ? 0.5 : 0,
        seed: Math.floor(Math.random() * 1_000_000),
        codec: codec === "H.265" ? "libx265" : "libx264",
        lossless,
        spoof,
        ...(resolution !== "保持原始分辨率" ? { resolution } : {}),
        ...(bitrate ? { bitrateKbps: Number(bitrate) } : {}),
        ...(gop ? { gop: Number(gop) } : {}),
        ...(fpsOut ? { fpsOut: Number(fpsOut) } : {}),
        ...(anti
          ? {
              rotate: anti.rotate,
              phashAttack: !anti.jointAttack,
              phashEpsilon: anti.epsilon,
              ...(anti.median ? { median: anti.median } : {}),
              ...(anti.noise ? { noise: anti.noise } : {}),
              ...(anti.requant ? { requant: anti.requant } : {}),
              ...(anti.dctStep ? { dctStep: anti.dctStep } : {}),
              ...(anti.dropEvery ? { dropEvery: anti.dropEvery } : {}),
              ...(anti.jitter ? { jitter: anti.jitter } : {}),
              ...(anti.perspective ? { perspective: anti.perspective } : {}),
              ...(anti.warp ? { warp: anti.warp } : {}),
              ...(anti.mirror ? { mirror: true } : {}),
              ...(anti.chromaLevels ? { chromaLevels: anti.chromaLevels } : {}),
              ...(anti.subtractBeta ? { subtractBeta: anti.subtractBeta } : {}),
              ...(anti.transcodeChain ? { transcodeChain: true } : {}),
              ...(anti.jointAttack ? { multiHashAttack: true } : {}),
              ...(anti.saliency ? { saliency: anti.saliency } : {}),
              ...(anti.nativeFilters ? { nativeFilters: true } : {}),
              ...(anti.detailProtect ? { detailProtect: anti.detailProtect } : {}),
              ...(anti.shotRetime ? { shotRetime: true } : {}),
              ...(anti.cutJitter ? { cutJitter: anti.cutJitter } : {}),
              ...(anti.audioStrong ? { audioStrong: true } : {}),
            }
          : {}),
      };
      cleanOptionsRef.current = cleanOptions;
      await enqueueJob(`${material.name} · 清洗去重`, [
        {
          kind: "desensitize",
          path: target.path,
          options: { output, ...toSnakeOptions(cleanOptions) },
        },
      ]);
      setLastClean(null);
      setLastOutput(null);
      toast("已加入处理队列，完成后自动刷新校验与预览");
    } catch (error) {
      toast(error instanceof Error ? error.message : "入队失败");
    }
  };

  const detectNow = async () => {
    if (!material?.path) {
      toast("该素材缺少本地路径，无法检测");
      return;
    }
    if (detecting || detectSubmitting) {
      toast("该素材正在检测中，请等待完成");
      return;
    }
    setDetectSubmitting(true);
    try {
      const job = await enqueueJob(`${material.name} · 检测`, [
        { kind: "detect", path: material.path },
      ]);
      // 立即写入队列状态，避免按钮在事件到达前仍可重复点击。
      useQueueStore.getState().applyEvent({ type: "job:state", job });
      toast("已加入检测队列，完成后结果自动更新");
    } catch {
      toast("入队失败，请确认引擎在线");
    } finally {
      setDetectSubmitting(false);
    }
  };

  const runCandidates = async () => {
    const target = material as (Material & { path?: string }) | null;
    if (!target?.path || !cleanOptionsRef.current) {
      toast("请先执行一次清洗，再生成候选");
      return;
    }
    setCandidatesBusy(true);
    try {
      const dir = `${target.path.replace(/\.(mp4|mov|mkv|avi|flv|ts)$/i, "")}_候选`;
      const report = await generateCandidates(target.path, dir, 3, cleanOptionsRef.current);
      setCandidates(report.candidates);
      void useMaterialsStore.getState().refreshOutputCounts();
      toast(`已生成 ${report.candidates.length} 个候选，按低损优选排序`);
    } catch (error) {
      toast(error instanceof Error ? error.message : "生成候选失败");
    } finally {
      setCandidatesBusy(false);
    }
  };

  const showVariant = (outputPath: string) => {
    if (!material?.path) return;
    void listVariants(material.path)
      .then((list) => {
        const found = list.find((item) => item.output === outputPath);
        if (found) {
          setVariantDetail(found);
        } else {
          toast("该产物没有参数记录", "旧版本产物可能没有记录");
        }
      })
      .catch(() => toast("读取产物记录失败"));
  };

  const runComparePair = (a: string, b: string, leftLabel: string, rightLabel: string) => {
    onStartCompare(a, b, leftLabel, rightLabel);
  };

  const compareOriginal = (output: OutputInfo) => {
    if (material?.path) runComparePair(material.path, output.path, "原片", "处理后");
  };

  const confirmDelete = async () => {
    if (!pendingDelete) return;
    try {
      await deleteOutput(pendingDelete.path);
      await fetchOutputs(material?.path ?? "")
        .then((report) => setOutputs(report.outputs))
        .catch(() => undefined);
      void useMaterialsStore.getState().refreshOutputCounts();
      toast(`已删除产物：${pendingDelete.name}`);
    } catch (error) {
      toast(error instanceof Error ? error.message : "删除失败");
    } finally {
      setPendingDelete(null);
    }
  };

  // 清洗任务完成后：更新校验指标、产物列表、历史与并排预览。
  const lastHandledTaskRef = useRef<string | null>(null);
  useEffect(() => {
    const task = jobs
      .flatMap((job) => job.tasks)
      .find(
        (item) =>
          item.kind === "desensitize" &&
          item.path === material?.path &&
          item.status === "done" &&
          item.result != null &&
          item.id !== lastHandledTaskRef.current,
      );
    if (!task) return;
    lastHandledTaskRef.current = task.id;
    const result = task.result as { output?: string; similarity_after?: { content_cosine: number } } | null;
    if (!result?.output) return;
    setLastOutput(result.output);
    setLastClean(result as never);
    const dedup = (task.result as { dedup?: { duplicate_risk: number; risk_level: string } | null } | null)?.dedup;
    setDedupRisk(dedup ?? null);
    addHistory({
      name: material?.name ?? "",
      time: nowStr(),
      action: "清洗去重",
      params: `${level}档 · 对抗${antiLevel} · ${codec}`,
      out: result.output,
      result: "成功",
    });
    void useMaterialsStore.getState().refreshOutputCounts();
    void fetchOutputs(material?.path ?? "")
      .then((report) => setOutputs(report.outputs))
      .catch(() => undefined);
    toast(`清洗完成：内容相似度 ${(result.similarity_after?.content_cosine ?? 0).toFixed(2)}`, result.output, {
      label: "打开文件夹",
      onClick: () => openInFolder(result.output as string),
    });
  }, [jobs, material?.path, level, antiLevel, codec, addHistory]);

  const runRepairJob = async () => {
    const target = material as (Material & { path?: string }) | null;
    if (!target?.path) {
      toast("该素材缺少本地路径，无法执行");
      return;
    }
    if (!regions.length) {
      toast("请先添加水印区域");
      return;
    }
    const output = `${target.path.replace(/\.(mp4|mov|mkv|avi|flv|ts)$/i, "")}_repaired.mp4`;
    try {
      await enqueueJob(`${target.name} · 修复`, [
        {
          kind: "repair",
          path: target.path,
          options: {
            output,
            crf: 23,
            regions: regions.map((region) => ({
              x: region.left / 100,
              y: region.top / 100,
              w: region.w / 100,
              h: region.h / 100,
              ...(region.start != null ? { start: region.start } : {}),
              ...(region.end != null ? { end: region.end } : {}),
            })),
          },
        },
      ]);
      addHistory({
        name: target.name,
        time: nowStr(),
        action: "可见水印修复",
        params: `${regions.length} 个区域 · delogo`,
        out: output,
        result: "排队中",
      });
      toast("修复任务已加入队列");
    } catch {
      toast("入队失败，请确认引擎在线");
    }
  };

  return (
    <aside className="pane pane-right">
      <Tabs value={tab} onValueChange={(value) => setTab(value as (typeof TAB_KEYS)[number])} className="min-h-0 flex-1">
        <div ref={tabRowRef} className="flex w-full shrink-0 items-center border-b border-border">
          <TabsList
            variant="line"
            className="flex min-w-0 flex-1 justify-start gap-1 overflow-hidden rounded-none border-0 p-0"
          >
            {TAB_KEYS.map((key, index) => (
              <TabsTrigger
                key={key}
                ref={(el) => {
                  tabItemRefs.current[index] = el;
                }}
                value={key}
                className={cn(
                  "shrink-0 whitespace-nowrap px-2",
                  index >= visibleTabCount && "invisible",
                )}
              >
                {key}
              </TabsTrigger>
            ))}
          </TabsList>
          {visibleTabCount < TAB_KEYS.length && (
            <Select
              value=""
              onValueChange={(value) => setTab(value as (typeof TAB_KEYS)[number])}
            >
              <SelectTrigger
                className="mr-1 h-8 w-8 shrink-0 justify-center gap-0 border-0 bg-transparent p-0 text-muted-foreground shadow-none hover:text-foreground [&>svg:last-child]:hidden"
                aria-label="更多标签页"
              >
                <MoreVertical className="size-4" />
              </SelectTrigger>
              <SelectContent align="end">
                {TAB_KEYS.slice(visibleTabCount).map((key) => (
                  <SelectItem key={key} value={key}>
                    {key}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          )}
        </div>

        <TabsContent value="检测参考" className="tab-pane">
          <ScrollArea className="h-full">
            <div className="flex flex-col gap-2.5">
              {material?.path && (
                <Button
                  variant="secondary"
                  size="sm"
                  className="w-full"
                  disabled={detecting || detectSubmitting}
                  onClick={() => void detectNow()}
                >
                  {detecting || detectSubmitting ? "检测中…" : report ? "重新检测" : "检测此素材"}
                </Button>
              )}
              {report ? (
                <>
                  <Card className="flex items-center gap-3.5 border border-border p-3.5 shadow-none ring-0">
                    <div
                      className="ring"
                      style={{
                        background: `conic-gradient(var(--warn) 0 ${Math.min(100, report.bitstream.score)}%, var(--muted) ${Math.min(100, report.bitstream.score)}% 100%)`,
                      }}
                    >
                      <div className="ring-inner">
                        <span className="ring-score">{report.bitstream.score}</span>
                      </div>
                    </div>
                    <div>
                  <div className="score-title">
                        {riskLabel(risk)}
                  </div>
                      <div className="score-sub">码流层启发式检测 · 容器/SEI/压缩域联合</div>
                    </div>
                  </Card>

                  <div className="section-title">检测信息（真实引擎）</div>
                  <div className="kv-card">
                    <div className="kv-row">
                      <span>编码 / 分辨率</span>
                      <b>
                        {report.probe.codec} · {report.probe.width}×{report.probe.height}
                      </b>
                    </div>
                    <div className="kv-row">
                      <span>帧率 / 时长</span>
                      <b>
                        {report.probe.fps}fps · {report.probe.duration.toFixed(1)}s
                      </b>
                    </div>
                    <div className="kv-row">
                      <span>SEI 数量</span>
                      <b>{report.sei_count ?? "—"}</b>
                    </div>
                    <div className="kv-row">
                      <span>可疑元数据</span>
                      <b>
                        {report.container.suspicious?.length
                          ? report.container.suspicious.join("、")
                          : "无"}
                      </b>
                    </div>
                  </div>

                  <div className="section-title">命中项</div>
                  {report.bitstream.flags.length > 0 ? (
                    <ul className="suggest">
                      {report.bitstream.flags.map((flag) => (
                        <li key={flag}>{flag}</li>
                      ))}
                    </ul>
                  ) : (
                    <p className="note">未发现明显码流异常。</p>
                  )}

                  {report.blind && (
                    <>
                      <div className="section-title">盲检测（启发式）</div>
                      <div className="kv-card">
                        <div className="kv-row">
                          <span>空域扩频 SS</span>
                          <b>{report.blind.ss.toFixed(2)}</b>
                        </div>
                        <div className="kv-row">
                          <span>DCT-QIM 量化格</span>
                          <b>{report.blind.qim.toFixed(2)}</b>
                        </div>
                        {report.blind.echo !== undefined && (
                          <div className="kv-row">
                            <span>音频回声</span>
                            <b>{report.blind.echo.toFixed(2)}</b>
                          </div>
                        )}
                      </div>
                      <p className="note">
                        0~1 置信度，单项分数仅供参考，需干净同源基准做差分判定。
                      </p>
                    </>
                  )}

                  <div className="section-title">处理校验</div>
                  {lastClean ? (
                    <>
                      <div className="metric-grid">
                        <div className="metric">
                          <div className="metric-value">
                            {lastClean.psnr_db != null ? `${lastClean.psnr_db} dB` : "—"}
                          </div>
                          <div className="metric-label">PSNR</div>
                        </div>
                        <div className="metric">
                          <div className="metric-value">{lastClean.ssim ?? "—"}</div>
                          <div className="metric-label">SSIM</div>
                        </div>
                        <div className="metric">
                          <div className="metric-value">
                            {lastClean.vmaf_aligned != null
                              ? lastClean.vmaf_aligned.toFixed(1)
                              : lastClean.vmaf != null
                                ? lastClean.vmaf.toFixed(1)
                                : "—"}
                          </div>
                          <div className="metric-label">VMAF（对齐）</div>
                        </div>
                        <div className="metric">
                          <div className="metric-value">
                            {dedupRisk
                              ? `${(dedupRisk.duplicate_risk * 100).toFixed(0)}% · ${dedupRisk.risk_level}`
                              : "—"}
                          </div>
                          <div className="metric-label">判重风险（相对原片）</div>
                        </div>
                      </div>
                      {lastOutput && (
                        <div className="kv-row mt-2">
                          <span className="mono truncate text-xs text-muted-foreground">
                            {lastOutput}
                          </span>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => openInFolder(lastOutput)}
                          >
                            打开文件夹
                          </Button>
                        </div>
                      )}
                      {lastOutput && material?.path && (
                        <Button
                          variant="secondary"
                          size="sm"
                          className="mt-2 w-full"
                          onClick={() =>
                            runComparePair(material.path as string, lastOutput, "原片", "处理后")
                          }
                        >
                          与原片同屏对比（同步播放）
                        </Button>
                      )}
                      <Button
                        variant="secondary"
                        size="sm"
                        className="mt-2 w-full"
                        disabled={candidatesBusy}
                        onClick={runCandidates}
                      >
                        {candidatesBusy ? "生成中…" : "生成候选（多版本优选）"}
                      </Button>
                      {candidates && (
                        <div className="mt-2 flex flex-col gap-1">
                          {candidates.map((candidate) => (
                            <div key={candidate.index} className="mono text-xs text-muted-foreground">
                              候选{candidate.index} · 评分 {candidate.score} · 内容{" "}
                              {candidate.content_cosine.toFixed(3)} · 稳定 {candidate.stability_ratio}
                              {candidate.duplicate_risk != null &&
                                ` · 判重 ${(candidate.duplicate_risk * 100).toFixed(0)}%`}
                            </div>
                          ))}
                        </div>
                      )}
                    </>
                  ) : (
                    <p className="note">
                      清洗后展示 PSNR / SSIM / VMAF 与清除复核结果。
                    </p>
                  )}

                  <div className="section-title">音频分析</div>
                  {audio ? (
                    <>
                      <div className="kv-row">
                        <span>回声隐藏置信度</span>
                        <b>{audio.echo_score.toFixed(2)}</b>
                      </div>
                      <svg
                        viewBox={`0 0 ${Math.max(1, audio.waveform.length - 1)} 40`}
                        preserveAspectRatio="none"
                        className="h-10 w-full"
                        aria-label="音轨波形"
                      >
                        <polyline
                          points={audio.waveform
                            .map((value, index) => `${index},${20 - value * 18}`)
                            .join(" ")}
                          fill="none"
                          stroke="var(--primary)"
                          strokeWidth="1.2"
                        />
                      </svg>
                      <div className="audio-spectrum" aria-label="对数频谱">
                        {audio.spectrum.map((value, index) => (
                          <span key={index} style={{ height: `${Math.max(4, value * 100)}%` }} />
                        ))}
                      </div>
                      <p className="note">
                        回声置信度 0~1，压缩与转码会抬高基线。
                      </p>
                    </>
                  ) : audioMissing ? (
                    <p className="note">该素材没有音轨或音轨无法解析。</p>
                  ) : (
                    <p className="note">正在分析音轨…</p>
                  )}

                  <div className="section-title">清洗建议</div>
                  <ul className="suggest">
                    {(() => {
                      const suggestions: string[] = [];
                      if (report.bitstream.flags.length > 0) {
                        suggestions.push(
                          `码流层命中 ${report.bitstream.flags.length} 项疑似特征，建议开启清洗并人工复核`,
                        );
                      }
                      if (report.blind) {
                        if (report.blind.ss > 0.5 || report.blind.qim > 0.6) {
                          suggestions.push(
                            `空域/频域疑似度偏高（ss ${report.blind.ss.toFixed(2)}、qim ${report.blind.qim.toFixed(2)}），建议开启空间降噪与 DCT 重量化`,
                          );
                        }
                        if ((report.blind.echo ?? 0) > 0.6) {
                          suggestions.push(
                            `音频回声置信度 ${(report.blind.echo ?? 0).toFixed(2)}，建议同步音频重混`,
                          );
                        }
                      }
                      if (score >= 60) {
                        suggestions.push(`判重风险评分 ${score}，建议至少开启标准指纹对抗档`);
                      }
                      if (score >= 40 && antiLevel === "关闭") {
                        suggestions.push("当前指纹对抗关闭，建议至少开启轻度档");
                      }
                      if (suggestions.length === 0) {
                        suggestions.push(
                          "未命中明显异常，保持基础清洗即可",
                        );
                      }
                      return suggestions.map((text) => <li key={text}>{text}</li>);
                    })()}
                  </ul>
                </>
              ) : (
                <>
                  <Card className="flex items-center gap-3.5 border border-border p-3.5 shadow-none ring-0">
                    <div
                      className="ring"
                      style={{ background: `conic-gradient(${ringColor} 0 ${pct}%, var(--muted) ${pct}% 100%)` }}
                    >
                      <div className="ring-inner">
                        <span className="ring-score">{risk === "待检测" ? "—" : score}</span>
                      </div>
                    </div>
                    <div>
                      <div className="score-title">{riskLabel(risk)}</div>
                      <div className="score-sub">{scoreSub}</div>
                    </div>
                  </Card>

                  <Card className="border border-border p-4 shadow-none ring-0">
                    <p className="note">
                      尚未检测，检测后此处展示码流异常、元数据与命中维度。
                    </p>
                  </Card>
                </>
              )}
            </div>
          </ScrollArea>
        </TabsContent>

        <TabsContent value="水印区域" className="tab-pane">
          <ScrollArea className="h-full">
            <div className="flex flex-col gap-2.5">
              <div className="undo-row">
                <Button variant="ghost" size="sm" disabled={undoStack.length === 0} onClick={undo}>
                  <Undo2 />
                  撤销
                </Button>
                <Button variant="ghost" size="sm" disabled={redoStack.length === 0} onClick={redo}>
                  <Redo2 />
                  重做
                </Button>
              </div>

              {regions.length === 0 ? (
                <div className="mat-empty">还没有水印区域，点击下方按钮添加</div>
              ) : (
                regions.map((r) => (
                  <div
                    key={r.id}
                    className={cn("region-card", r.id === activeRegionId && "active")}
                    onClick={() => selectRegion(r.id)}
                  >
                    <div className="region-head">
                      <span className="region-name">{r.name}</span>
                      <span className="region-tools">
                        <Button
                          variant="ghost"
                          size="icon"
                          className="size-6"
                          aria-label={`删除${r.name}`}
                          onClick={(e) => {
                            e.stopPropagation();
                            deleteRegion(r.id);
                          }}
                        >
                          <Trash2 />
                        </Button>
                      </span>
                    </div>
                    <div className="field">
                      <span className="field-label">生效时间（秒，留空为全程）</span>
                      <div className="flex gap-2">
                        <Input
                          className="h-8"
                          type="number"
                          min={0}
                          placeholder="开始"
                          value={r.start ?? ""}
                          onChange={(e) =>
                            patchRegion(r.id, {
                              start: e.target.value === "" ? null : Number(e.target.value),
                            })
                          }
                        />
                        <Input
                          className="h-8"
                          type="number"
                          min={0}
                          placeholder="结束"
                          value={r.end ?? ""}
                          onChange={(e) =>
                            patchRegion(r.id, {
                              end: e.target.value === "" ? null : Number(e.target.value),
                            })
                          }
                        />
                      </div>
                    </div>
                  </div>
                ))
              )}

              <Button
                variant="secondary"
                size="sm"
                className="w-full"
                onClick={() => {
                  addRegion();
                  toast("已添加水印区域，可在预览中拖动调整");
                }}
              >
                <Plus />
                添加水印区域
              </Button>
              <Button variant="secondary" size="sm" className="w-full" onClick={runRepairJob}>
                执行修复
              </Button>
              <p className="note">
                在预览中拖动选框圈住水印位置，可调整大小与参数。
              </p>
            </div>
          </ScrollArea>
        </TabsContent>

        <TabsContent value="清洗去重" className="tab-pane">
          <ScrollArea className="h-full">
            <div className="flex flex-col gap-2.5">
              <div className="field">
                <span className="field-label">模板</span>
                <Select value={templateValue} onValueChange={applyTemplateById}>
                  <SelectTrigger className="form-input h-8 w-full">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="manual">不使用模板（手动）</SelectItem>
                    {templateList.length === 0 ? (
                      <SelectItem value="__empty__" disabled>
                        还没有模板，可到「去重模板」页创建
                      </SelectItem>
                    ) : (
                      templateList.map((item) => (
                        <SelectItem key={item.id} value={item.id}>
                          {item.name}
                        </SelectItem>
                      ))
                    )}
                  </SelectContent>
                </Select>
                <div className="form-help">选择后自动回填下方参数，可继续调整</div>
              </div>
              <p className="note">
                针对空域、DCT-QIM、小波、LSB 与音频回声五类常见水印；未知方案建议加强对抗档。
              </p>
              <div className="section-title">清除档位</div>
              <Tabs
                value={level}
                onValueChange={(value) => {
                  const next = value as CleanLevel;
                  setLevel(next);
                  setRetime(LEVEL_PRESETS[next].retime);
                  setPerturb(LEVEL_PRESETS[next].perturb);
                }}
              >
                <TabsList className="grid w-full grid-cols-3">
                  {CLEAN_LEVELS.map((l) => (
                    <TabsTrigger key={l} value={l} className="px-1">
                      {l}
                    </TabsTrigger>
                  ))}
                </TabsList>
              </Tabs>
              <div className="param-table">
                <div className="param-row">
                  <span>变速倍率</span>
                  <b>{levelDisplay(level, recropOn).speed.toFixed(3)}×</b>
                </div>
                <div className="param-row">
                  <span>调光微扰</span>
                  <b>
                    γ±{levelDisplay(level, recropOn).gamma.toFixed(2)} · 亮度±
                    {levelDisplay(level, recropOn).brightness.toFixed(2)}
                  </b>
                </div>
                <div className="param-row">
                  <span>重新构图</span>
                  <b>
                    {recropOn
                      ? `四周裁 ${(levelDisplay(level, recropOn).crop * 100).toFixed(1)}%`
                      : "关闭（见下方开关）"}
                  </b>
                </div>
              </div>
              <p className="note">
                三档只控制变速与微扰强度，指纹对抗与编码在下方独立设置。
              </p>

              <div className="field">
                <span className="field-label">
                  变速幅度 <span className="field-value">{retime}%</span>
                </span>
                <Slider value={[retime]} min={0} max={100} onValueChange={([v]) => setRetime(v)} />
              </div>
              <div className="field">
                <span className="field-label">
                  画质微扰强度 <span className="field-value">{perturb}%</span>
                </span>
                <Slider value={[perturb]} min={0} max={100} onValueChange={([v]) => setPerturb(v)} />
              </div>
              <div className="switch">
                <div>
                  <div className="switch-label">同步处理音频指纹</div>
                  <div className="switch-desc">对音轨做频谱轻微处理</div>
                </div>
                <Switch checked={audioClean} onCheckedChange={setAudioClean} />
              </div>
              <div className="switch">
                <div>
                  <div className="switch-label">抗二次检测增强</div>
                  <div className="switch-desc">扰动中频 DCT 系数，破坏二次嵌入</div>
                </div>
                <Switch checked={antiReembed} onCheckedChange={setAntiReembed} />
              </div>
              <div className="switch">
                <div>
                  <div className="switch-label">重新构图（裁剪回缩）</div>
                  <div className="switch-desc">对抗内容指纹，画质损失较大</div>
                </div>
                <Switch checked={recropOn} onCheckedChange={setRecropOn} />
              </div>
              <div className="switch">
                <div>
                  <div className="switch-label">细节保护（人脸/字幕/纹理）</div>
                  <div className="switch-desc">保护区域回退原帧，保留部分水印特征</div>
                </div>
                <Switch checked={detailProtectOn} onCheckedChange={setDetailProtectOn} />
              </div>
              <div className="field">
                <span className="field-label">指纹对抗强度</span>
                <Select value={antiLevel} onValueChange={setAntiLevel}>
                  <SelectTrigger className="form-input h-9 w-full">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="关闭">关闭（仅基础清洗）</SelectItem>
                    <SelectItem value="轻度">轻度 · 几乎无损，日常推荐</SelectItem>
                    <SelectItem value="标准">标准 · 逐镜头变速 + 强音频，轻微损失</SelectItem>
                    <SelectItem value="强力">强力 · 幅度更大，可见轻微加工</SelectItem>
                    <SelectItem value="全兵器">极限 · 仅研究测试，不保证观感</SelectItem>
                  </SelectContent>
                </Select>
                <div className="form-help">
                  轻度/标准日常可用；强力可见轻微加工；极限仅供研究。语义级指纹提升有限，建议配合多版本分发
                </div>
              </div>

              <div className="section-title">画质优化</div>
              <div className="switch">
                <div>
                  <div className="switch-label">锐度补偿</div>
                  <div className="switch-desc">抵消清除算法带来的轻微模糊</div>
                </div>
                <Switch checked={sharpness} onCheckedChange={setSharpness} />
              </div>
              <div className="switch">
                <div>
                  <div className="switch-label">色彩还原</div>
                  <div className="switch-desc">把处理后亮度均值校准回原片</div>
                </div>
                <Switch checked={colorFix} onCheckedChange={setColorFix} />
              </div>
              <div className="switch">
                <div>
                  <div className="switch-label">空间降噪</div>
                  <div className="switch-desc">破坏空域扩频水印</div>
                </div>
                <Switch checked={aiDenoise} onCheckedChange={setAiDenoise} />
              </div>
              <div className="switch">
                <div>
                  <div className="switch-label">伪水印注入（溯源干扰）</div>
                  <div className="switch-desc">注入随机干扰水印，轻微损失画质</div>
                </div>
                <Switch checked={spoof} onCheckedChange={setSpoof} />
              </div>

              <div className="section-title">输出编码 · MP4</div>
              <div className="field">
                <span className="field-label">视频编码</span>
                <Select value={codec} onValueChange={setCodec}>
                  <SelectTrigger className="h-9 w-full"><SelectValue /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="H.264">H.264</SelectItem>
                    <SelectItem value="H.265">H.265</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="field">
                <span className="field-label">分辨率策略</span>
                <Select value={resolution} onValueChange={setResolution}>
                  <SelectTrigger className="h-9 w-full"><SelectValue /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="保持原始分辨率">保持原始分辨率</SelectItem>
                    <SelectItem value="1920x1080">1920×1080</SelectItem>
                    <SelectItem value="1280x720">1280×720</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="field-row">
                <div className="field">
                  <span className="field-label">码率（kbps）</span>
                  <Input
                    className="h-9"
                    type="number"
                    min={100}
                    value={bitrate}
                    onChange={(e) => setBitrate(e.target.value)}
                    placeholder="留空使用 CRF"
                  />
                </div>
                <div className="field">
                  <span className="field-label">帧率（fps）</span>
                  <Input
                    className="h-9"
                    type="number"
                    min={1}
                    value={fpsOut}
                    onChange={(e) => setFpsOut(e.target.value)}
                    placeholder="留空原帧率"
                  />
                </div>
                <div className="field">
                  <span className="field-label">GOP 帧数</span>
                  <Input
                    className="h-9"
                    type="number"
                    min={1}
                    value={gop}
                    onChange={(e) => setGop(e.target.value)}
                    placeholder="留空自动"
                  />
                </div>
              </div>
              <div className="switch">
                <div>
                  <div className="switch-label">无损输出</div>
                  <div className="switch-desc">极致保留画质，文件体积相应增大</div>
                </div>
                <Switch checked={lossless} onCheckedChange={setLossless} />
              </div>

              <Button variant="secondary" className="w-full" disabled={cleaning} onClick={runClean}>
                {cleaning ? "处理中…" : "执行清洗"}
              </Button>
            </div>
          </ScrollArea>
        </TabsContent>

        <TabsContent value="处理产物" className="tab-pane">
          <ScrollArea className="h-full">
            <div className="flex min-w-0 flex-col gap-2.5">
              {outputs.length === 0 ? (
                <p className="note">
                  还没有处理产物，清洗或生成候选后会自动出现。
                </p>
              ) : (
                <>
                  <p className="note">与原片对比，或播放查看。</p>
                  {outputs.map((output) => (
                    <div key={output.path} className="out-item">
                      <div className="out-item-head">
                        <span className="out-kind">{OUTPUT_KIND_LABEL[output.kind]}</span>
                        <span className="out-name" title={output.name}>{output.name}</span>
                      </div>
                      <div className="out-meta">
                        {fmtSize(output.size)} · {outputTimeLabel(output.mtime)}
                      </div>
                      <div className="out-item-actions">
                        <Button
                          variant="secondary"
                          size="sm"
                          className="w-full"
                          onClick={() => compareOriginal(output)}
                        >
                          <GitCompareArrows /> 对比原片
                        </Button>
                        <div className="out-item-secondary">
                          <Button
                            variant="ghost"
                            size="sm"
                            className="flex-1"
                            onClick={() => setPlayerPath(output.path)}
                          >
                            播放
                          </Button>
                          <Button
                            variant="ghost"
                            size="sm"
                            className="flex-1"
                            onClick={() => openInFolder(output.path)}
                          >
                            打开
                          </Button>
                          <Button
                            variant="ghost"
                            size="sm"
                            className="flex-1"
                            onClick={() => showVariant(output.path)}
                          >
                            参数
                          </Button>
                          <Button
                            variant="ghost"
                            size="sm"
                            className="flex-1 text-destructive"
                            onClick={() => setPendingDelete(output)}
                          >
                            删除
                          </Button>
                        </div>
                      </div>
                    </div>
                  ))}
                </>
              )}
            </div>
          </ScrollArea>
        </TabsContent>

      </Tabs>

      <Dialog open={!!variantDetail} onOpenChange={(open) => !open && setVariantDetail(null)}>
        <DialogContent className="max-w-[min(92vw,520px)]">
          <DialogHeader>
            <DialogTitle>产物参数</DialogTitle>
            <DialogDescription>{variantDetail?.output}</DialogDescription>
          </DialogHeader>
          {variantDetail && (
            <div className="flex flex-col gap-2 text-sm">
              <div className="kv-row">
                <span>源素材</span>
                <b className="mono truncate">{variantDetail.source}</b>
              </div>
              <div className="kv-row">
                <span>随机种子</span>
                <b className="mono">{variantDetail.seed}</b>
              </div>
              <div className="kv-row">
                <span>创建时间</span>
                <b className="mono">
                  {new Date(variantDetail.created_at * 1000).toLocaleString()}
                </b>
              </div>
              {variantDetail.metrics.duplicate_risk != null && (
                <div className="kv-row">
                  <span>判重风险</span>
                  <b>
                    {((variantDetail.metrics.duplicate_risk as number) * 100).toFixed(0)}%
                  </b>
                </div>
              )}
              {variantDetail.metrics.ssim != null && (
                <div className="kv-row">
                  <span>SSIM</span>
                  <b>{String(variantDetail.metrics.ssim)}</b>
                </div>
              )}
              <div className="section-title">完整参数</div>
              <pre className="max-h-64 overflow-auto rounded-md border border-border bg-muted/40 p-2 text-xs leading-5">
                {JSON.stringify(variantDetail.options, null, 2)}
              </pre>
            </div>
          )}
        </DialogContent>
      </Dialog>

      <Dialog open={!!playerPath} onOpenChange={(open) => !open && setPlayerPath(null)}>
        <DialogContent className="max-w-[min(90vw,720px)]">
          <DialogHeader>
            <DialogTitle>产物预览</DialogTitle>
            <DialogDescription>播放产物视频，确认观感。</DialogDescription>
          </DialogHeader>
          {playerPath && (
            <video
              src={mediaUrl(playerPath)}
              controls
              autoPlay
              className="max-h-[70vh] w-full rounded-md bg-black"
            />
          )}
        </DialogContent>
      </Dialog>

      <AlertDialog
        open={!!pendingDelete}
        onOpenChange={(open) => !open && setPendingDelete(null)}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>确认删除产物</AlertDialogTitle>
            <AlertDialogDescription>
              将删除「{pendingDelete?.name}」，文件会从磁盘移除，此操作不可恢复。
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>取消</AlertDialogCancel>
            <AlertDialogAction onClick={() => void confirmDelete()}>确认删除</AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </aside>
  );
}
