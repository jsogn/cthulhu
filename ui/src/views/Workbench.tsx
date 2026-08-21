import { useEffect, useRef, useState } from "react";
import type { KeyboardEvent as ReactKeyboardEvent, PointerEvent as ReactPointerEvent } from "react";
import {
  Download,
  ListPlus,
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
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
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
import { fmtFrames, nowStr } from "@/lib/format";
import {
  analyzeAudio,
  enqueueJob,
  exportOutputs,
  frameUrl,
  generateCandidates,
  listTemplates,
  makePreview,
  mediaUrl,
  previewImageUrl,
  runDesensitize,
  runDetect,
  thumbUrl,
  type AudioAnalysis,
  type CandidateInfo,
  type DetectReport,
  type DesensitizeOptions,
  type JobInfo,
  type TemplateInfo,
} from "@/lib/backend";
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
const TAB_KEYS = ["合规清洗", "水印区域", "检测参考", "导出设置"] as const;
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

const CLEAN_LEVELS = ["轻度", "平衡", "深度"] as const;
type CleanLevel = (typeof CLEAN_LEVELS)[number];

// PRD 5.6 三档清除强度核心参数对照表。
const LEVEL_PARAMS: Record<
  CleanLevel,
  { dct: string; shift: string; lsb: string; crf: number; rate: string }
> = {
  轻度: { dct: "5 步长", shift: "±0.1px", lsb: "30%", crf: 23, rate: "≈85%" },
  平衡: { dct: "10 步长", shift: "±0.2px", lsb: "50%", crf: 24, rate: "≈95%" },
  深度: { dct: "18 步长", shift: "±0.4px", lsb: "70%", crf: 25, rate: "≈99%" },
};

const LEVEL_PRESETS: Record<CleanLevel, { restruct: number; perturb: number }> = {
  轻度: { restruct: 25, perturb: 15 },
  平衡: { restruct: 30, perturb: 20 },
  深度: { restruct: 40, perturb: 30 },
};

// PRD 3.2.4 场景模板预设。
const SCENES: Record<string, { level: CleanLevel; restruct: number; perturb: number; audio: boolean; codec: string }> = {
  抖音投流: { level: "轻度", restruct: 25, perturb: 15, audio: true, codec: "H.264" },
  快手分发: { level: "平衡", restruct: 30, perturb: 20, audio: true, codec: "H.264" },
  跨平台通用: { level: "深度", restruct: 40, perturb: 30, audio: true, codec: "H.265" },
};

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
  },
};

/** 单条清洗走 /api/desensitize（camelCase），批量走 /api/jobs（snake_case）。 */
const snakeAnti = (anti: AntiPreset) => ({
  rotate: anti.rotate,
  phash_attack: !anti.jointAttack,
  phash_epsilon: anti.epsilon,
  median: anti.median ?? 0,
  noise: anti.noise ?? 0,
  requant: anti.requant ?? 0,
  dct_step: anti.dctStep ?? 0,
  drop_every: anti.dropEvery ?? 0,
  jitter: anti.jitter ?? 0,
  perspective: anti.perspective ?? 0,
  warp: anti.warp ?? 0,
  mirror: anti.mirror ?? false,
  chroma_levels: anti.chromaLevels ?? 0,
  subtract_beta: anti.subtractBeta ?? 0,
  transcode_chain: anti.transcodeChain ?? false,
  multi_hash_attack: anti.jointAttack ?? false,
  saliency: anti.saliency ?? 0,
  native_filters: anti.nativeFilters ?? false,
});

function riskLabel(risk: RiskLevel): string {
  return risk;
}

function riskClass(risk: RiskLevel): string {
  return risk === "有疑似特征" ? "suspect" : risk === "未检出异常" ? "clear" : "pending";
}

export default function Workbench() {
  const materials = useMaterialsStore((state) => state.materials);
  const activeId = useMaterialsStore((state) => state.activeId);
  const selected = useMaterialsStore((state) => state.selected);
  const addHistory = useHistoryStore((state) => state.add);

  const active = materials.find((m) => m.id === activeId) ?? null;
  const [tab, setTab] = useState<(typeof TAB_KEYS)[number]>("合规清洗");
  const [enqueued, setEnqueued] = useState(false);

  const [frame, setFrame] = useState(132);
  const [playing, setPlaying] = useState(false);
  const [markers, setMarkers] = useState<number[]>([]);
  const [comparePct, setComparePct] = useState(50);
  const frameRef = useRef(frame);
  frameRef.current = frame;

  const handleEnqueue = async () => {
    if (enqueued) return;
    const target = active as (Material & { path?: string }) | null;
    if (!target?.path) {
      toast("该素材缺少本地路径，无法执行");
      return;
    }
    setEnqueued(true);
    try {
      const output = `${target.path.replace(/\.(mp4|mov|mkv|avi|flv|ts)$/i, "")}_cleaned.mp4`;
      await enqueueJob(`${target.name} · 清洗`, [
        {
          kind: "desensitize",
          path: target.path,
          options: {
            output,
            reorder: true,
            speed: 0.95,
            regrade: true,
            perturb: 0.2,
            audio_remix: true,
            sharpness: true,
            color_restore: true,
            seed: Math.floor(Math.random() * 1_000_000),
          },
        },
      ]);
      addHistory({
        name: target.name,
        time: nowStr(),
        action: "加入处理队列",
        params: "平衡档 · 默认参数",
        out: output,
        result: "排队中",
      });
      toast("已加入任务队列");
    } catch (error) {
      toast(error instanceof Error ? error.message : "入队失败");
    } finally {
      window.setTimeout(() => setEnqueued(false), 1600);
    }
  };

  // 全局快捷键（工作台内生效）。
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      const typing =
        !!target &&
        (target.tagName === "INPUT" ||
          target.tagName === "TEXTAREA" ||
          target.tagName === "SELECT" ||
          target.isContentEditable);

      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "enter") {
        e.preventDefault();
        handleEnqueue();
        return;
      }
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "z") {
        if (typing) return;
        e.preventDefault();
        if (e.shiftKey) {
          useRegionsStore.getState().redo();
        } else {
          useRegionsStore.getState().undo();
        }
        return;
      }

      if (typing || e.altKey) return;
      const interactive =
        !!target &&
        (target.tagName === "BUTTON" || !!target.closest("[role='tab']") || !!target.closest(".sel-box"));
      if (interactive) return;

      if (e.key === " ") {
        e.preventDefault();
        setPlaying((p) => !p);
      } else if (e.key === "ArrowLeft") {
        e.preventDefault();
        setPlaying(false);
        setFrame((f) => Math.max(0, f - 1));
      } else if (e.key === "ArrowRight") {
        e.preventDefault();
        setPlaying(false);
        setFrame((f) => Math.min(FRAME_MAX, f + 1));
      } else if (e.key.toLowerCase() === "v") {
        e.preventDefault();
        setComparePct((p) => (p === 100 ? 50 : 100));
      } else if (e.key.toLowerCase() === "b") {
        e.preventDefault();
        const f = frameRef.current;
        setMarkers((ms) => (ms.includes(f) ? ms : [...ms, f].sort((a, b) => a - b)));
        const fps = Number.parseFloat(active?.fps ?? "30") || 30;
        toast(`已打点 ${fmtFrames(f, fps)}`);
      } else if (e.key.toLowerCase() === "c") {
        e.preventDefault();
        setMarkers([]);
        toast("已清除全部标记");
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  return (
    <section className="workbench">
      <MaterialPane />
      <PreviewPane
        material={active}
        frame={frame}
        setFrame={setFrame}
        playing={playing}
        setPlaying={setPlaying}
        markers={markers}
        comparePct={comparePct}
        setComparePct={setComparePct}
        onEnqueue={handleEnqueue}
      />
      <ContextPanel tab={tab} setTab={setTab} onEnqueue={handleEnqueue} enqueued={enqueued} />
      {Object.values(selected).length > 0 && (
        <BatchBar count={Object.values(selected).length} />
      )}
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
  const addHistory = useHistoryStore((state) => state.add);
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

  const runExport = async () => {
    const paths = visible
      .filter((m) => !!m.path)
      .map((m) => m.path as string);
    if (!paths.length) {
      toast("没有可导出的素材，请先导入本地视频");
      return;
    }
    try {
      const { exported, missing } = await exportOutputs(paths);
      if (exported.length) {
        addHistory({
          name: `${exported.length} 个素材`,
          time: nowStr(),
          action: "导出",
          params: "清洗/修复产物",
          out: `导出目录 · ${exported.length} 个`,
          result: "成功",
        });
      }
      const parts: string[] = [];
      if (exported.length) parts.push(`已导出 ${exported.length} 个产物到导出目录`);
      if (missing.length) parts.push(`${missing.length} 个素材尚未处理，无产物可导出`);
      const firstDest = exported[0]?.dest;
      toast(
        parts.length ? parts.join(" · ") : "所选素材均未处理，请先清洗或修复",
        undefined,
        firstDest
          ? { label: "打开文件夹", onClick: () => openInFolder(firstDest) }
          : undefined,
      );
    } catch (error) {
      toast(error instanceof Error ? error.message : "导出失败");
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
          <Button variant="secondary" size="sm" onClick={() => setImportOpen(true)}>
            <Upload />
            导入
          </Button>
          <Button variant="ghost" size="sm" onClick={() => void runExport()}>
            <Download />
            导出
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

/* ============ 中栏：预览 ============ */
interface PreviewProps {
  material: Material | null;
  frame: number;
  setFrame: (value: number) => void;
  playing: boolean;
  setPlaying: (value: boolean) => void;
  markers: number[];
  comparePct: number;
  setComparePct: (value: number) => void;
  onEnqueue: () => void;
}

function PreviewPane({
  material,
  frame,
  setFrame,
  playing,
  setPlaying,
  markers,
  comparePct,
  setComparePct,
}: PreviewProps) {
  const regions = useRegionsStore((state) => state.regions);
  const activeRegionId = useRegionsStore((state) => state.activeId);
  const patchRegion = useRegionsStore((state) => state.patchRegion);
  const beginMove = useRegionsStore((state) => state.beginMove);

  const stageRef = useRef<HTMLDivElement>(null);
  const mediaBoxRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
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
    if (!video || !hasRealFrame) return;
    if (playing) {
      if (video.ended) video.currentTime = 0;
      video.play().catch(() => setPlaying(false));
    } else {
      video.pause();
      setFrame(Math.min(maxFrame, Math.round(video.currentTime * fps)));
    }
  }, [playing, hasRealFrame, setPlaying, setFrame, fps, maxFrame, material?.id]);

  // 暂停状态下，时间轴/逐帧/快捷键改变帧号时把视频 seek 到对应时间；
  // 播放中由视频自身推进，避免帧号回灌造成卡顿。
  useEffect(() => {
    if (playing) return;
    const video = videoRef.current;
    if (!video || !hasRealFrame || video.readyState < 1) return;
    const target = frame / fps;
    if (Math.abs(video.currentTime - target) > 0.5 / fps) {
      video.currentTime = target;
    }
  }, [frame, fps, playing, hasRealFrame, material?.id]);

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
        <span className="demo-badge">{hasRealFrame ? "实时预览" : "预览不可用"}</span>
        {material && <span className="preview-meta mono">{material.res} · {material.fps}</span>}
      </div>

      <div className="preview-stage" ref={stageRef}>
        <div
          ref={mediaBoxRef}
          className={hasRealFrame ? "media-box" : "media-box fill"}
          style={hasRealFrame && fitBox ? { width: fitBox.w, height: fitBox.h } : undefined}
          onDoubleClick={onCompareDoubleClick}
        >
          {hasRealFrame ? (
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
          <div className="marks">
            {markers.map((f) => (
              <span
                key={f}
                className="mark-dot"
                style={{ left: `${(f / maxFrame) * 100}%` }}
                title={`标记 ${fmtFrames(f, fps)}`}
              />
            ))}
          </div>
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

      <div className="shortcut-hint">
        空格 播放 · ←/→ 逐帧 · V 对比 · 双击放大 · B 打点 · C 清除标记 · ⌘/Ctrl+Enter 入队 · ⌘/Ctrl+Z 撤销
      </div>
    </section>
  );
}

/* ============ 右栏：上下文面板 ============ */
interface ContextProps {
  tab: (typeof TAB_KEYS)[number];
  setTab: (tab: (typeof TAB_KEYS)[number]) => void;
  onEnqueue: () => void;
  enqueued: boolean;
}

function ContextPanel({ tab, setTab, onEnqueue, enqueued }: ContextProps) {
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

  // PRD 3.1.2 清除档位与 3.2.4 场景模板。
  const [scene, setScene] = useState<string>("无");
  const [level, setLevel] = useState<CleanLevel>("平衡");
  const [restruct, setRestruct] = useState(30);
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
  const [exportReport, setExportReport] = useState(true);
  const [resolution, setResolution] = useState("保持原始分辨率");
  const [bitrate, setBitrate] = useState("");
  const [gop, setGop] = useState("");
  const [fpsOut, setFpsOut] = useState("");
  const [outputDir, setOutputDir] = useState("");
  const [naming, setNaming] = useState("原文件名_cleaned");
  const [cleaning, setCleaning] = useState(false);
  const [detectSubmitting, setDetectSubmitting] = useState(false);
  const [audio, setAudio] = useState<AudioAnalysis | null>(null);
  const [audioMissing, setAudioMissing] = useState(false);
  const [lastClean, setLastClean] = useState<{
    psnr_db?: number;
    ssim?: number;
    vmaf?: number | null;
    vmaf_aligned?: number | null;
  } | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [previewOpen, setPreviewOpen] = useState(false);
  const [candidates, setCandidates] = useState<CandidateInfo[] | null>(null);
  const [candidatesBusy, setCandidatesBusy] = useState(false);
  const lastOptionsRef = useRef<DesensitizeOptions | null>(null);

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

  const applyScene = (name: string) => {
    setScene(name);
    const preset = SCENES[name];
    if (!preset) return;
    setLevel(preset.level);
    setRestruct(preset.restruct);
    setPerturb(preset.perturb);
    setAudioClean(preset.audio);
    setCodec(preset.codec);
    toast(`已应用场景模板：${name}`);
  };

  const runClean = async () => {
    const target = material as (Material & { path?: string }) | null;
    if (!target?.path || !material) {
      toast("该素材缺少本地路径，无法执行");
      return;
    }
    const srcStem = target.path.replace(/\.(mp4|mov|mkv|avi|flv|ts)$/i, "");
    const srcName = srcStem.split(/[\\/]/).pop() ?? srcStem;
    const ts = new Date().toISOString().replace(/[-:TZ]/g, "").slice(0, 14);
    const fileName =
      naming === "前缀 + 时间戳" ? `${ts}_${srcName}_cleaned.mp4` : `${srcName}_cleaned.mp4`;
    const output = outputDir.trim()
      ? `${outputDir.trim().replace(/\/+$/, "")}/${fileName}`
      : `${srcStem}_cleaned.mp4`;
    const anti = ANTI_PRESETS[antiLevel];
    setCleaning(true);
    try {
      const cleanOptions: DesensitizeOptions = {
        reorder: restruct > 0,
        speed: Math.max(0.85, 1 - 0.15 * (restruct / 100)),
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
            }
          : {}),
      };
      lastOptionsRef.current = cleanOptions;
      const report = await runDesensitize(target.path, output, cleanOptions);
      addHistory({
        name: material.name,
        time: nowStr(),
        action: "合规清洗",
        params: `${level}档 · 重构${restruct}% · 微扰${perturb}% · 对抗${antiLevel} · ${codec}${lossless ? "无损" : ""}`,
        out: output,
        result: "成功",
      });
      setLastClean(report);
      setPreviewUrl(null);
      try {
        const preview = await makePreview(target.path, output);
        setPreviewUrl(previewImageUrl(preview.image_path));
      } catch {
        // 预览生成失败不阻断清洗主流程
      }
      if (report.psnr_db != null && report.psnr_db < 20) {
        toast(
          "画质损失偏大",
          `PSNR ${report.psnr_db}dB 低于 20dB 阈值，建议降低清洗档位或关闭部分增强开关`,
        );
      }
      let residualNote = "";
      try {
        const post = await runDetect(output);
        const residual = post.blind?.ss;
        if (residual != null) {
          residualNote = ` · 空间水印残留 ${residual.toFixed(2)}（干净基线约 0.28）`;
        }
      } catch {
        // 复检失败不阻断清洗主流程
      }
      toast(
        `清洗完成：内容相似度 ${report.similarity_after.content_cosine.toFixed(2)} · 时序乱序 ${(report.order_disruption ?? 0).toFixed(2)} · VMAF ${report.vmaf?.toFixed(1) ?? "—"}${residualNote}`,
        output,
        { label: "打开文件夹", onClick: () => openInFolder(output) },
      );
    } catch (error) {
      toast(error instanceof Error ? error.message : "清洗失败");
    } finally {
      setCleaning(false);
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
    if (!target?.path || !lastOptionsRef.current) {
      toast("请先执行一次清洗，再生成候选");
      return;
    }
    setCandidatesBusy(true);
    try {
      const dir = `${target.path.replace(/\.(mp4|mov|mkv|avi|flv|ts)$/i, "")}_候选`;
      const report = await generateCandidates(target.path, dir, 3, lastOptionsRef.current);
      setCandidates(report.candidates);
      toast(`已生成 ${report.candidates.length} 个候选，按低损优选排序`);
    } catch (error) {
      toast(error instanceof Error ? error.message : "生成候选失败");
    } finally {
      setCandidatesBusy(false);
    }
  };

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
      <div className="scene-row">
        <span className="section-title">场景模板</span>
        <Select value={scene} onValueChange={applyScene}>
          <SelectTrigger className="h-7 w-44 text-xs">
            <SelectValue placeholder="选择场景模板" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="无">无</SelectItem>
            <SelectItem value="抖音投流">抖音投流 · 轻度</SelectItem>
            <SelectItem value="快手分发">快手分发 · 平衡</SelectItem>
            <SelectItem value="跨平台通用">跨平台通用 · 深度</SelectItem>
          </SelectContent>
        </Select>
      </div>

      <Tabs value={tab} onValueChange={(value) => setTab(value as (typeof TAB_KEYS)[number])} className="min-h-0 flex-1">
        <TabsList variant="line" className="w-full shrink-0 justify-between rounded-none border-b border-border p-0">
          {TAB_KEYS.map((key) => (
            <TabsTrigger key={key} value={key} className="flex-1">
              {key}
            </TabsTrigger>
          ))}
        </TabsList>

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
                    <p className="note">未发现明显码流层异常（启发式结果，不替代平台实测）。</p>
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
                        0~1 置信度；真实视频基线随内容差异较大，且 LSB 位平面在压缩域
                        恒定误报（已不展示）。任何单项分数都不能直接判定，需干净同源
                        基准做差分判定。
                      </p>
                    </>
                  )}

                  <div className="section-title">处理校验（PRD 3.1.4）</div>
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
                      </div>
                      {previewUrl && (
                        <Button
                          variant="secondary"
                          size="sm"
                          className="mt-2 w-full"
                          onClick={() => setPreviewOpen(true)}
                        >
                          查看并排预览（确认观感）
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
                            </div>
                          ))}
                        </div>
                      )}
                    </>
                  ) : (
                    <p className="note">
                      执行清洗或修复后，此处将展示 PSNR / SSIM / VMAF 与清除复核结果。
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
                        波形与频谱供人工核验；回声置信度 0~1，压缩与转码会抬高基线。
                      </p>
                    </>
                  ) : audioMissing ? (
                    <p className="note">该素材没有音轨或音轨无法解析。</p>
                  ) : (
                    <p className="note">正在分析音轨…</p>
                  )}

                  <div className="section-title">清洗建议</div>
                  <ul className="suggest">
                    <li>分镜拆解后乱序重组，打散帧时序</li>
                    <li>施加低强度画质微扰，改变单帧频域分布</li>
                    <li>同步处理音频频谱指纹</li>
                    <li>内容指纹较高，建议优先复用叙事而非原画面像素</li>
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
                      尚未检测。扫描完成后，此处将展示码流层异常、容器元数据与命中维度等完整报告；当前素材没有可展示的检测数据。
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
                执行修复（本地文件）
              </Button>
              <p className="note">
                点击「添加水印区域」后，在预览中拖动选框圈住水印位置，可继续调整大小与参数；⌘/Ctrl+Z 撤销，⌘/Ctrl+Shift+Z 重做。
              </p>
            </div>
          </ScrollArea>
        </TabsContent>

        <TabsContent value="合规清洗" className="tab-pane">
          <ScrollArea className="h-full">
            <div className="flex flex-col gap-2.5">
              <p className="note">
                清洗已通过验收覆盖空域扩频、DCT-QIM、小波域、LSB 位平面与音频回声
                五个常见暗水印方案族；
                对未知方案或带纠错冗余的水印，建议使用平衡档以上并人工复核。
              </p>
              <div className="section-title">清除档位（PRD 5.6）</div>
              <Tabs
                value={level}
                onValueChange={(value) => {
                  const next = value as CleanLevel;
                  setLevel(next);
                  setRestruct(LEVEL_PRESETS[next].restruct);
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
                <div className="param-row"><span>DCT 收缩阈值</span><b>{LEVEL_PARAMS[level].dct}</b></div>
                <div className="param-row"><span>帧间位移幅度</span><b>{LEVEL_PARAMS[level].shift}</b></div>
                <div className="param-row"><span>LSB 翻转概率</span><b>{LEVEL_PARAMS[level].lsb}</b></div>
                <div className="param-row"><span>输出 CRF</span><b>{LEVEL_PARAMS[level].crf}</b></div>
                <div className="param-row"><span>清除率（基准库实测）</span><b>{LEVEL_PARAMS[level].rate}</b></div>
              </div>
              <p className="note">智能匹配：频域水印 → 中频系数软阈值收缩 · {level}档；强鲁棒水印建议深度档。</p>

              <div className="field">
                <span className="field-label">
                  分镜重构强度 <span className="field-value">{restruct}%</span>
                </span>
                <Slider value={[restruct]} min={0} max={100} onValueChange={([v]) => setRestruct(v)} />
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
                  <div className="switch-desc">亮度/像素/帧间微扰动，破坏二次嵌入条件</div>
                </div>
                <Switch checked={antiReembed} onCheckedChange={setAntiReembed} />
              </div>
              <div className="switch">
                <div>
                  <div className="switch-label">重新构图（裁剪回缩）</div>
                  <div className="switch-desc">
                    对抗内容指纹，但实测对真实素材画质损失大，默认关闭
                  </div>
                </div>
                <Switch checked={recropOn} onCheckedChange={setRecropOn} />
              </div>
              <div className="switch">
                <div>
                  <div className="switch-label">细节保护（人脸/字幕/纹理）</div>
                  <div className="switch-desc">
                    观感优先：保护区域回退原帧，但会保留部分水印特征，默认关闭
                  </div>
                </div>
                <Switch checked={detailProtectOn} onCheckedChange={setDetailProtectOn} />
              </div>
              <div className="field">
                <span className="field-label">指纹对抗强度</span>
                <Select value={antiLevel} onValueChange={setAntiLevel}>
                  <SelectTrigger className="form-input h-9">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="关闭">关闭（仅基础清洗）</SelectItem>
                    <SelectItem value="轻度">轻度 · 几乎无损，日常推荐</SelectItem>
                    <SelectItem value="标准">标准 · 轻微损失，正常观看</SelectItem>
                    <SelectItem value="强力">强力 · 可见轻微加工，建议预览</SelectItem>
                    <SelectItem value="全兵器">极限 · 仅研究测试，不保证观感</SelectItem>
                  </SelectContent>
                </Select>
                <div className="form-help">
                  观感优先：轻度/标准适合日常，强力有轻微可见加工，极限仅供素材研究
                </div>
              </div>

              <div className="section-title">画质优化（PRD 3.2.1）</div>
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
                  <div className="switch-desc">修正处理后色偏，偏差 ≤2%</div>
                </div>
                <Switch checked={colorFix} onCheckedChange={setColorFix} />
              </div>
              <div className="switch">
                <div>
                  <div className="switch-label">空间降噪</div>
                  <div className="switch-desc">Wiener 滤波，有效破坏空域扩频水印（默认开启）</div>
                </div>
                <Switch checked={aiDenoise} onCheckedChange={setAiDenoise} />
              </div>
              <div className="switch">
                <div>
                  <div className="switch-label">伪水印注入（溯源干扰）</div>
                  <div className="switch-desc">
                    注入随机干扰水印，可能在平台二次嵌入层面干扰溯源提取；
                    效果需投流实测验证，轻微损失画质
                  </div>
                </div>
                <Switch checked={spoof} onCheckedChange={setSpoof} />
              </div>
              <p className="note">默认低强度优先，需在预览中确认效果。</p>
              <Button variant="secondary" className="w-full" disabled={cleaning} onClick={runClean}>
                {cleaning ? "处理中…" : "执行清洗（本地文件）"}
              </Button>
            </div>
          </ScrollArea>
        </TabsContent>

        <TabsContent value="导出设置" className="tab-pane">
          <ScrollArea className="h-full">
            <div className="flex flex-col gap-2.5">
              <div className="field">
                <span className="field-label">输出格式</span>
                <Select defaultValue="MP4">
                  <SelectTrigger className="h-8">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="MP4">MP4</SelectItem>
                    <SelectItem value="MOV" disabled>
                      MOV (ProRes) · 后续版本
                    </SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="field">
                <span className="field-label">视频编码</span>
                <Select value={codec} onValueChange={setCodec}>
                  <SelectTrigger className="h-8">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="H.264">H.264</SelectItem>
                    <SelectItem value="H.265">H.265</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="field">
                <span className="field-label">分辨率策略</span>
                <Select value={resolution} onValueChange={setResolution}>
                  <SelectTrigger className="h-8">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="保持原始分辨率">保持原始分辨率</SelectItem>
                    <SelectItem value="1920x1080">1920×1080</SelectItem>
                    <SelectItem value="1280x720">1280×720</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="field">
                <span className="field-label">命名规则</span>
                <Select value={naming} onValueChange={setNaming}>
                  <SelectTrigger className="h-8">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="原文件名_cleaned">原文件名_cleaned</SelectItem>
                    <SelectItem value="前缀 + 时间戳">前缀 + 时间戳</SelectItem>
                    <SelectItem value="自定义序列" disabled>
                      自定义序列 · 后续版本
                    </SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="field">
                <span className="field-label">输出目录</span>
                <Input
                  className="h-8"
                  value={outputDir}
                  onChange={(e) => setOutputDir(e.target.value)}
                  placeholder="留空则输出到源文件同目录"
                />
              </div>

              <div className="section-title">编码参数（PRD 3.2.2）</div>
              <div className="field-row">
                <div className="field">
                  <span className="field-label">码率（kbps）</span>
                  <Input
                    className="h-8"
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
                    className="h-8"
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
                    className="h-8"
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
              <div className="switch">
                <div>
                  <div className="switch-label">生成汇总报告与审计日志</div>
                  <div className="switch-desc">即将支持（后续版本）</div>
                </div>
                <Switch checked={exportReport} onCheckedChange={setExportReport} disabled />
              </div>
            </div>
          </ScrollArea>
        </TabsContent>
      </Tabs>

      <div className="pane-cta">
        <Button onClick={onEnqueue} disabled={enqueued}>
          <ListPlus />
          {enqueued ? "已加入队列" : "加入处理队列"}
        </Button>
      </div>

      <Dialog open={previewOpen} onOpenChange={setPreviewOpen}>
        <DialogContent className="max-w-[min(90vw,960px)]">
          <DialogHeader>
            <DialogTitle>清洗前后并排预览</DialogTitle>
            <DialogDescription>
              左列为原片、右列为处理产物，确认观感无异常后再使用该产物。
            </DialogDescription>
          </DialogHeader>
          {previewUrl && (
            <img
              src={previewUrl}
              alt="清洗前后对比"
              className="max-h-[70vh] w-full rounded-md object-contain"
            />
          )}
        </DialogContent>
      </Dialog>
    </aside>
  );
}

/* ============ 批量操作条 ============ */
function BatchBar({ count }: { count: number }) {
  const selected = useMaterialsStore((state) => state.selected);
  const materials = useMaterialsStore((state) => state.materials);
  const deleteSelected = useMaterialsStore((state) => state.deleteSelected);
  const addHistory = useHistoryStore((state) => state.add);
  const jobs = useQueueStore((state) => state.jobs);
  const [batchLevel, setBatchLevel] = useState<CleanLevel>("平衡");
  const [batchAnti, setBatchAnti] = useState("关闭");
  const [batchTemplate, setBatchTemplate] = useState("manual");
  const [templates, setTemplates] = useState<TemplateInfo[]>([]);
  const [scanBusy, setScanBusy] = useState(false);

  const ids = Object.keys(selected);

  useEffect(() => {
    void listTemplates()
      .then(setTemplates)
      .catch(() => undefined);
  }, []);

  const templateParams = (template: TemplateInfo) => {
    const payload = (template.payload ?? {}) as Record<string, unknown>;
    return {
      level: (payload.level as CleanLevel) ?? "平衡",
      restruct: (payload.restruct as number) ?? 30,
      perturb: (payload.perturb as number) ?? 20,
      denoise: (payload.denoise as boolean) ?? true,
      audio: (payload.audio as boolean) ?? true,
      codec: (payload.codec as string) ?? "H.264",
    };
  };

  const runScan = async () => {
    if (scanBusy) return;
    const pendingPaths = pendingDetectPaths(jobs);
    const targets = ids
      .map((id) => materials.find((m) => m.id === id))
      .filter(
        (m): m is Material & { path: string } => !!m?.path && !pendingPaths.has(m.path),
      )
      .map((m) => ({ kind: "detect" as const, path: m.path }));
    if (!targets.length) {
      toast(
        ids.some((id) => materials.find((m) => m.id === id)?.path)
          ? "所选素材均已在检测中，请等待完成"
          : "所选素材缺少本地路径，无法执行",
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

  const enqueueBatch = async () => {
    const template = templates.find((item) => item.name === batchTemplate);
    const params = template
      ? templateParams(template)
      : {
          level: batchLevel,
          restruct: LEVEL_PRESETS[batchLevel].restruct,
          perturb: LEVEL_PRESETS[batchLevel].perturb,
          denoise: batchLevel !== "轻度",
          audio: true,
          codec: "H.264",
        };
    const preset = LEVEL_PRESETS[params.level as CleanLevel];
    const restruct = template ? params.restruct : preset.restruct;
    const perturbPct = template ? params.perturb : preset.perturb;
    const speed = Math.max(0.85, 1 - 0.15 * (restruct / 100));
    const recrop = 0.015 + 0.075 * (perturbPct / 100);
    const perturb = perturbPct / 100;
    const denoise = template ? params.denoise : batchLevel !== "轻度";
    const anti = ANTI_PRESETS[batchAnti];
    const targets = ids
      .map((id) => materials.find((m) => m.id === id))
      .filter((m): m is Material & { path: string } => !!m?.path)
      .map((m) => ({
        kind: "desensitize" as const,
        path: m.path,
        options: {
          output: `${m.path.replace(/\.(mp4|mov|mkv|avi|flv|ts)$/i, "")}_cleaned.mp4`,
          reorder: true,
          speed,
          recrop,
          regrade: true,
          perturb,
          audio_remix: params.audio,
          sharpness: true,
          color_restore: true,
          denoise,
          codec: params.codec === "H.265" ? "libx265" : "libx264",
          seed: Math.floor(Math.random() * 1_000_000),
          ...(anti ? snakeAnti(anti) : {}),
        },
      }));
    if (!targets.length) {
      toast("所选素材没有本地路径，需通过本地路径导入");
      return;
    }
    try {
      await enqueueJob("批量清洗", targets);
      targets.forEach((task) => {
        const name = task.path.split(/[\\/]/).pop() ?? task.path;
        addHistory({
          name,
          time: nowStr(),
          action: "加入处理队列",
          params: `${params.level}档 · 对抗${batchAnti} · 批量`,
          out: task.options.output as string,
          result: "排队中",
        });
      });
      toast(`已将 ${targets.length} 个素材加入队列`);
    } catch {
      toast("入队失败，请确认引擎在线");
    }
  };

  const exportSelected = async () => {
    const paths = ids
      .map((id) => materials.find((m) => m.id === id))
      .filter((m) => !!m?.path)
      .map((m) => m!.path as string);
    if (!paths.length) {
      toast("所选素材缺少本地路径，无法导出");
      return;
    }
    try {
      const { exported, missing } = await exportOutputs(paths);
      if (exported.length) {
        addHistory({
          name: `${exported.length} 个素材`,
          time: nowStr(),
          action: "批量导出",
          params: "清洗/修复产物",
          out: `导出目录 · ${exported.length} 个`,
          result: "成功",
        });
      }
      const parts: string[] = [];
      if (exported.length) parts.push(`已导出 ${exported.length} 个产物到导出目录`);
      if (missing.length) parts.push(`${missing.length} 个素材尚未处理，无产物可导出`);
      const firstDest = exported[0]?.dest;
      toast(
        parts.length ? parts.join(" · ") : "所选素材均未处理，请先清洗或修复",
        undefined,
        firstDest
          ? { label: "打开文件夹", onClick: () => openInFolder(firstDest) }
          : undefined,
      );
    } catch (error) {
      toast(error instanceof Error ? error.message : "导出失败");
    }
  };

  return (
    <div className="batch-bar">
      <span className="batch-count">已选 {count} 个</span>
      <Select value={batchTemplate} onValueChange={setBatchTemplate}>
        <SelectTrigger className="h-7 w-28" aria-label="批量套用模板">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="manual">手动档位</SelectItem>
          {templates.map((template) => (
            <SelectItem key={template.id} value={template.name}>
              {template.name}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      <Select
        value={batchLevel}
        disabled={batchTemplate !== "manual"}
        onValueChange={(value) => setBatchLevel(value as CleanLevel)}
      >
        <SelectTrigger className="h-7 w-20" aria-label="批量清洗档位">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {CLEAN_LEVELS.map((level) => (
            <SelectItem key={level} value={level}>
              {level}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      <Select value={batchAnti} onValueChange={setBatchAnti}>
        <SelectTrigger className="h-7 w-20" aria-label="批量指纹对抗档">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {Object.keys(ANTI_PRESETS).map((level) => (
            <SelectItem key={level} value={level}>
              {level}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      <Button variant="secondary" size="sm" onClick={enqueueBatch}>
        加入队列
      </Button>
      <Button variant="secondary" size="sm" disabled={scanBusy} onClick={runScan}>
        {scanBusy ? "提交中…" : "批量检测"}
      </Button>
      <Button variant="secondary" size="sm" onClick={exportSelected}>
        导出
      </Button>
      <Button
        variant="ghost"
        size="sm"
        className="text-destructive"
        onClick={() => {
          deleteSelected(ids);
          toast(`已删除 ${ids.length} 个素材`);
        }}
      >
        删除
      </Button>
    </div>
  );
}
