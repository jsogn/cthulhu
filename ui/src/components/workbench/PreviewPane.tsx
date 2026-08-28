import { useEffect, useRef, useState } from "react";
import type { KeyboardEvent as ReactKeyboardEvent, PointerEvent as ReactPointerEvent } from "react";
import { Code, Pause, Play, SkipBack, SkipForward } from "lucide-react";
import { Button } from "@/components/ui/button";
import { frameUrl, mediaUrl } from "@/lib/backend";
import { fmtFrames } from "@/lib/format";
import { useRegionsStore } from "@/stores/regions";
import type { Material } from "@/stores/materials";

const FRAME_MAX = 540;

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

export function PreviewPane({
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
  const hasRealFrame = !!material?.path && !material.missing;
  const durationSec = mediaDuration ?? material?.duration ?? null;
  const maxFrame = durationSec
    ? Math.max(1, Math.round(durationSec * fps) - 1)
    : FRAME_MAX;
  const frameSrc = hasRealFrame
    ? frameUrl(material.path as string, frame / fps)
    : (material?.frame ?? "/assets/frame-drama.svg");

  // 切换素材时退出对比，回到新素材的普通预览。
  useEffect(() => {
    if (compareMode) onExitCompare();
    // 只监听素材变化；onExitCompare 每次渲染都是新引用，不能作为依赖。
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [material?.id]);

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

  // 播放/暂停状态与真实 <video> 同步。
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
                    <Code />
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
                playsInline
                crossOrigin="anonymous"
                onTimeUpdate={onTimeUpdate}
                onLoadedMetadata={onLoadedMetadata}
                onEnded={() => setPlaying(false)}
              />
            )
          ) : material?.missing ? (
            <div className="mat-empty">
              <p>文件不存在或已被移动</p>
              <p className="mat-empty-hint">素材记录仍保留；文件放回原路径后会自动恢复，或在素材库中移除记录</p>
            </div>
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
