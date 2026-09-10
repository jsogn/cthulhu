import { useEffect, useRef, useState } from "react";
import type { PointerEvent as ReactPointerEvent } from "react";
import { Code, Pause, Play, SkipBack, SkipForward } from "lucide-react";
import { Button } from "@/components/ui/button";
import { frameUrl, mediaUrl } from "@/lib/backend";
import { fmtFrames } from "@/lib/format";
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
  const stageRef = useRef<HTMLDivElement>(null);
  const mediaBoxRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const compareLeftRef = useRef<HTMLVideoElement>(null);
  const compareMode = !!compareLeft && !!compareRight;
  const lastEmitRef = useRef(0);
  const [mediaDuration, setMediaDuration] = useState<number | null>(null);
  const [mediaSize, setMediaSize] = useState<{ w: number; h: number } | null>(null);
  const [fitBox, setFitBox] = useState<{ w: number; h: number } | null>(null);

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

  // 素材切换：重置真实时长/尺寸，避免串台。
  useEffect(() => {
    setMediaDuration(null);
    setMediaSize(null);
    setFitBox(null);
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
    const el = e.currentTarget;
    el.setPointerCapture(e.pointerId);
    setComparePct(Math.max(0, Math.min(100, percentFrom(e, el).x)));
  };

  const onComparePointerMove = (e: ReactPointerEvent<HTMLDivElement>) => {
    if (e.buttons & 1) {
      setComparePct(Math.max(0, Math.min(100, percentFrom(e, e.currentTarget).x)));
    }
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
