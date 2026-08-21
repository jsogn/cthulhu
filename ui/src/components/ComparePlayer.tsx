import { useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { mediaUrl } from "@/lib/backend";
import { Pause, Play } from "lucide-react";

interface ComparePlayerProps {
  left: string;
  right: string;
  leftLabel: string;
  rightLabel: string;
}

function fmtTime(seconds: number): string {
  const total = Math.max(0, Math.floor(seconds));
  const mm = String(Math.floor(total / 60)).padStart(2, "0");
  const ss = String(total % 60).padStart(2, "0");
  return `${mm}:${ss}`;
}

/** 两侧同步播放对比：左侧为主时间轴，右侧跟随，共用一个播放控制条。 */
export default function ComparePlayer({ left, right, leftLabel, rightLabel }: ComparePlayerProps) {
  const leftRef = useRef<HTMLVideoElement>(null);
  const rightRef = useRef<HTMLVideoElement>(null);
  const [playing, setPlaying] = useState(false);
  const [duration, setDuration] = useState(0);
  const [progress, setProgress] = useState(0);

  const syncRight = () => {
    const master = leftRef.current;
    const slave = rightRef.current;
    if (!master || !slave) return;
    if (Math.abs(slave.currentTime - master.currentTime) > 0.08) {
      slave.currentTime = Math.min(master.currentTime, slave.duration || master.currentTime);
    }
  };

  const toggle = async () => {
    const master = leftRef.current;
    const slave = rightRef.current;
    if (!master || !slave) return;
    if (playing) {
      master.pause();
      slave.pause();
      setPlaying(false);
      return;
    }
    if (master.ended || (slave.ended && master.currentTime >= (slave.duration || 0))) {
      master.currentTime = 0;
      slave.currentTime = 0;
    }
    await master.play();
    await slave.play().catch(() => undefined);
    setPlaying(true);
  };

  const seek = (value: number) => {
    const master = leftRef.current;
    const slave = rightRef.current;
    if (!master) return;
    master.currentTime = value;
    if (slave) slave.currentTime = Math.min(value, slave.duration || value);
    setProgress(value);
  };

  return (
    <div className="compare-player">
      <div className="compare-grid">
        <div className="compare-side">
          <div className="compare-label">{leftLabel}</div>
          <video
            ref={leftRef}
            src={mediaUrl(left)}
            muted
            playsInline
            className="compare-video"
            onLoadedMetadata={(event) => setDuration(event.currentTarget.duration || 0)}
            onTimeUpdate={(event) => {
              setProgress(event.currentTarget.currentTime);
              if (playing) syncRight();
            }}
            onEnded={() => setPlaying(false)}
          />
        </div>
        <div className="compare-side">
          <div className="compare-label">{rightLabel}</div>
          <video
            ref={rightRef}
            src={mediaUrl(right)}
            muted
            playsInline
            className="compare-video"
          />
        </div>
      </div>
      <div className="compare-controls">
        <Button variant="outline" size="icon" onClick={() => void toggle()} aria-label="播放或暂停">
          {playing ? <Pause /> : <Play />}
        </Button>
        <input
          className="flex-1"
          type="range"
          min={0}
          max={duration || 1}
          step={0.01}
          value={Math.min(progress, duration || 1)}
          aria-label="播放进度"
          onChange={(event) => seek(Number(event.target.value))}
        />
        <span className="mono text-xs text-muted-foreground">
          {fmtTime(progress)} / {fmtTime(duration)}
        </span>
      </div>
    </div>
  );
}
