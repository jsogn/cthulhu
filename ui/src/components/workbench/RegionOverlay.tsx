import type {
  KeyboardEvent as ReactKeyboardEvent,
  PointerEvent as ReactPointerEvent,
  RefObject,
} from "react";
import { useRegionsStore } from "@/stores/regions";

interface RegionOverlayProps {
  mediaBoxRef: RefObject<HTMLDivElement | null>;
}

/** 水印修复区域：预览遮罩 + 可拖拽/缩放/键盘微调的选框。 */
export function RegionOverlay({ mediaBoxRef }: RegionOverlayProps) {
  const activeRegion = useRegionsStore((state) =>
    state.regions.find((region) => region.id === state.activeId),
  );
  const beginMove = useRegionsStore((state) => state.beginMove);
  const patchRegion = useRegionsStore((state) => state.patchRegion);

  const percentFrom = (e: ReactPointerEvent, el: HTMLElement) => {
    const rect = el.getBoundingClientRect();
    return {
      x: ((e.clientX - rect.left) / rect.width) * 100,
      y: ((e.clientY - rect.top) / rect.height) * 100,
    };
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

  return (
    <>
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
    </>
  );
}
