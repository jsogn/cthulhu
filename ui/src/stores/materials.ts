import { create } from "zustand";
import {
  fetchLibrary,
  fetchOutputCounts,
  thumbUrl,
  unregisterLibrary,
  type LibraryFile,
} from "@/lib/backend";
import { fmtSize } from "@/lib/format";

export type RiskLevel = "有疑似特征" | "未发现异常" | "待检测";

export interface Material {
  id: string;
  name: string;
  dur: string;
  res: string;
  fps: string;
  size: string;
  risk: RiskLevel;
  score: number;
  tags: string[];
  frame: string;
  path?: string;
  duration?: number;
  missing?: boolean;
  report?: unknown;
  outputCount?: number;
}

interface MaterialsState {
  materials: Material[];
  activeId: string | null;
  selected: Record<string, boolean>;
  search: string;
  loadFailed: boolean;
  select: (id: string) => void;
  toggleSelected: (id: string) => void;
  setSearch: (search: string) => void;
  toggleSelectAll: (ids: string[]) => void;
  deleteSelected: (ids: string[]) => void;
  importOne: (material: Material) => void;
  applyDetectResult: (
    path: string,
    report: { bitstream: { score: number; flags: string[] } },
  ) => void;
  loadLibrary: (files: LibraryFile[], outputCounts?: Record<string, number>) => void;
  refreshLibrary: () => Promise<void>;
  refreshOutputCounts: () => Promise<void>;
  markLoadFailed: () => void;
}

function reportRisk(report?: { bitstream: { flags: string[] } } | null): RiskLevel {
  if (!report) return "待检测";
  return report.bitstream.flags.length > 0 ? "有疑似特征" : "未发现异常";
}

function fmtDur(seconds: number): string {
  const total = Math.round(seconds || 0);
  const mm = Math.floor(total / 60);
  const ss = total % 60;
  return `${String(mm).padStart(2, "0")}:${String(ss).padStart(2, "0")}`;
}

export const useMaterialsStore = create<MaterialsState>((set, get) => ({
  materials: [],
  activeId: null,
  selected: {},
  search: "",
  loadFailed: false,

  select: (id) => set({ activeId: id }),

  toggleSelected: (id) =>
    set((state) => {
      const selected = { ...state.selected };
      if (selected[id]) delete selected[id];
      else selected[id] = true;
      return { selected };
    }),

  setSearch: (search) => set({ search }),

  toggleSelectAll: (ids) =>
    set((state) => {
      const allSelected = ids.length > 0 && ids.every((id) => state.selected[id]);
      const selected = { ...state.selected };
      ids.forEach((id) => {
        if (allSelected) delete selected[id];
        else selected[id] = true;
      });
      return { selected };
    }),

  deleteSelected: (ids) => {
    const removed = get().materials.filter((m) => ids.includes(m.id));
    set((state) => {
      const remaining = state.materials.filter((m) => !ids.includes(m.id));
      const selected = { ...state.selected };
      ids.forEach((id) => delete selected[id]);
      return {
        materials: remaining,
        selected,
        activeId:
          state.activeId && ids.includes(state.activeId)
            ? (remaining[0]?.id ?? null)
            : state.activeId,
      };
    });
    // 同步移除持久化记录；源视频文件一律保留，仅解除登记。
    removed.forEach((material) => {
      if (material.path) {
        unregisterLibrary(material.path).catch(() => {
          // 素材已从界面移除，后端清理失败不影响本次删除。
        });
      }
    });
  },

  // 最新导入排在最前，与后端「新增时间倒序」保持一致。
  importOne: (material) =>
    set((state) => {
      const duplicate = state.materials.find(
        (item) => material.path && item.path === material.path,
      );
      if (duplicate) {
        // 同路径已存在：保留原 id（选中态不失效），更新元数据并置顶。
        return {
          materials: [
            { ...material, id: duplicate.id },
            ...state.materials.filter((item) => item.path !== material.path),
          ],
        };
      }
      return { materials: [material, ...state.materials] };
    }),

  applyDetectResult: (path, report) =>
    set((state) => ({
      materials: state.materials.map((m) =>
        m.path === path
          ? {
              ...m,
              report,
              score: report.bitstream.score,
              risk: reportRisk(report),
            }
          : m,
      ),
    })),

  loadLibrary: (files, outputCounts?) => {
    const materials: Material[] = files.map((file) => {
      const report = file.report ?? undefined;
      const score = report?.bitstream.score ?? 0;
      const missing = file.video == null;
      return {
        id: file.path,
        name: file.name,
        dur: missing ? "—" : fmtDur(file.video!.duration),
        res: missing ? "—" : `${file.video!.width}×${file.video!.height}`,
        fps: missing ? "—" : `${file.video!.fps}fps`,
        size: fmtSize(file.size),
        risk: missing ? "待检测" : reportRisk(report),
        score,
        tags: ["本地"],
        frame: missing ? "" : thumbUrl(file.path, 320),
        path: file.path,
        duration: file.video?.duration,
        missing,
        report,
        outputCount: outputCounts?.[file.path] ?? 0,
      };
    });
    set({ materials, activeId: materials[0]?.id ?? null });
  },

  refreshLibrary: async () => {
    try {
      const files = await fetchLibrary();
      const counts = await fetchOutputCounts().catch(() => ({} as Record<string, number>));
      get().loadLibrary(files, counts);
      set({ loadFailed: false });
    } catch {
      set({ loadFailed: true });
    }
  },

  refreshOutputCounts: async () => {
    try {
      const counts = await fetchOutputCounts();
      set((state) => ({
        materials: state.materials.map((material) => ({
          ...material,
          outputCount: counts[material.path ?? ""] ?? 0,
        })),
      }));
    } catch {
      // 数量刷新失败保持现状
    }
  },

  markLoadFailed: () => set({ loadFailed: true }),
}));
