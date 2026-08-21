import { create } from "zustand";
import { fetchLibrary, thumbUrl, unregisterLibrary, type LibraryFile } from "@/lib/backend";
import { fmtSize } from "@/lib/format";

export type RiskLevel = "有疑似特征" | "未检出异常" | "待检测";

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
  report?: unknown;
}

interface MaterialsState {
  materials: Material[];
  activeId: string | null;
  selected: Record<string, boolean>;
  riskFilter: "全部" | RiskLevel;
  search: string;
  loadFailed: boolean;
  select: (id: string) => void;
  toggleSelected: (id: string) => void;
  setRiskFilter: (filter: "全部" | RiskLevel) => void;
  setSearch: (search: string) => void;
  toggleSelectAll: (ids: string[]) => void;
  deleteSelected: (ids: string[]) => void;
  importOne: (material: Material) => void;
  applyDetectResult: (
    path: string,
    report: { bitstream: { score: number; flags: string[] } },
  ) => void;
  loadLibrary: (files: LibraryFile[]) => void;
  refreshLibrary: () => Promise<void>;
  markLoadFailed: () => void;
}

function reportRisk(report?: { bitstream: { flags: string[] } } | null): RiskLevel {
  if (!report) return "待检测";
  return report.bitstream.flags.length > 0 ? "有疑似特征" : "未检出异常";
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
  riskFilter: "全部",
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

  setRiskFilter: (riskFilter) => set({ riskFilter }),
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
    // 同步移除持久化记录：网页上传副本会删除文件，桌面源文件仅解除登记。
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
    set((state) => ({ materials: [material, ...state.materials] })),

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

  loadLibrary: (files) => {
    const materials: Material[] = files
      .filter((file) => file.video)
      .map((file) => {
        const report = file.report ?? undefined;
        const score = report?.bitstream.score ?? 0;
        return {
          id: file.path,
          name: file.name,
          dur: fmtDur(file.video!.duration),
          res: `${file.video!.width}×${file.video!.height}`,
          fps: `${file.video!.fps}fps`,
          size: fmtSize(file.size),
          risk: reportRisk(report),
          score,
          tags: ["本地"],
          frame: thumbUrl(file.path, 320),
          path: file.path,
          duration: file.video!.duration,
          report,
        };
      });
    set({ materials, activeId: materials[0]?.id ?? null });
  },

  refreshLibrary: async () => {
    try {
      const files = await fetchLibrary();
      get().loadLibrary(files);
      set({ loadFailed: false });
    } catch {
      set({ loadFailed: true });
    }
  },

  markLoadFailed: () => set({ loadFailed: true }),
}));
