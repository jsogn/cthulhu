import { create } from "zustand";

export interface Region {
  id: string;
  name: string;
  left: number;
  top: number;
  w: number;
  h: number;
  start: number | null;
  end: number | null;
}

interface Snapshot {
  regions: Region[];
  activeId: string | null;
}

interface RegionsState {
  regions: Region[];
  activeId: string | null;
  undoStack: Snapshot[];
  redoStack: Snapshot[];
  selectRegion: (id: string) => void;
  addRegion: () => void;
  deleteRegion: (id: string) => void;
  patchRegion: (id: string, patch: Partial<Region>) => void;
  beginMove: () => void;
  undo: () => void;
  redo: () => void;
}

function snapshot(regions: Region[], activeId: string | null): Snapshot {
  return { regions: regions.map((r) => ({ ...r })), activeId };
}

export const useRegionsStore = create<RegionsState>((set) => {
  const pushUndo = (regions: Region[], activeId: string | null, stack: Snapshot[]) =>
    [...stack, snapshot(regions, activeId)].slice(-40);

  return {
    // 水印区域只应由用户手动添加或未来接入自动检测后生成，
    // 不预置任何默认框，避免误导用户误以为已检测到水印。
    regions: [],
    activeId: null,
    undoStack: [],
    redoStack: [],

    selectRegion: (activeId) => set({ activeId }),

    addRegion: () =>
      set((state) => {
        const region: Region = {
          id: `r${Date.now()}`,
          name: `区域${state.regions.length + 1} · 水印`,
          left: 30,
          top: 30,
          w: 17,
          h: 12,
          start: null,
          end: null,
        };
        return {
          regions: [...state.regions, region],
          activeId: region.id,
          undoStack: pushUndo(state.regions, state.activeId, state.undoStack),
          redoStack: [],
        };
      }),

    deleteRegion: (id) =>
      set((state) => {
        const regions = state.regions.filter((r) => r.id !== id);
        return {
          regions,
          activeId: state.activeId === id ? (regions[0]?.id ?? null) : state.activeId,
          undoStack: pushUndo(state.regions, state.activeId, state.undoStack),
          redoStack: [],
        };
      }),

    patchRegion: (id, patch) =>
      set((state) => ({
        regions: state.regions.map((r) => (r.id === id ? { ...r, ...patch } : r)),
      })),

    beginMove: () =>
      set((state) => ({
        undoStack: pushUndo(state.regions, state.activeId, state.undoStack),
        redoStack: [],
      })),

    undo: () =>
      set((state) => {
        const target = state.undoStack.at(-1);
        if (!target) return state;
        return {
          regions: target.regions,
          activeId: target.activeId,
          undoStack: state.undoStack.slice(0, -1),
          redoStack: [...state.redoStack, snapshot(state.regions, state.activeId)],
        };
      }),

    redo: () =>
      set((state) => {
        const target = state.redoStack.at(-1);
        if (!target) return state;
        return {
          regions: target.regions,
          activeId: target.activeId,
          redoStack: state.redoStack.slice(0, -1),
          undoStack: [...state.undoStack, snapshot(state.regions, state.activeId)],
        };
      }),
  };
});
