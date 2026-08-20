import { create } from "zustand";
import { appendAudit, type AuditEntry } from "@/lib/backend";

export type HistoryEntry = AuditEntry;

interface HistoryState {
  entries: HistoryEntry[];
  add: (entry: HistoryEntry) => void;
  markResult: (id: number, result: string) => void;
  load: (entries: HistoryEntry[]) => void;
}

export const useHistoryStore = create<HistoryState>((set) => ({
  entries: [],
  add: (entry) => {
    set((state) => ({ entries: [entry, ...state.entries].slice(0, 200) }));
    // 持久化到后端审计日志（失败不阻塞交互）。
    void appendAudit(entry)
      .then((saved) => {
        set((state) => ({
          entries: state.entries.map((item) => (item === entry ? { ...item, id: saved.id } : item)),
        }));
      })
      .catch(() => undefined);
  },
  markResult: (id, result) =>
    set((state) => ({
      entries: state.entries.map((item) => (item.id === id ? { ...item, result } : item)),
    })),
  load: (entries) => set({ entries }),
}));
