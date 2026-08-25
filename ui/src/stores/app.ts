import { create } from "zustand";
import type { TemplatePayload } from "@/lib/templates";

export interface PendingTemplate {
  id: string;
  name: string;
  payload: TemplatePayload;
}

export type ViewKey = "workbench" | "jobs" | "templates" | "products" | "settings";
export type ThemePref = "dark" | "light" | "system";

interface AppState {
  view: ViewKey;
  themePref: ThemePref;
  backendConnected: boolean;
  importOpen: boolean;
  pendingTemplate: PendingTemplate | null;
  setView: (view: ViewKey) => void;
  setThemePref: (pref: ThemePref) => void;
  setBackendConnected: (connected: boolean) => void;
  setImportOpen: (open: boolean) => void;
  queueTemplate: (template: PendingTemplate) => void;
  consumePendingTemplate: () => PendingTemplate | null;
}

function initialThemePref(): ThemePref {
  const saved = localStorage.getItem("theme");
  if (saved === "dark" || saved === "light" || saved === "system") return saved;
  return "system";
}

export const useAppStore = create<AppState>((set, get) => ({
  view: "workbench",
  themePref: initialThemePref(),
  backendConnected: false,
  importOpen: false,
  pendingTemplate: null,
  setView: (view) => set({ view }),
  setThemePref: (themePref) => set({ themePref }),
  setBackendConnected: (backendConnected) => set({ backendConnected }),
  setImportOpen: (importOpen) => set({ importOpen }),
  queueTemplate: (template) => set({ pendingTemplate: template }),
  consumePendingTemplate: () => {
    const pending = get().pendingTemplate;
    if (pending) set({ pendingTemplate: null });
    return pending;
  },
}));

/** 把偏好解析为当前实际主题。 */
export function resolveTheme(pref: ThemePref): "dark" | "light" {
  if (pref !== "system") return pref;
  return window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
}
