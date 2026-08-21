import { create } from "zustand";

export type ViewKey = "workbench" | "outputs" | "jobs" | "templates" | "history" | "settings";
export type ThemePref = "dark" | "light" | "system";

interface AppState {
  view: ViewKey;
  themePref: ThemePref;
  backendConnected: boolean;
  importOpen: boolean;
  setView: (view: ViewKey) => void;
  setThemePref: (pref: ThemePref) => void;
  setBackendConnected: (connected: boolean) => void;
  setImportOpen: (open: boolean) => void;
}

function initialThemePref(): ThemePref {
  const saved = localStorage.getItem("theme");
  if (saved === "dark" || saved === "light" || saved === "system") return saved;
  return "system";
}

export const useAppStore = create<AppState>((set) => ({
  view: "workbench",
  themePref: initialThemePref(),
  backendConnected: false,
  importOpen: false,
  setView: (view) => set({ view }),
  setThemePref: (themePref) => set({ themePref }),
  setBackendConnected: (backendConnected) => set({ backendConnected }),
  setImportOpen: (importOpen) => set({ importOpen }),
}));

/** 把偏好解析为当前实际主题。 */
export function resolveTheme(pref: ThemePref): "dark" | "light" {
  if (pref !== "system") return pref;
  return window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
}
