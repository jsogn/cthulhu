import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import "./index.css";

// Electron 桌面端用平台标识给顶栏留出 macOS 红绿灯位置。
document.documentElement.dataset.platform = window.appEnv?.platform ?? "web";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
