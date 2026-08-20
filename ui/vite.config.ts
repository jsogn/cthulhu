import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { fileURLToPath, URL } from "node:url";

// 开发模式下将 API 与事件通道代理到 Python 后端。
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./src", import.meta.url)),
    },
  },
  server: {
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:57173",
      "/ws": {
        target: "ws://127.0.0.1:57173",
        ws: true,
      },
    },
  },
});
