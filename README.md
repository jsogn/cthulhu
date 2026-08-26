# Cthulhu · 推广素材处理台

本地端推广素材处理平台，当前核心能力为素材去重与暗水印清洗，并预留
多版本候选、裂变、分组等扩展空间。私有分发、不对外公开；macOS 优先，
Windows 次之。产品定位与扩展设计见
[docs/产品定位与架构.md](docs/产品定位与架构.md)。

## 技术栈

- 前端：React 19 + TypeScript + Vite + Tailwind CSS 4 + Zustand
- 桌面壳：Electron
- 后端：Python 3.12 + FastAPI（sidecar 进程，REST + WebSocket）
- 媒体/算法：FFmpeg、OpenCV、NumPy/SciPy、ONNX Runtime（后续里程碑接入）
- 规划文档见 [docs/实施文档.md](docs/实施文档.md)

## 环境要求

- Node.js 22+、pnpm 11+
- uv（Python 3.12，由 uv 自动管理）
- FFmpeg（系统安装，M3 起使用）

> 网络较慢时，Electron 二进制可走镜像安装：
> `ELECTRON_MIRROR=https://npmmirror.com/mirrors/electron/ pnpm --dir electron rebuild electron`

## 开发运行

```bash
pnpm install                # 根依赖（concurrently）
pnpm --dir ui install       # 前端依赖
pnpm --dir electron install # Electron 运行时
cd backend && uv sync       # 后端依赖

pnpm dev                    # 开发模式：仅启动 Vite + Python 后端，浏览器打开 http://localhost:5173
pnpm dev:desktop            # 桌面模式：额外拉起 Electron（其主进程会自动拉起后端）
```

开发阶段默认用 `pnpm dev` + 浏览器即可，轻量快速；需要验证桌面壳时才用 `pnpm dev:desktop`。

## 目录结构

```text
docs/        规划文档
ui/          React 前端（由早期高保真原型迁移实现）
electron/    Electron 主进程与 preload
backend/     Python 后端（FastAPI）
data/        样本与实验数据（不入库）
packaging/   打包配置（后续补充）
```

## 研究 CLI（暗水印基准库与攻击评估）

M2 起，研究闭环优先走命令行，评估策略为「自建参考水印基准库 + 平台黑盒终验」：

```bash
cd backend && uv sync          # 同步研究依赖（numpy/scipy/typer/pytest/ruff）

uv run cthulhu gen             # 生成合成干净样本（视频帧 npy + 音频 wav）
uv run cthulhu harness-video   # 水印(Lsb/SS/QIM) × 攻击矩阵 → BER/PSNR/SSIM
uv run cthulhu harness-audio   # 回声隐藏水印 × 音频攻击矩阵
uv run cthulhu probe 素材.mp4                # 容器/元数据/SEI 探测
uv run cthulhu sample-diff clean.mp4 wm.mp4  # 对齐 + 差分报告 + DCT 热图
uv run cthulhu bitstream 素材.mp4 [--reference clean.mp4]  # 码流层 QP/码量/GOP 分析
uv run cthulhu desensitize in.mp4 out.mp4 [--speed 0.95] [--recrop 0.03] [--banner 文字]
uv run cthulhu similarity a.mp4 b.mp4         # 内容/运动/哈希/画质相似度

uv run pytest                  # 基准库往返与攻击鲁棒性测试
uv run ruff check src tests    # 静态检查
```

GUI 后端 API（`pnpm dev` 时运行在 127.0.0.1:57173，带任务进度 WebSocket）：

```text
POST /api/detect        {"path": "..."}   → 容器/SEI/压缩域联合检测报告
POST /api/similarity    {"a": "...", "b": "..."} → 内容/运动/哈希相似度
POST /api/desensitize   {"path": "...", "output": "...", ...} → 脱敏 + 前后相似度 + VMAF
```

## 打包（macOS 私有分发）

```bash
pnpm package                 # 前端构建 → PyInstaller 后端 → electron-builder dmg
```

产物：`packaging/dist/Cthulhu-0.1.0-arm64.dmg`。当前为未签名私有构建；发布级签名/公证与 Windows 打包留待功能收尾后执行。打包版依赖系统安装的 ffmpeg/ffprobe（检测/清洗/修复/VMAF 均需要）。

发布级清单见 [docs/发布打包.md](docs/发布打包.md)：图标已就绪，ffmpeg 内置机制已就绪（`packaging/fetch_ffmpeg.sh` 获取静态构建），macOS 签名/公证与 Windows 构建为待办。

报告输出到 `backend/data/`（已忽略入库）。当前实现：LSB / 空域扩频 / DCT-QIM / 回声隐藏四种嵌入器，去噪 / 重量化 / 几何 / 时序 / 协同平均 / 音频变速等攻击，BER / PSNR / SSIM / VMAF（对齐后）指标，真实视频解码/编码、相位相关对齐与差分样本制备，压缩域（码流层）QP 图/码量分配/GOP/SEI 分析（支持与干净基准的差分判定），以及内容脱敏（分镜重排/变速/重新构图/重调光/贴纸 + 自建内容/运动 embedding 与 dHash 相似度量化）。
