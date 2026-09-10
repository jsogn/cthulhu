# Cthulhu · 数字水印与内容指纹研究实验台

本项目用于数字水印、内容指纹、媒体取证与视频变换鲁棒性的研究、学习和授权
实验。所有能力均以可复现实验、代理指标和边界分析为目标，不作为商业化规避
工具，也不承诺任何平台审核、流量或账号安全效果。

> **研究用途声明**
>
> 本项目及仓库内全部文档仅供研究、教学和授权测试使用。严禁用于去除版权或
> 溯源水印、规避平台审核与风控、侵权搬运、账号矩阵规避、欺诈或其他违法违规
> 用途。完整边界见
> [docs/研究用途与合规声明.md](docs/研究用途与合规声明.md)。

项目当前保留素材库、参考检测、变换实验、相似度分析、批量任务与历史记录等
研究能力；所有实验仅可使用合成样本、自有素材或已获明确授权的素材与测试账号。
研究定位与扩展设计见 [docs/产品定位与架构.md](docs/产品定位与架构.md)。

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
cd backend && uv sync --extra purify   # 后端依赖（含潜空间净化运行时）

pnpm dev                    # 开发模式：仅启动 Vite + Python 后端，浏览器打开 http://localhost:5173
pnpm dev:desktop            # 桌面模式：额外拉起 Electron（其主进程会自动拉起后端）
```

开发阶段默认用 `pnpm dev` + 浏览器即可，轻量快速；需要验证桌面壳时才用 `pnpm dev:desktop`。

### 潜空间净化能力（开箱即用）

净化运行时（torch/diffusers/transformers 等）随开发环境与桌面端一起安装，
用户不需要手工装 Python 包。TAESD 权重（`madebyollin/taesd`，约 10MB）随
安装包内置，安装后零下载、首次开启即可推理。引擎是「潜空间瓶颈重建」：
公开深度水印写在低频，重建由解码器先验主导，清除率与原先的 sd-turbo 扩散
方案持平或更好，速度快约 50 倍（research §19）。开发环境可先执行一次
`pnpm package:model` 生成 `packaging/models`（或从已有权重目录复制），
之后 `pnpm dev` 会自动使用内置目录。

```bash
# 查看运行时/模型/设备状态
uv run --project backend --extra purify cthulhu purify-status
# 仅开发/自托管环境需要：没有内置权重时的下载兜底
uv run --project backend --extra purify cthulhu purify-install
```

可配置项：

- `CTHULHU_PURIFY_MODEL_DIR=<dir>`：自定义权重目录（覆盖内置目录）；
- `CTHULHU_PURIFY_ALLOW_DOWNLOAD=0`：禁用兜底下载（离线/内网部署）；
- `CTHULHU_PURIFY_TAESD_MODEL=<repo-id>`：替换默认权重 `madebyollin/taesd`；
- `HF_ENDPOINT=https://hf-mirror.com`：国内镜像。

API：`GET /api/purify/status` 查询状态与进度，`POST /api/purify/install`
触发后台下载（内置权重已就绪时是个空操作）。

### 白盒定向（仅研究侧）

白盒 EOT-PGD / mask PGD 只保留在 `research/`：
`research/scripts/targeted_worker.py`、`research/scripts/eot_core.py` 与
`research/vendor` 的公开权重。正式 App 不再内置白盒 worker、权重或入口；
原因是白盒只支持已公开方案、600 帧短片上限，5 分钟视频即使分段也要数小时。
正式产品统一走黑盒净化 + 嵌入域重写。

## 目录结构

```text
docs/        规划文档
ui/          React 前端（由早期高保真原型迁移实现）
electron/    Electron 主进程与 preload
backend/     Python 后端（FastAPI）
data/        样本与实验数据（不入库）
packaging/   打包配置（后续补充）
```

## 研究 CLI（水印鲁棒性与内容指纹评估）

M2 起，研究闭环优先走命令行，评估策略为「自建参考水印基准库 + 授权环境下的
合规性观察」。以下命令仅用于合成样本、自有素材或已授权素材：

> 说明：CLI 中 `desensitize`、`cleanse` 等历史命名仅表示研究性变换实验，
> 不代表允许用于规避平台规则、去除版权/溯源水印或侵权搬运。

```bash
cd backend && uv sync          # 同步研究依赖（numpy/scipy/typer/pytest/ruff）

uv run cthulhu gen             # 生成合成干净样本（视频帧 npy + 音频 wav）
uv run cthulhu harness-video   # 水印(Lsb/SS/QIM) × 鲁棒性压力测试矩阵 → BER/PSNR/SSIM
uv run cthulhu harness-audio   # 回声隐藏水印 × 音频鲁棒性压力测试矩阵
uv run cthulhu probe 素材.mp4                # 容器/元数据/SEI 探测
uv run cthulhu sample-diff clean.mp4 wm.mp4  # 对齐 + 差分报告 + DCT 热图
uv run cthulhu bitstream 素材.mp4 [--reference clean.mp4]  # 码流层 QP/码量/GOP 分析
uv run cthulhu desensitize in.mp4 out.mp4 [--speed 0.95] [--recrop 0.03] [--banner 文字]
uv run cthulhu similarity a.mp4 b.mp4         # 内容/运动/哈希/画质相似度

uv run pytest                  # 基准库往返与鲁棒性压力测试
uv run ruff check src tests    # 静态检查
```

研究实验 API（`pnpm dev` 时运行在 127.0.0.1:57173，带任务进度 WebSocket）：

```text
POST /api/detect        {"path": "..."}   → 容器/SEI/压缩域联合检测报告
POST /api/similarity    {"a": "...", "b": "..."} → 内容/运动/哈希相似度
POST /api/desensitize   {"path": "...", "output": "...", ...} → 研究性变换 + 前后相似度 + VMAF
```

## 打包（研究实验构建）

```bash
pnpm package                 # 前端构建 → PyInstaller 后端 → electron-builder dmg
```

产物：`packaging/dist/Cthulhu-0.1.0-arm64.dmg`。当前为未签名研究实验构建；
签名/公证与 Windows 打包仅作为工程化验证项，不代表允许对外商业化分发。
打包版依赖系统安装的 ffmpeg/ffprobe（检测/变换实验/修复/VMAF 均需要）。

工程化构建清单见 [docs/发布打包.md](docs/发布打包.md)：图标已就绪，ffmpeg
内置机制已就绪（`packaging/fetch_ffmpeg.sh` 获取静态构建），macOS 签名/公证与
Windows 构建为待办。

报告输出到 `backend/data/`（已忽略入库）。当前实现：LSB / 空域扩频 / DCT-QIM /
回声隐藏等参考嵌入器，去噪 / 重量化 / 几何 / 时序 / 协同平均 / 音频变速等
鲁棒性压力测试，BER / PSNR / SSIM / VMAF（对齐后）指标，真实视频解码/编码、
相位相关对齐与差分样本制备，压缩域（码流层）QP 图/码量分配/GOP/SEI 分析
（支持与干净基准的差分判定），以及内容变换实验（分镜重排/变速/重新构图/重调光/
贴纸 + 自建内容/运动 embedding 与 dHash 相似度量化）。
