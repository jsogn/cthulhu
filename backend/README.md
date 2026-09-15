# Cthulhu 后端（Python / FastAPI sidecar）

> 文档导航：[README](../README.md) · [文档索引](../docs/README.md) · [研究用途与合规声明](../docs/研究用途与合规声明.md) · [产品定位与架构](../docs/产品定位与架构.md) · [实施文档](../docs/实施文档.md) · [发布打包](../docs/发布打包.md)

> **研究用途声明**：本目录代码仅用于数字水印、内容指纹、媒体取证与视频变换的
> 研究、教学和授权测试。严禁用于规避平台规则、去除版权/溯源水印、侵权搬运、
> 账号矩阵规避或其他违法违规用途。完整边界见
> [研究用途与合规声明](../docs/研究用途与合规声明.md)。

前端与 Electron 只通过本目录暴露的 REST + WebSocket 契约交互：桌面端启动时拉起
该进程，退出时回收；开发模式下由 `pnpm dev` 与 Vite（5173）并行启动，默认监听
`127.0.0.1:57173`。

## 快速开始

```bash
cd backend
uv sync                 # 研究依赖（numpy/scipy/typer/pytest/ruff）
uv sync --extra purify  # 追加潜空间净化运行时（torch/diffusers）

# 开发服务器（与前端联调）
CTHULHU_AUTH_TOKEN=dev-local uv run --extra purify uvicorn cthulhu_backend.main:app --port 57173 --reload

# 质量门
uv run ruff check src tests && uv run pytest tests -q
uv run python scripts/export_openapi.py   # 契约漂移：openapi.json / ui/src/lib/api-types.ts
```

## 分层与模块地图

| 层 | 模块 | 职责 |
| :--- | :--- | :--- |
| 进程与契约 | `main.py`、`api.py`、`schemas.py`、`events.py`、`progress.py` | FastAPI 应用、令牌鉴权、REST 路由、WebSocket 进度事件、请求/响应模型 |
| 编排与调度 | `jobs.py`、`services.py`、`planning.py`、`parallel.py`、`heartbeat.py` | 任务队列（排队 / 并行 / 取消 / 暂停 / 续跑）、业务编排、并行度与资源预算 |
| 变换管线 | `pipeline.py`、`transform/` | 变换原语、策略与档位、分块处理、YUV 快路径、原生滤镜下沉、编码与封装 |
| 度量与评估 | `similarity/`、`quality.py`、`evaluate/`、`fingerprint/` | 内容 / 运动 / 哈希 / 画质相似度、BER/PSNR/SSIM/VMAF、基准矩阵与回归门 |
| 检测与分析 | `bitstream/`、`watermark/`、`attacks/`、`baseline.py` | 容器与 SEI 扫描、码流层 QP/GOP 分析、参考水印嵌入与检测、鲁棒性压力测试原语 |
| 媒体与素材 | `media/`、`sample_prep/`、`samples.py`、`cache.py` | FFmpeg 调用与内置安装、样本制备与对齐差分、缩略图与分析缓存 |
| 存储 | `db.py` | SQLite：素材库、模板、任务历史、产物参数快照（`variants`） |
| 加速与数值 | `native/`、`native_dct.py`、`numeric.py`、`tiers.py` | 原生加速库加载与降级、数值原语、档位（耗时 / 画质代价）单一真值 |

约定：新增处理类型必须复用任务队列；新算法必须挂进回归门；新模型必须支持缺失
降级；不把重依赖（如 PyTorch）打进研究构建。

## 研究 CLI

```bash
uv run cthulhu gen                 # 生成合成样本（帧 npy + 音频 wav）
uv run cthulhu harness-video       # 参考水印 × 鲁棒性压力测试矩阵 → BER/PSNR/SSIM
uv run cthulhu harness-audio       # 回声隐藏水印 × 音频鲁棒性矩阵
uv run cthulhu probe 素材.mp4                # 容器 / 元数据 / SEI 探测
uv run cthulhu sample-diff clean.mp4 wm.mp4  # 对齐 + 差分报告 + DCT 热图
uv run cthulhu bitstream 素材.mp4            # 码流层 QP / 码量 / GOP 分析
uv run cthulhu detect-harness / detect-baseline / detect-watermark
uv run cthulhu attack-matrix / content-matrix / desensitize-matrix / cleanse-matrix
uv run cthulhu dedup-fingerprint / dedup-compare   # 判重代理指纹与候选排序
uv run cthulhu similarity a.mp4 b.mp4              # 内容 / 运动 / 哈希 / 画质相似度
uv run cthulhu desensitize in.mp4 out.mp4          # 单文件变换实验
uv run cthulhu benchmark / calibrate-presets       # 性能基准与预设校准
uv run cthulhu purify-status / purify-install      # 潜空间净化运行时状态与安装
```

具体命令参数见 `uv run cthulhu --help`；命令语义与边界见
[研究 CLI 说明](../README.md#研究-cli-与实验-api) 与
[实施文档](../docs/实施文档.md)。

## 服务端接口

| 接口 | 说明 |
| :--- | :--- |
| `GET /api/health` | 健康检查（版本、ffmpeg、主机信息），无需令牌 |
| `GET /api/library`、`POST /api/library`、`DELETE /api/library` | 素材库读写与去重导入 |
| `POST /api/detect`、`POST /api/similarity` | 参考检测报告、成对相似度 |
| `POST /api/desensitize`、`POST /api/jobs` | 单文件变换实验、批量任务入队 |
| `GET /api/jobs`、`/api/outputs`、`/api/templates`、`/api/settings` | 任务历史、产物清单、模板与设置 |
| `GET /api/thumb`、`/api/frame`、`/api/media` | 封面、抽帧与媒体流（供预览与对比） |
| `WS /ws/events` | 任务进度、状态与日志事件推送 |

除 `/api/health` 外，所有接口都要求携带令牌（请求头 `x-cthulhu-token` 或查询参数
`token`），令牌来自 `CTHULHU_AUTH_TOKEN`；未配置时启动会生成一次性令牌并打印到
日志。契约变更后必须重新导出 `openapi.json` 并同步前端类型。

## 常用环境变量

| 变量 | 作用 |
| :--- | :--- |
| `CTHULHU_AUTH_TOKEN` | API 令牌（开发模式由 `pnpm dev` 注入 `dev-local`） |
| `CTHULHU_PORT` / `CTHULHU_STATIC_DIR` | 监听端口、静态资源目录（桌面打包时托管前端产物） |
| `CTHULHU_DB` / `CTHULHU_LIBRARY_DIR` | SQLite 路径与素材目录（用于隔离实验环境） |
| `CTHULHU_THUMB_CACHE` / `CTHULHU_CACHE_DIR` / `CTHULHU_CACHE_MAX_BYTES` | 封面缓存、分析缓存与容量上限 |
| `CTHULHU_TRANSFORM_THREADS` / `CTHULHU_PROCESS_WORKERS` / `CTHULHU_SEGMENT_K` | 线程数、多进程接管与分块策略 |
| `CTHULHU_PURIFY_MODEL_DIR` / `CTHULHU_PURIFY_TAESD_MODEL` / `CTHULHU_PURIFY_ALLOW_DOWNLOAD` | 潜空间净化权重目录、模型 ID 与下载兜底开关 |
| `CTHULHU_FFMPEG_INSTALL_DIR` | 内置 ffmpeg 安装位置 |
| `CTHULHU_DEMO_LIBRARY` | 仅演示/截屏：置 `1` 时生成合成演示素材库 |

## 测试与调试

```bash
uv run pytest tests -q                 # 全量研究套件（含往返、鲁棒性、持久化）
uv run pytest tests/test_api.py -q     # 单个文件
uv run ruff check src tests            # 静态检查
uv run python scripts/export_openapi.py
```

报告与中间产物默认写入 `backend/data/`（已在 `.gitignore` 中忽略）。性能与画质
结论的口径、样本与失败案例见 [研究笔记](../docs/README.md#研究笔记)。
