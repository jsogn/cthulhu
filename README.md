<p align="center">
  <img src="docs/assets/banner.png" alt="Cthulhu · 本地离线视频去重工作台" width="880">
</p>

<p align="center">
  <a href="docs/使用边界与合规声明.md"><img alt="使用前提：自有 / 已授权素材" src="https://img.shields.io/badge/%E4%BD%BF%E7%94%A8%E5%89%8D%E6%8F%90-%E8%87%AA%E6%9C%89%20%2F%20%E5%B7%B2%E6%8E%88%E6%9D%83%E7%B4%A0%E6%9D%90-f59e0b"></a>
  <img alt="平台：macOS · Apple Silicon" src="https://img.shields.io/badge/%E5%B9%B3%E5%8F%B0-macOS%20%C2%B7%20Apple%20Silicon-111111?logo=apple&logoColor=white">
  <img alt="运行方式：本地离线" src="https://img.shields.io/badge/%E8%BF%90%E8%A1%8C-%E6%9C%AC%E5%9C%B0%E7%A6%BB%E7%BA%BF-5B7CFA">
  <img alt="Python 3.12" src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white">
  <img alt="Electron 桌面壳" src="https://img.shields.io/badge/Electron-%E6%A1%8C%E9%9D%A2%E5%A3%B3-47848F?logo=electron&logoColor=white">
</p>

<p align="center">
  本地离线的视频去重工作台<br>
  素材与产物都留在你自己的机器上
</p>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/screenshots/ui-workbench-dark.png">
    <img src="docs/assets/screenshots/ui-workbench-light.png" alt="工作台：左侧素材库、中间视频预览、右侧去重参数与产物面板" width="880">
  </picture>
</p>

> **使用边界**：仅处理自有或已获授权的素材；严禁用于规避平台规则、去除版权 / 溯源水印、侵权搬运、账号矩阵规避等违法违规用途。完整边界见 [使用边界与合规声明](docs/使用边界与合规声明.md)。

Cthulhu 面向自有或已获授权的视频素材，提供本地视频去重与版本管理：批量导入后识别重复文件、跳过重复导入，并通过画面重建、成组变换与重新编码生成可追溯的去重版本；同时可通过 CLI / API 生成处理前后的相似度、码流与画质报告。配合批量任务与产物管理，全程不上传、无账号。

> **「视频去重」的范围**：指在本地素材库中识别重复文件、跳过重复导入，并为需要保留的素材生成可追溯的新版本；可通过内置度量接口查看相似度与画质对照。不用于规避平台审核、版权识别或重复内容识别规则，效果以本地指标为准，不代表平台判定。

产品级定位是视频去重；具体能力会按目标层明确标注「暗水印」「判重指纹」等，尤其暗水印画面重建会提示更高的耗时与画质代价，避免用户把高成本处理误当成普通去重。

## 典型场景

- 素材库整理：识别内容相同的重复文件并跳过重复导入，统一管理素材记录
- 自有或授权素材的版本处理：为同一素材生成多档去重版本，保留参数快照与产物追溯
- 去重效果量化验证：通过 CLI / API 对照相似度、码流与画面指标，人工复核观感与画质代价

## 功能

- **视频去重**：重复文件识别与跳过、处理前后相似度对照（CLI / API）、可追溯的去重版本生成
- **工作台**：批量导入、封面与元信息、逐帧预览、「原片 / 处理后」对比
- **画面重建与变换**：分组可组合的变换原语与多档预设；可通过 CLI / API 获取画质代价（PSNR / SSIM / VMAF）
- **检测与度量**：容器 / 元数据 / SEI 扫描、码流分析、内容 / 运动 / 哈希 / 画质相似度
- **任务中心**：批量入队、并行调度、进度与取消、重启后续跑
- **模板与产物**：处理预设一键回填；产物追溯、参数快照、对比与清理
- **开箱即用**：FFmpeg 与画面重建运行时随包内置；无账号、无云服务，全程本地离线

## 界面

<p align="center">
  <img src="docs/assets/screenshots/ui-workbench-compare.png" alt="原片与处理产物的前后对比视图" width="880">
  <br>
  <sub>原片 / 处理后对比：同一时间轴上的滑块对照</sub>
</p>

<p align="center">
  <img src="docs/assets/screenshots/ui-jobs.png" alt="任务中心：队列统计与任务历史" width="880">
  <br>
  <sub>任务中心：队列统计与任务历史</sub>
</p>

完整界面说明见 [界面导览](docs/界面导览.md)（含截图复现步骤；截图均使用合成演示素材）。

## 快速开始

环境：Node.js 22+、pnpm 11+、uv（Python 3.12）、FFmpeg。

```bash
pnpm install && pnpm --dir ui install && pnpm --dir electron install
cd backend && uv sync --extra purify && cd ..

pnpm dev            # 开发模式：浏览器打开 http://localhost:5173
pnpm dev:desktop    # 桌面模式：Electron 窗口（主进程自动拉起后端）
```

> 试用时可用独立数据目录，不影响日常素材库：
> `CTHULHU_DB=/tmp/cthulhu-demo/cthulhu.db pnpm dev`。

## 打包与发布

```bash
pnpm package        # 前端构建 → 后端打包 → electron-builder dmg
```

产物：`packaging/dist/Cthulhu-<版本>-arm64.dmg`。当前为未签名构建；签名 / 公证与
Windows 打包为待办项。打包版依赖系统 ffmpeg，也可内置静态构建
（`packaging/fetch_ffmpeg.sh`）。

## 开发

技术栈：Electron + React 19（Vite） + Python 3.12 FastAPI sidecar + FFmpeg /
ONNX Runtime，数据存本地 SQLite。

```bash
pnpm typecheck && pnpm test
cd backend && uv run ruff check src tests && uv run pytest tests -q
uv run python scripts/export_openapi.py   # 契约漂移检查
```

后端 CLI、REST / WebSocket 接口与模块地图见 [backend/README.md](backend/README.md)。

## 文档

- [文档索引](docs/README.md)
- [使用边界与合规声明](docs/使用边界与合规声明.md) · [产品定位与架构](docs/产品定位与架构.md)
- [界面导览](docs/界面导览.md)
- [research/README.md](research/README.md)：研究档案（公开方案报告与脚本）

## 许可与使用边界

- 本仓库未附开源许可证，默认保留全部权利；许可范围外使用请先与维护者确认。
- 仅处理自有或已获授权的素材；禁止规避平台规则、去除版权 / 溯源水印、侵权搬运、
  账号矩阵等用途（详见 [使用边界与合规声明](docs/使用边界与合规声明.md)）。
- 不提供「保证过审 / 保证不限流 / 保证不被识别」等承诺；本地指标不代表平台真实判定。
