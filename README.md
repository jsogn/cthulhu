<p align="center">
  <img src="docs/assets/banner.webp" alt="Cthulhu · 本地视频去重工作台" width="880">
</p>

<p align="center">
  <a href="docs/使用边界与合规声明.md"><img alt="使用前提：自有 / 已授权素材" src="https://img.shields.io/badge/%E4%BD%BF%E7%94%A8%E5%89%8D%E6%8F%90-%E8%87%AA%E6%9C%89%20%2F%20%E5%B7%B2%E6%8E%88%E6%9D%83%E7%B4%A0%E6%9D%90-f59e0b"></a>
  <img alt="平台：macOS · Apple Silicon" src="https://img.shields.io/badge/%E5%B9%B3%E5%8F%B0-macOS%20%C2%B7%20Apple%20Silicon-111111?logo=apple&logoColor=white">
  <img alt="运行方式：本地离线" src="https://img.shields.io/badge/%E8%BF%90%E8%A1%8C-%E6%9C%AC%E5%9C%B0%E7%A6%BB%E7%BA%BF-5B7CFA">
  <img alt="Python 3.12" src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white">
  <img alt="Electron 桌面应用" src="https://img.shields.io/badge/Electron-%E6%A1%8C%E9%9D%A2%E5%BA%94%E7%94%A8-47848F?logo=electron&logoColor=white">
</p>

<p align="center">
  本地视频去重、批量变换与产物管理<br>
  素材和导出文件留在自己的电脑上
</p>

<p align="center">
  <img src="docs/assets/screenshots/ui-workbench-light.webp" alt="Cthulhu 工作台：素材库、真实视频预览和视频去重参数" width="880">
</p>

> **使用边界**：仅处理自有或已获授权的视频；严禁用于规避平台规则、去除版权 /
> 溯源标识、侵权搬运、账号矩阵规避或其他违法违规用途。完整说明见
> [使用边界与合规声明](docs/使用边界与合规声明.md)。

Cthulhu 是一个本地桌面工作台，把视频导入、播放预览、批量变换、任务调度和产物管理
收在同一个界面里。它适合整理同一素材的多个版本、批量生成独立的处理版本，并在导出
前后人工检查画面、参数和文件占用。

这里的“去重”有两层含义：

- 导入时根据文件内容签名跳过重复素材；
- 处理时为原片生成新的去重版本，不覆盖也不删除原片。

本地指标只描述文件与画面变化，不代表任何平台的实际判定。处理后的文件需要自行播放、
抽查和复核。

## 主要能力

Cthulhu 的核心工作流集中在工作台：

- 导入视频或文件夹，导入时跳过重复内容；支持播放、逐帧预览和进度拖动。
- 使用内置模板或手动调整参数，按画面重建、画面变换、音频、重编码、画质和输出
  分组控制处理方式。
- 单条或批量执行去重，并在任务中心查看进度、暂停、继续或重试。
- 生成与原片独立的去重版本，可在工作台播放或用滑块对比，确认后再导出使用。

画面重建和神经类选项耗时、画质代价更高，建议先用短视频验证。

使用流程：导入素材 → 预览确认 → 选择模板或调整参数 → 执行去重 → 对比处理产物。

## 快速开始

环境要求：

- macOS Apple Silicon
- Node.js 22+、pnpm 11+
- uv（Python 3.12）
- FFmpeg 与 ffprobe

```bash
pnpm install
pnpm --dir ui install
pnpm --dir electron install

cd backend
uv sync --extra purify
cd ..

pnpm dev:desktop   # 桌面应用：Electron + Vite + Python 后端
```

只在浏览器中开发界面时使用：

```bash
pnpm dev           # http://localhost:5173
```

需要临时避开日常素材库时，可指定独立数据库：

```bash
CTHULHU_DB=/tmp/cthulhu-demo/cthulhu.db pnpm dev
```

## 打包

先准备静态 FFmpeg，再执行完整打包：

```bash
bash packaging/fetch_ffmpeg.sh
pnpm package
```

`pnpm package` 会构建前端、拉取画面的内置重建模型并打包 Python sidecar，最后生成
`packaging/dist/Cthulhu-<版本>-arm64.dmg`。当前 macOS 产物未签名、未公证；Windows
已有打包配置，但尚未在仓库中提供已验证的正式安装包。

## 开发与验证

技术栈：Electron + React 19（Vite） + Python 3.12 FastAPI sidecar + FFmpeg /
ONNX Runtime，本地数据使用 SQLite。

```bash
pnpm typecheck
pnpm test

cd backend
uv run ruff check src tests
uv run pytest tests -q
```

后端 CLI、REST / WebSocket 接口和模块说明见 [backend/README.md](backend/README.md)。

## 文档

- [文档索引](docs/README.md)
- [界面导览](docs/界面导览.md)
- [使用边界与合规声明](docs/使用边界与合规声明.md)
- [产品定位与架构](docs/产品定位与架构.md)
- [实施文档](docs/实施文档.md)
- [发布打包](docs/发布打包.md)
- [research/README.md](research/README.md)：研究档案与实验脚本

## 许可与使用边界

- 本仓库未附开源许可证，默认保留全部权利；许可范围外使用前请先与维护者确认。
- 仅处理自有或已获授权的素材；禁止规避平台规则、去除版权 / 溯源标识、侵权搬运、
  账号矩阵等用途。
- 不提供“保证过审”“保证不限流”“保证不被识别”等承诺；本地测试结果不代表平台结果。
