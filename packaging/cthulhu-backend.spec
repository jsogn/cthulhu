# -*- mode: python ; coding: utf-8 -*-

import os

from PyInstaller.utils.hooks import collect_all, copy_metadata

_ROOT = os.path.abspath(os.path.join(os.path.dirname(SPEC), ".."))

datas, binaries, hiddenimports = [], [], []
# 开箱即用（A）：运行时依赖必须进包；扩散权重首次使用时下载到应用数据目录，
# 不塞进安装包（约 2.5GB，且不同设备精度不同）。
_PURIFY_PACKAGES = (
    "torch",
    "diffusers",
    "transformers",
    "accelerate",
    "safetensors",
    "huggingface_hub",
    "tokenizers",
    "tqdm",
)
for package in ("scipy", "numpy", *_PURIFY_PACKAGES):
    try:
        collected_datas, collected_binaries, collected_hidden = collect_all(package)
    except Exception as exc:  # noqa: BLE001 - 打包依赖缺失必须显式失败
        raise SystemExit(
            f"缺少打包依赖 {package}；请先执行 "
            "`uv sync --project backend --extra purify`"
        ) from exc
    datas += collected_datas
    binaries += collected_binaries
    hiddenimports += collected_hidden

# 运行期版本校验：diffusers 在 import 时用 importlib.metadata 检查
# requests/filelock/numpy 的元数据（transformers 的同类检查已由 collect_all 覆盖）。
# PyInstaller 只带 collect_all 过的包的 dist-info，requests 这种纯运行期依赖会缺
# 元数据并抛 PackageNotFoundError —— 实测后果是打包后净化引擎静默降级
# （purify_note: "fallback: ... requests distribution was not found"）。
for distribution in ("requests",):
    try:
        datas += copy_metadata(distribution)
    except Exception as exc:  # noqa: BLE001 - 缺元数据必须让构建直接失败
        raise SystemExit(
            f"缺少运行期依赖元数据 {distribution}：{exc}；"
            "请先执行 `uv sync --project backend --extra purify`"
        ) from exc

# 扩散净化权重随包内置：安装后零下载。缺失时构建直接失败，避免产出
# “运行时可用但模型要用户自己下”的包。
_MODEL_DIR = os.path.join(_ROOT, "packaging", "models")
if not os.path.isdir(_MODEL_DIR):
    raise SystemExit("缺少内置扩散净化模型：请先执行 `pnpm package:model`")
datas.append((_MODEL_DIR, "models"))

a = Analysis(
    [os.path.join(_ROOT, "backend", "launcher.py")],
    pathex=[os.path.join(_ROOT, "backend", "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["tkinter"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    exclude_binaries=True,
    name="cthulhu-backend",
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    name="cthulhu-backend",
)
