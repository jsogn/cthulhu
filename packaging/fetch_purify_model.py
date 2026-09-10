#!/usr/bin/env python3
"""把潜空间净化权重（TAESD）准备到 packaging/models，供安装包内置（零下载）。

只准备 `madebyollin/taesd`（约 10MB）：潜空间瓶颈重建引擎的全部权重。原先
随包的 sd-turbo 已移除（research §19.1~19.3：清除率持平、保真度更高、快 50 倍，
但体积 2.4GB）。也支持 `--source-dir` 从已有目录复制，便于离线/重复构建。

用法：
    uv run --project backend --extra purify python packaging/fetch_purify_model.py
    python packaging/fetch_purify_model.py --source-dir ~/.../madebyollin--taesd
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TAESD_ID = os.environ.get("CTHULHU_PURIFY_TAESD_MODEL", "madebyollin/taesd")


def model_dir_name(model_id: str) -> str:
    return model_id.replace("/", "--")


def is_ready(path: Path) -> bool:
    """TAESD 目录是否可加载：config.json + 至少一个非空 safetensors。"""
    config = path / "config.json"
    if not config.is_file() or config.stat().st_size <= 0:
        return False
    return any(item.stat().st_size > 0 for item in path.glob("*.safetensors"))


def dir_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_TAESD_ID)
    parser.add_argument("--out", type=Path, default=ROOT / "packaging" / "models")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="已有 TAESD 目录（如应用数据目录下的 madebyollin--taesd），复制而不是下载",
    )
    parser.add_argument("--force", action="store_true", help="目标已存在时也重新准备")
    args = parser.parse_args()

    target = args.out / model_dir_name(args.model)
    if is_ready(target) and not args.force:
        print(f"TAESD 已就绪，跳过：{target}（{dir_size(target) / 1024**2:.1f} MB）")
        return

    args.out.mkdir(parents=True, exist_ok=True)
    if args.source_dir is not None:
        source = args.source_dir.expanduser().resolve()
        if not is_ready(source):
            raise SystemExit(f"源目录不是完整的 TAESD 权重：{source}")
        print(f"从本地复制 TAESD：{source} → {target}")
        shutil.copytree(source, target, dirs_exist_ok=True)
    else:
        from huggingface_hub import snapshot_download

        print(f"下载 TAESD {args.model}（约 10MB）→ {target}")
        target.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            args.model,
            local_dir=str(target),
            max_workers=4,
            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.md"],
        )

    if not is_ready(target):
        raise SystemExit(f"TAESD 权重校验失败：{target}")
    print(f"完成：{target}（{dir_size(target) / 1024**2:.1f} MB）")


if __name__ == "__main__":
    sys.exit(main())
