"""生成应用图标：从正方形品牌原图裁出圆角图标、多尺寸 ICO 与前端资源。"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent / "icons"
ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(__file__).resolve().parent / "logo-source.webp"
LOGO = ROOT / "ui" / "public" / "logo.webp"
FAVICON = ROOT / "ui" / "public" / "favicon.png"
ICON_SIZE = 512


def _run(*args: str) -> None:
    subprocess.run(["magick", *args], check=True)


def _source_size() -> tuple[int, int]:
    out = subprocess.check_output(
        ["magick", "identify", "-format", "%w %h", str(SOURCE)],
        text=True,
    ).strip()
    width, height = (int(value) for value in out.split())
    return width, height


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    width, height = _source_size()
    if width != height:
        raise RuntimeError(f"品牌原图必须是正方形，当前为 {width}x{height}")

    with tempfile.TemporaryDirectory(prefix="cthulhu-icon-") as tmp:
        square = Path(tmp) / "square.png"
        _run(str(SOURCE), "-resize", f"{ICON_SIZE}x{ICON_SIZE}", str(square))
        # 原图是满版方形背景，直接裁成 macOS 圆角方形即可。
        radius = round(ICON_SIZE * 0.22)
        _run(
            str(square),
            "(",
            "-size",
            f"{ICON_SIZE}x{ICON_SIZE}",
            "xc:none",
            "-fill",
            "white",
            "-draw",
            f"roundrectangle 0,0,{ICON_SIZE - 1},{ICON_SIZE - 1},{radius},{radius}",
            ")",
            "-alpha",
            "off",
            "-compose",
            "CopyOpacity",
            "-composite",
            "-depth",
            "8",
            "-strip",
            "-define",
            "png:compression-level=9",
            str(OUT_DIR / "icon.png"),
        )

    _run(
        str(OUT_DIR / "icon.png"),
        "-define",
        "icon:auto-resize=256,128,64,48,32,16",
        str(OUT_DIR / "icon.ico"),
    )
    _run(
        str(OUT_DIR / "icon.png"),
        "-resize",
        "256x256",
        "-strip",
        "-quality",
        "90",
        "-define",
        "webp:method=6",
        str(LOGO),
    )
    _run(
        str(OUT_DIR / "icon.png"),
        "-resize",
        "64x64",
        "-strip",
        "-define",
        "png:compression-level=9",
        str(FAVICON),
    )
    _run(
        str(OUT_DIR / "icon.png"),
        "-resize",
        "104x104",
        "-strip",
        "-define",
        "png:compression-level=9",
        str(OUT_DIR / "icon-104.png"),
    )
    print(
        f"图标已生成：{OUT_DIR / 'icon.png'} / {OUT_DIR / 'icon.ico'} / "
        f"{LOGO} / {FAVICON}"
    )


if __name__ == "__main__":
    main()
