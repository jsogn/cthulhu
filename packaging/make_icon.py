"""生成应用图标：从品牌几何图形渲染 1024 PNG 与多尺寸 ICO。"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

OUT_DIR = Path(__file__).resolve().parent / "icons"
SIZE = 1024


def render(size: int = SIZE) -> Image.Image:
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    # 圆角深色底
    draw.rounded_rectangle([0, 0, size - 1, size - 1], radius=int(size * 0.22), fill=(16, 19, 26, 255))

    def pt(x: float, y: float) -> tuple[float, float]:
        return (x * size / 24, y * size / 24)

    accent = (91, 124, 250, 255)
    soft = (91, 124, 250, 90)

    # 品牌几何：斜置六边形（水印标记）
    hexagon = [pt(4.5, 11), pt(9, 7), pt(14, 8.2), pt(19, 10.6), pt(19, 13), pt(14, 14.2), pt(9, 16), pt(4.5, 13.4)]
    draw.polygon(hexagon, outline=accent, width=max(6, size // 48))

    # 内部波纹
    draw.line([pt(4.5, 11), pt(9, 7), pt(14, 8.2), pt(19, 10.6)], fill=soft, width=max(5, size // 56))
    draw.line([pt(4.5, 13.4), pt(9, 16), pt(14, 14.2), pt(19, 13)], fill=soft, width=max(5, size // 56))
    draw.line([pt(9, 7.4), pt(9, 15.6)], fill=soft, width=max(4, size // 64))
    draw.line([pt(14, 8.6), pt(14, 13.8)], fill=soft, width=max(4, size // 64))
    return image


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    image = render()
    image.save(OUT_DIR / "icon.png")
    image.save(
        OUT_DIR / "icon.ico",
        sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
    )
    print(f"图标已生成：{OUT_DIR / 'icon.png'} / {OUT_DIR / 'icon.ico'}")


if __name__ == "__main__":
    main()
