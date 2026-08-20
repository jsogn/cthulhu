"""MP4 容器扫描：udta/meta/XMP/异常 track 检测与 SEI 统计。"""

from __future__ import annotations

import shutil
import struct
import subprocess
from dataclasses import dataclass

# 允许递归解析的容器型 box。
CONTAINER_BOXES = {
    "moov", "trak", "mdia", "minf", "stbl", "dinf", "edts",
    "udta", "meta", "moof", "traf", "mvex",
}


@dataclass
class Box:
    path: str
    type: str
    size: int


def _walk(data: bytes, offset: int, end: int, path: str, out: list[Box]) -> None:
    while offset + 8 <= end:
        size, box_type = struct.unpack_from(">I4s", data, offset)
        header = 8
        if size == 1:
            (size,) = struct.unpack_from(">Q", data, offset + 8)
            header = 16
        elif size == 0:
            size = end - offset
        if size < header:
            break
        box_type_s = box_type.decode("latin1")
        box_path = f"{path}/{box_type_s}" if path else box_type_s
        out.append(Box(box_path, box_type_s, size))
        if box_type_s in CONTAINER_BOXES:
            _walk(data, offset + header, min(offset + size, end), box_path, out)
        offset += size


def scan_mp4(path: str) -> dict:
    with open(path, "rb") as fh:
        data = fh.read()
    boxes: list[Box] = []
    _walk(data, 0, len(data), "", boxes)
    paths = [b.path for b in boxes]
    has_xmp = any(p.endswith("XMP_") for p in paths)
    return {
        "udta_boxes": sum(1 for p in paths if p == "udta" or p.endswith("/udta")),
        "meta_boxes": sum(1 for p in paths if p == "meta" or p.endswith("/meta")),
        "has_xmp": has_xmp,
        "n_trak": sum(1 for p in paths if p.endswith("/trak")),
        "suspicious": ["XMP_ 元数据"] if has_xmp else [],
        "boxes": [{"path": b.path, "size": b.size} for b in boxes if b.path.count("/") <= 3],
    }


def count_sei(path: str) -> int | None:
    """通过 trace_headers 位流过滤器统计 SEI NAL 数量；失败返回 None。"""
    from cthulhu_backend.media.ffmpeg import FFMPEG_BIN

    if shutil.which(FFMPEG_BIN) is None:
        return None
    result = subprocess.run(
        [FFMPEG_BIN, "-v", "info", "-i", path, "-c", "copy", "-bsf:v", "trace_headers", "-f", "null", "-"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stderr.count("SEI")
