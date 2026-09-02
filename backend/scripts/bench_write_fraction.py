"""编码写回耗时探测：判断「写线程重叠」在目标机器上是否值得实现。

用法（在 backend 目录下）：

    uv run python scripts/bench_write_fraction.py <video> [--chunk 48] [--max-frames 480]

原理：跳过武器变换，按真实管线节奏解码一批帧直写编码器，单独累计
解码等待与 encoder.write 耗时。write 的每帧毫秒数就是写线程理论上最多
能回收的量——它与武器变换耗时（均衡档约 55~89ms/帧）相比才有意义。

决策口径：
    write < 3ms/帧   收益约 2~5%，不建议实现；
    write 3~8ms/帧   边际收益，视目标档位再定；
    write > 8ms/帧   预计 ≥10%，值得实现。
"""

from __future__ import annotations

import argparse
import os
import tempfile
import time

from cthulhu_backend.media import ffmpeg


def main() -> None:
    parser = argparse.ArgumentParser(description="编码写回耗时探测")
    parser.add_argument("video", help="被测视频路径")
    parser.add_argument("--chunk", type=int, default=48, help="每批帧数（对齐 phash 路径默认）")
    parser.add_argument("--max-frames", type=int, default=480, help="最多探测帧数，0 表示全片")
    args = parser.parse_args()

    info = ffmpeg.video_info(args.video)
    total = max(1, round(info["duration"] * info["fps"]))
    count = min(total, args.max_frames) if args.max_frames else total

    with tempfile.TemporaryDirectory(prefix="cthulhu-write-bench-") as tmp:
        out_path = os.path.join(tmp, "out.mp4")
        decoder = ffmpeg.StreamingDecoder(args.video, 0, count, grayscale=False)
        encoder = ffmpeg.StreamingEncoder(
            out_path,
            info["width"],
            info["height"],
            info["fps"],
            codec="libx264",
            hardware=False,
            crf=23,
            preset="veryfast",
            color=True,
            input_pix_fmt="rgb24",
        )
        read_time = 0.0
        write_time = 0.0
        frames = 0
        wall_start = time.perf_counter()
        try:
            while frames < count:
                start = time.perf_counter()
                batch = decoder.read(min(args.chunk, count - frames), dtype="uint8")
                read_time += time.perf_counter() - start
                if len(batch) == 0:
                    break
                start = time.perf_counter()
                encoder.write(batch)
                write_time += time.perf_counter() - start
                frames += len(batch)
        finally:
            decoder.close()
            encoder.finish()
        wall = time.perf_counter() - wall_start

    write_ms = write_time / frames * 1000 if frames else 0.0
    read_ms = read_time / frames * 1000 if frames else 0.0
    print(f"frames={frames} wall={wall:.2f}s read={read_time:.2f}s write={write_time:.2f}s")
    print(f"per-frame: write={write_ms:.2f}ms decode={read_ms:.2f}ms")
    if write_ms >= 8.0:
        verdict = "值得实现（预计 ≥10%）"
    elif write_ms >= 3.0:
        verdict = "边际收益，视目标武器档位再定"
    else:
        verdict = "收益有限（约 2~5%），不建议实现"
    print(f"结论：{verdict}")


if __name__ == "__main__":
    main()
