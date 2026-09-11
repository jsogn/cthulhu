"""检测基准快照与策略 A/B 记录工具。

用途：把测试素材的盲检测信号得分冻结成 JSON 基准；新策略处理后追加
运行记录，并与基准逐项对比。策略只有在"盲检测分数不劣于基准"且
墙钟时间更优时才允许切换为默认，保证每次提速都有检测数据背书。

用法：
    python -m cthulhu_backend.baseline snapshot <视频...> -o 基准.json
    python -m cthulhu_backend.baseline run <输入> <输出> -o 记录.json -s fast ...
    python -m cthulhu_backend.baseline compare --baseline 基准.json --runs 记录.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cthulhu_backend import services
from cthulhu_backend.media import ffmpeg
from cthulhu_backend.watermark import detect

SCHEMA_VERSION = 1
SAMPLED_FRAMES = 300
AUDIO_SECONDS = 120


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _ffmpeg_version() -> str:
    try:
        result = subprocess.run(
            [ffmpeg.FFMPEG_BIN, "-version"], capture_output=True, text=True, check=True
        )
        return result.stdout.splitlines()[0].strip()
    except Exception:  # noqa: BLE001 - 版本信息缺失不阻塞快照
        return "unknown"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(payload: dict, out_file: str) -> None:
    target = Path(out_file)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def probe_input(path: str) -> dict:
    """抽样解码素材，输出探针信息与盲检测信号得分。"""
    target = Path(path)
    info = ffmpeg.video_info(path)
    frames, _ = ffmpeg.decode_sampled(path, cap=SAMPLED_FRAMES)
    scores: dict[str, Any] = detect.video_scores(frames)
    audio = ffmpeg.decode_audio(path, max_seconds=AUDIO_SECONDS)
    if audio is not None:
        signal, sample_rate = audio
        scores["echo"] = detect.audio_scores(signal, sample_rate)["echo"]
    else:
        scores["echo"] = None
    return {
        "name": target.name,
        "path": str(target.resolve()),
        "size": target.stat().st_size,
        "sha256": _sha256(target),
        "probe": info,
        "frames_sampled": len(frames),
        "scores": scores,
    }


def snapshot_inputs(paths: list[str], out_file: str) -> dict:
    """冻结一组输入素材的检测基准，返回写入的完整 payload。"""
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at": _now(),
        "ffmpeg_version": _ffmpeg_version(),
        "files": [probe_input(path) for path in paths],
    }
    write_json(payload, out_file)
    return payload


def run_and_record(
    path: str,
    output: str,
    out_file: str,
    strategy: str = "thorough",
    **options: Any,
) -> dict:
    """跑一次脱敏并记录墙钟时间、产物信号得分，追加到运行记录文件。"""
    started = time.monotonic()
    result = services.run_desensitize(path, output, transform_strategy=strategy, **options)
    wall = time.monotonic() - started

    out_path = Path(output)
    out_frames, _ = ffmpeg.decode_sampled(output, cap=SAMPLED_FRAMES)
    scores: dict[str, Any] = detect.video_scores(out_frames)
    audio = ffmpeg.decode_audio(output, max_seconds=AUDIO_SECONDS)
    if audio is not None:
        signal, sample_rate = audio
        scores["echo"] = detect.audio_scores(signal, sample_rate)["echo"]
    else:
        scores["echo"] = None

    record = {
        "schema_version": SCHEMA_VERSION,
        "created_at": _now(),
        "strategy": result.get("transform_strategy", strategy),
        "input": str(Path(path).resolve()),
        "output": str(out_path.resolve()),
        "input_size": Path(path).stat().st_size,
        "output_size": out_path.stat().st_size,
        "wall_seconds": round(wall, 3),
        "scores": scores,
        "similarity_after": result.get("similarity_after"),
        "psnr_db": result.get("psnr_db"),
        "ssim": result.get("ssim"),
        "vmaf_aligned": result.get("vmaf_aligned"),
    }
    append_record(out_file, record)
    return record


def append_record(out_file: str, record: dict) -> None:
    path = Path(out_file)
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = {"schema_version": SCHEMA_VERSION, "records": []}
    payload.setdefault("records", []).append(record)
    write_json(payload, out_file)


def compare(baseline_file: str, runs_file: str) -> dict:
    """把运行记录按输入路径关联回基准，输出信号差值（越负越干净）与耗时。"""
    baseline = json.loads(Path(baseline_file).read_text(encoding="utf-8"))
    runs = json.loads(Path(runs_file).read_text(encoding="utf-8"))
    by_input = {entry["path"]: entry for entry in baseline.get("files", [])}
    rows: list[dict] = []
    for record in runs.get("records", []):
        entry = by_input.get(record.get("input"))
        if entry is None:
            rows.append({"strategy": record.get("strategy"), "input": "（未匹配基准）"})
            continue
        base_scores = entry["scores"]
        deltas: dict[str, Any] = {}
        for key in ("ss", "qim", "lsb", "echo"):
            base_value = base_scores.get(key)
            run_value = record["scores"].get(key)
            deltas[key] = (
                round(run_value - base_value, 4)
                if isinstance(base_value, (int, float)) and isinstance(run_value, (int, float))
                else None
            )
        rows.append(
            {
                "strategy": record.get("strategy"),
                "input": Path(record.get("input", "")).name,
                "wall_seconds": record.get("wall_seconds"),
                "signal_delta": deltas,
                "similarity_after": record.get("similarity_after"),
            }
        )
    return {"rows": rows}


def _build_run_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("-o", "--out", required=True, help="运行记录 JSON 路径")
    parser.add_argument("-s", "--strategy", default="thorough")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--recrop", type=float, default=0.0)
    parser.add_argument("--perturb", type=float, default=0.0)
    parser.add_argument("--codec", default="libx264")
    parser.add_argument("--preset", default="medium")
    parser.add_argument("--phash-attack", action="store_true")
    parser.add_argument("--phash-epsilon", type=float, default=0.03)
    parser.add_argument("--phash-iters", type=int, default=120)
    parser.add_argument("--multi-hash-attack", action="store_true")
    parser.add_argument("--rotate", type=float, default=0.0)
    parser.add_argument("--median", type=int, default=0)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--requant", type=int, default=0)
    parser.add_argument("--dct-step", type=float, default=0.0)
    parser.add_argument("--jitter", type=float, default=0.0)
    parser.add_argument("--perspective", type=float, default=0.0)
    parser.add_argument("--warp", type=float, default=0.0)
    parser.add_argument("--no-regrade", action="store_true")
    parser.add_argument("--no-sharpness", action="store_true")
    parser.add_argument("--no-color-restore", action="store_true")
    parser.add_argument("--denoise", action="store_true")
    parser.add_argument("--anti-reembed", action="store_true")
    parser.add_argument("--spoof", action="store_true")


def _main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="cthulhu-baseline", description="检测基准与策略 A/B 工具")
    sub = parser.add_subparsers(dest="command", required=True)

    snapshot = sub.add_parser("snapshot", help="冻结输入素材的盲检测信号基准")
    snapshot.add_argument("files", nargs="+", help="素材路径（可多个）")
    snapshot.add_argument("-o", "--out", required=True, help="基准 JSON 输出路径")

    _build_run_parser(sub.add_parser("run", help="跑一次脱敏并追加运行记录"))

    compare_parser = sub.add_parser("compare", help="对比基准与运行记录")
    compare_parser.add_argument("--baseline", required=True)
    compare_parser.add_argument("--runs", required=True)

    args = parser.parse_args(argv)
    if args.command == "snapshot":
        snapshot_inputs(args.files, args.out)
        print(f"基准已写入：{args.out}")
    elif args.command == "run":
        record = run_and_record(
            args.input,
            args.output,
            args.out,
            strategy=args.strategy,
            seed=args.seed,
            speed=args.speed,
            recrop=args.recrop,
            perturb=args.perturb,
            codec=args.codec,
            preset=args.preset,
            phash_attack=args.phash_attack,
            phash_epsilon=args.phash_epsilon,
            phash_iters=args.phash_iters,
            multi_hash_attack=args.multi_hash_attack,
            rotate=args.rotate,
            median=args.median,
            noise=args.noise,
            requant=args.requant,
            dct_step=args.dct_step,
            jitter=args.jitter,
            perspective=args.perspective,
            warp=args.warp,
            regrade=not args.no_regrade,
            sharpness=not args.no_sharpness,
            color_restore=not args.no_color_restore,
            denoise=args.denoise,
            anti_reembed=args.anti_reembed,
            spoof=args.spoof,
        )
        print(json.dumps(record, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(compare(args.baseline, args.runs), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _main()
