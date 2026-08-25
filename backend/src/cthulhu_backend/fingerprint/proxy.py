"""平台代理评估器：两段视频的多维剩余相似度与判重风险报告。

判定视角模仿平台查重：把「原片」当作已登记指纹库，对「待测件」的每一帧
做最近邻查询。若待测件多帧能低距离命中原片指纹，则平台大概率判重。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cthulhu_backend.fingerprint import audiofp, hashes
from cthulhu_backend.media import ffmpeg
from cthulhu_backend.similarity import embedding

HASH_KINDS = ("phash", "dhash", "ahash", "dct_sign")


def frame_signatures(frames: np.ndarray) -> dict[str, np.ndarray]:
    """为每帧计算哈希与浅层特征签名。"""
    return {
        "phash": np.stack([hashes.phash(frame) for frame in frames]),
        "dhash": np.stack([embedding.dhash(frame) for frame in frames]),
        "ahash": np.stack([hashes.ahash(frame) for frame in frames]),
        "dct_sign": np.stack([hashes.dct_sign(frame) for frame in frames]),
        "edge": np.stack([hashes.edge_embedding(frame) for frame in frames]),
        "grid": np.stack([hashes.interest_grid(frame) for frame in frames]),
    }


def _aligned_pairs(n_ref: int, n_cand: int) -> list[tuple[int, int]]:
    """把两段抽帧按时间轴比例对齐成帧对。"""
    n = max(n_ref, n_cand, 1)
    return [
        (
            round(i * (n_ref - 1) / max(n - 1, 1)),
            round(i * (n_cand - 1) / max(n - 1, 1)),
        )
        for i in range(n)
    ]


def video_compare(ref_frames: np.ndarray, cand_frames: np.ndarray) -> dict:
    """视频维度的最近邻哈希命中率与 embedding 相似度。"""
    ref = frame_signatures(ref_frames)
    cand = frame_signatures(cand_frames)
    report: dict = {}
    for kind in HASH_KINDS:
        distances = [
            min(hashes.hamming_bits(query, target) for target in ref[kind])
            for query in cand[kind]
        ]
        bits = int(ref[kind][0].size)
        report[kind] = {
            "bits": bits,
            "mean_min_hamming": round(float(np.mean(distances)), 3),
            "normalized_distance": round(float(np.mean(distances)) / bits, 4),
            "flag_rate_le_2": round(float(np.mean([d <= 2 for d in distances])), 4),
            "flag_rate_le_4": round(float(np.mean([d <= 4 for d in distances])), 4),
        }
    pairs = _aligned_pairs(len(ref_frames), len(cand_frames))
    report["content_cosine"] = round(
        hashes.cosine(
            embedding.video_content_embedding(ref_frames),
            embedding.video_content_embedding(cand_frames),
        ),
        4,
    )
    report["motion_cosine"] = round(
        hashes.cosine(
            embedding.video_motion_embedding(ref_frames),
            embedding.video_motion_embedding(cand_frames),
        ),
        4,
    )
    report["edge_cosine"] = round(
        float(np.mean([hashes.cosine(ref["edge"][i], cand["edge"][j]) for i, j in pairs])),
        4,
    )
    report["keypoint_grid_overlap"] = round(
        float(
            np.mean(
                [
                    np.minimum(ref["grid"][i], cand["grid"][j]).sum()
                    for i, j in pairs
                ]
            )
        ),
        4,
    )
    try:
        from cthulhu_backend.fingerprint import deep

        report["deep_cosine"] = (
            round(
                deep.cosine(
                    deep.video_embedding(ref_frames),
                    deep.video_embedding(cand_frames),
                ),
                4,
            )
            if deep.available()
            else None
        )
    except Exception:  # noqa: BLE001 - 模型异常不影响其它代理维度
        report["deep_cosine"] = None
    return report


def audio_compare(
    ref_signal: np.ndarray,
    cand_signal: np.ndarray,
    sample_rate: int,
) -> dict:
    """音频维度的谱哈希距离。"""
    mel_ref = audiofp.mel_hash(ref_signal, sample_rate)
    mel_cand = audiofp.mel_hash(cand_signal, sample_rate)
    mfcc_ref = audiofp.mfcc_hash(ref_signal, sample_rate)
    mfcc_cand = audiofp.mfcc_hash(cand_signal, sample_rate)
    return {
        "mel_hash": {
            "bits": len(mel_ref),
            "hamming": audiofp.hamming_bits(mel_ref, mel_cand),
            "normalized_distance": round(
                audiofp.hamming_bits(mel_ref, mel_cand) / max(len(mel_ref), 1), 4
            ),
        },
        "mfcc_hash": {
            "bits": len(mfcc_ref),
            "hamming": audiofp.hamming_bits(mfcc_ref, mfcc_cand),
            "normalized_distance": round(
                audiofp.hamming_bits(mfcc_ref, mfcc_cand) / max(len(mfcc_ref), 1), 4
            ),
        },
    }


def compare_files(a: str, b: str, cap: int = 120, audio_seconds: int = 90) -> dict:
    """对比两段视频，返回代理评估报告（a 为原片/登记侧，b 为待测件）。"""
    ref_frames, _ = ffmpeg.decode_sampled(a, cap=cap)
    cand_frames, _ = ffmpeg.decode_sampled(b, cap=cap)
    video = video_compare(ref_frames, cand_frames)
    audio = {}
    ref_audio = ffmpeg.decode_audio(a, max_seconds=audio_seconds)
    cand_audio = ffmpeg.decode_audio(b, max_seconds=audio_seconds)
    if ref_audio is not None and cand_audio is not None:
        audio = audio_compare(ref_audio[0], cand_audio[0], ref_audio[1])
    report = {
        "ref": str(Path(a).resolve()),
        "cand": str(Path(b).resolve()),
        "video": video,
        "audio": audio,
        "risk": _risk_score(video, audio),
    }
    return report


def _risk_score(video: dict, audio: dict) -> dict:
    """把多维指标压缩成 0~1 的判重风险与逐维判定，阈值是研究口径。"""
    dimensions: dict[str, float] = {}
    for kind in HASH_KINDS:
        entry = video.get(kind, {})
        rate = entry.get("flag_rate_le_2", 0.0)
        dimensions[f"hash_{kind}"] = min(1.0, rate * 4.0)
    dimensions["content"] = max(0.0, min(1.0, (video.get("content_cosine", 0.0) - 0.85) / 0.15))
    dimensions["edge"] = max(0.0, min(1.0, (video.get("edge_cosine", 0.0) - 0.8) / 0.2))
    deep_cosine = video.get("deep_cosine")
    if isinstance(deep_cosine, (int, float)):
        dimensions["deep"] = max(0.0, min(1.0, (deep_cosine - 0.7) / 0.3))
    for key in ("mel_hash", "mfcc_hash"):
        entry = audio.get(key)
        if entry:
            dimensions[f"audio_{key}"] = max(0.0, 1.0 - entry["normalized_distance"] * 4.0)
    score = round(float(np.mean(list(dimensions.values()))), 4) if dimensions else 0.0
    level = "低" if score < 0.4 else ("中" if score < 0.7 else "高")
    return {"score": score, "level": level, "dimensions": {k: round(v, 4) for k, v in dimensions.items()}}


def matrix(files: list[str], out_file: str | None = None, cap: int = 120) -> dict:
    """一组文件的两两对比矩阵（首列为原片侧，行内为待测件）。"""
    rows: dict[str, dict] = {}
    for ref in files:
        row = {}
        for cand in files:
            row[str(Path(cand).resolve())] = compare_files(ref, cand, cap=cap)["risk"]["score"]
        rows[str(Path(ref).resolve())] = row
    payload = {"files": [str(Path(f).resolve()) for f in files], "risk_matrix": rows}
    if out_file:
        target = Path(out_file)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def gate_report(
    ref: str,
    old: str,
    new: str,
    cap: int = 80,
    tolerance: float = 0.03,
) -> dict:
    """性能优化回归门：同一原片下比较新旧实现的对抗指标，不得劣化。"""
    from cthulhu_backend.watermark import detect

    ref_frames, _ = ffmpeg.decode_sampled(ref, cap=cap)

    def measure(path: str) -> dict:
        frames, _ = ffmpeg.decode_sampled(path, cap=cap)
        video = video_compare(ref_frames, frames)
        payload = detect.video_scores(frames)
        return {
            "phash_hit": video["phash"]["flag_rate_le_2"],
            "dhash_hit": video["dhash"]["flag_rate_le_2"],
            "ahash_hit": video["ahash"]["flag_rate_le_2"],
            "dctsign_hit": video["dct_sign"]["flag_rate_le_2"],
            "deep": video.get("deep_cosine") or 0.0,
            "ss": payload["ss"],
            "qim": payload["qim"],
        }

    old_metrics = measure(old)
    new_metrics = measure(new)
    delta = {key: round(new_metrics[key] - old_metrics[key], 4) for key in old_metrics}
    passed = {key: value <= tolerance for key, value in delta.items()}
    return {
        "old": old_metrics,
        "new": new_metrics,
        "delta": delta,
        "passed": passed,
        "all_pass": all(passed.values()),
    }


def _main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="cthulhu-proxy", description="平台判重代理评估器")
    sub = parser.add_subparsers(dest="command", required=True)

    compare = sub.add_parser("compare", help="对比两段视频")
    compare.add_argument("ref")
    compare.add_argument("cand")
    compare.add_argument("--cap", type=int, default=120)
    compare.add_argument("-o", "--out")

    mat = sub.add_parser("matrix", help="一组文件两两对比")
    mat.add_argument("files", nargs="+")
    mat.add_argument("--cap", type=int, default=120)
    mat.add_argument("-o", "--out")

    gate = sub.add_parser("gate", help="性能优化回归门：新实现不得劣于旧实现")
    gate.add_argument("--ref", required=True)
    gate.add_argument("--old", required=True)
    gate.add_argument("--new", required=True)
    gate.add_argument("--cap", type=int, default=80)
    gate.add_argument("--tolerance", type=float, default=0.03)

    args = parser.parse_args(argv)
    if args.command == "compare":
        payload = compare_files(args.ref, args.cand, cap=args.cap)
    elif args.command == "gate":
        payload = gate_report(args.ref, args.old, args.new, cap=args.cap, tolerance=args.tolerance)
    else:
        payload = matrix(args.files, out_file=args.out, cap=args.cap)
    if getattr(args, "out", None) and args.command == "compare":
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _main()
