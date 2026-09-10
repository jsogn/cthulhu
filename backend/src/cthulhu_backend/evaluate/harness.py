"""攻击评估闭环：嵌入 → 攻击 → 提取 → BER/PSNR/SSIM。"""

from __future__ import annotations

import os
import tempfile

import numpy as np

from cthulhu_backend import parallel, samples, services
from cthulhu_backend.attacks import dct as dct_attacks
from cthulhu_backend.attacks import geometric, spatial, temporal
from cthulhu_backend.evaluate import metrics
from cthulhu_backend.media import ffmpeg
from cthulhu_backend.similarity import embedding
from cthulhu_backend.transform import audio as audio_transform
from cthulhu_backend.transform import shots
from cthulhu_backend.transform import video as video_transform
from cthulhu_backend.watermark import common, detect, dwt, echo, lsb, qim, qim_rep, ss

VIDEO_EMBEDDERS = {
    "lsb": (lsb.embed, lsb.extract),
    "ss": (ss.embed, ss.extract),
    "qim": (qim.embed, qim.extract),
    "qim-rep": (qim_rep.embed, qim_rep.extract),
    "dwt": (dwt.embed, dwt.extract),
}

VIDEO_ATTACKS = {
    "median": spatial.median,
    "gaussian": spatial.gaussian,
    "wiener": spatial.wiener_denoise,
    "requant": spatial.requant_pixels,
    "requant-dct": lambda frames: np.stack([dct_attacks.requant_dct(f) for f in frames]),
    "lsb-randomize": spatial.randomize_lsb,
    "geometric": geometric.crop_rotate_rescale,
    "temporal": temporal.drop_duplicate,
}

AUDIO_ATTACKS = {
    "speed": lambda sig, rng: temporal.speed_change_audio(sig, 0.99),
    "noise": lambda sig, rng: np.clip(sig + 0.01 * rng.standard_normal(len(sig)), -1, 1),
}


def run_video_harness(
    method: str,
    frames: np.ndarray,
    bits: list[int],
    attacks: list[str],
    seed: int = 0,
    workers: int | None = None,
    **kwargs: object,
) -> dict:
    embed_fn, extract_fn = VIDEO_EMBEDDERS[method]
    seeded = method in {"lsb", "ss", "dwt"}
    watermarked = np.stack(
        [
            embed_fn(frame, bits, seed=seed, **kwargs) if seeded else embed_fn(frame, bits, **kwargs)
            for frame in frames
        ]
    )
    def _attack_one(attack: str) -> tuple[str, dict]:
        attacked = VIDEO_ATTACKS[attack](watermarked)
        out_bits = []
        for frame in attacked:
            out_bits.extend(
                extract_fn(frame, len(bits), seed=seed, **kwargs)
                if seeded
                else extract_fn(frame, len(bits), **kwargs)
            )
        ref = (common.SYNC + bits) * len(frames)
        return attack, {
            "ber": metrics.ber(ref, out_bits),
            "psnr_db": metrics.psnr(frames, attacked),
            "ssim": metrics.ssim(frames, attacked),
        }

    results = dict(parallel.map_items(_attack_one, attacks, workers))
    return {"method": method, "attacks": results}


def run_audio_harness(
    signal: np.ndarray,
    bits: list[int],
    attacks: list[str],
    sample_rate: int = 16000,
    seed: int = 0,
    segment: float = 0.05,
    workers: int | None = None,
) -> dict:
    watermarked = echo.embed(signal, bits, sample_rate, segment=segment)

    def _attack_one(attack: str) -> tuple[str, dict]:
        rng = np.random.default_rng(seed)
        attacked = AUDIO_ATTACKS[attack](watermarked, rng)
        out_bits = echo.extract(attacked, len(bits), sample_rate, segment=segment)
        return attack, {
            "ber": metrics.ber(common.SYNC + bits, out_bits),
            "psnr_db": metrics.psnr(signal, attacked),
        }

    results = dict(parallel.map_items(_attack_one, attacks, workers))
    return {"method": "echo", "attacks": results}


def run_detection_video_harness(
    methods: list[str],
    frames: np.ndarray,
    bits: list[int],
    attacks: list[str],
    seed: int = 0,
    workers: int | None = None,
    **kwargs: object,
) -> dict:
    """盲检测评估：干净基线 vs 水印 vs 攻击后，各方案置信度对比。"""
    clean_scores = detect.video_scores(frames)
    report: dict = {"clean": clean_scores}

    def _method_one(method: str) -> tuple[str, dict]:
        embed_fn = VIDEO_EMBEDDERS[method][0]
        seeded = method in {"lsb", "ss", "dwt"}
        watermarked = np.stack(
            [
                embed_fn(frame, bits, seed=seed, **kwargs)
                if seeded
                else embed_fn(frame, bits, **kwargs)
                for frame in frames
            ]
        )
        watermarked_scores = detect.video_scores(watermarked)
        attacked_scores = {
            attack: detect.video_scores(VIDEO_ATTACKS[attack](watermarked))[method]
            for attack in attacks
        }
        return method, {
            "watermarked": watermarked_scores[method],
            "attacked": attacked_scores,
        }

    for method, entry in parallel.map_items(_method_one, methods, workers):
        report[method] = entry
    return report


def run_detection_audio_harness(
    signal: np.ndarray,
    bits: list[int],
    attacks: list[str],
    sample_rate: int = 16000,
    seed: int = 0,
    segment: float = 0.25,
    workers: int | None = None,
) -> dict:
    """音频盲检测评估：干净 vs 回声水印 vs 攻击后置信度。"""
    clean_score = detect.audio_scores(signal, sample_rate)["echo"]
    watermarked = echo.embed(signal, bits, sample_rate, segment=segment)
    watermarked_score = detect.audio_scores(watermarked, sample_rate)["echo"]
    def _attack_one(attack: str) -> tuple[str, float]:
        rng = np.random.default_rng(seed)
        attacked = AUDIO_ATTACKS[attack](watermarked, rng)
        return attack, detect.audio_scores(attacked, sample_rate)["echo"]

    attacked_scores = dict(parallel.map_items(_attack_one, attacks, workers))
    return {
        "clean": clean_score,
        "watermarked": watermarked_score,
        "attacked": attacked_scores,
    }


def _codec_roundtrip(frames: np.ndarray, fps: float, crf: int) -> np.ndarray:
    """模拟真实分发链路的编码-解码往返。"""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "roundtrip.mp4")
        ffmpeg.encode_video(frames, path, fps=fps, crf=crf)
        decoded, _ = ffmpeg.decode_video(path)
        return decoded


def _stats(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": round(float(arr.mean()), 4),
        "std": round(float(arr.std()), 4),
        "values": [round(float(v), 4) for v in arr],
    }


def run_compressed_detection_baseline(
    methods: list[str],
    width: int,
    height: int,
    frames_n: int,
    payload_bits: int,
    seeds: list[int],
    attacks: list[str],
    fps: float = 30.0,
    crf: int = 23,
    workers: int | None = None,
) -> dict:
    """压缩域差分基线：干净 / 水印 / 攻击后均经编码往返后检测。

    目的：量化「有损压缩对像素域检测器的污染」与「水印在真实链路上的
    生存力」，为差分判定提供可复现阈值依据。
    """
    report: dict = {"config": {
        "methods": methods,
        "resolution": f"{width}x{height}",
        "frames": frames_n,
        "payload_bits": payload_bits,
        "seeds": seeds,
        "attacks": attacks,
        "fps": fps,
        "crf": crf,
    }, "methods": {}}
    def _unit(item: tuple[str, int]) -> tuple[str, int, float, float, dict[str, float]]:
        method, seed = item
        embed_fn = VIDEO_EMBEDDERS[method][0]
        clean = samples.make_video_frames(frames_n, width, height, seed=seed)
        bits = common.payload_bits(seed + 1, payload_bits)
        clean_coded = _codec_roundtrip(clean, fps, crf)
        clean_score = detect.video_scores(clean_coded)[method]
        if method == "ss":
            watermarked = np.stack(
                [embed_fn(frame, bits, seed=seed, alpha=0.03) for frame in clean],
            )
        elif method in {"lsb", "dwt"}:
            watermarked = np.stack(
                [embed_fn(frame, bits, seed=seed) for frame in clean],
            )
        else:
            watermarked = np.stack([embed_fn(frame, bits) for frame in clean])
        watermarked_coded = _codec_roundtrip(watermarked, fps, crf)
        watermarked_score = detect.video_scores(watermarked_coded)[method]
        attacked_scores = {
            attack: detect.video_scores(_codec_roundtrip(VIDEO_ATTACKS[attack](watermarked_coded), fps, crf))[method]
            for attack in attacks
        }
        return method, seed, clean_score, watermarked_score, attacked_scores

    rows = parallel.map_items(
        _unit, [(method, seed) for method in methods for seed in seeds], workers
    )
    clean_values: dict[str, list[float]] = {method: [] for method in methods}
    watermarked_values: dict[str, list[float]] = {method: [] for method in methods}
    attacked_values: dict[str, dict[str, list[float]]] = {
        method: {attack: [] for attack in attacks} for method in methods
    }
    for method, _, clean_score, watermarked_score, attacked_scores in rows:
        clean_values[method].append(clean_score)
        watermarked_values[method].append(watermarked_score)
        for attack in attacks:
            attacked_values[method][attack].append(attacked_scores[attack])

    for method in methods:
        watermarked_stats = _stats(watermarked_values[method])
        clean_stats = _stats(clean_values[method])
        diff_mean = round(watermarked_stats["mean"] - clean_stats["mean"], 4)
        attacked_report: dict[str, dict] = {}
        for attack, values in attacked_values[method].items():
            attacked_stats = _stats(values)
            attacked_diff = attacked_stats["mean"] - clean_stats["mean"]
            residual_ratio = (
                round(attacked_diff / diff_mean, 4) if abs(diff_mean) > 1e-9 else None
            )
            attacked_report[attack] = {
                **attacked_stats,
                "diff_mean": round(attacked_diff, 4),
                "residual_ratio": residual_ratio,
            }
        report["methods"][method] = {
            "clean": clean_stats,
            "watermarked": watermarked_stats,
            "diff_mean": diff_mean,
            "attacked": attacked_report,
        }
    return report


def run_desensitize_harness(
    frames: np.ndarray,
    configs: dict[str, dict],
    seed: int = 0,
    workers: int | None = None,
) -> dict:
    """内容脱敏评估：各参数组合对内容/运动相似度与画质的影响。

    psnr_db 为未时间对齐的帧序差，变速/重排会造成帧错位而拉低分数，
    仅作同一口径下的横向对比，不代表逐帧视觉画质。
    """
    def _config_one(item: tuple[str, dict]) -> tuple[str, dict]:
        name, config = item
        rng = np.random.default_rng(seed)
        out = frames
        if config.get("reorder"):
            out = video_transform.reorder_shots(out, shots.detect_cuts(out), rng)
        speed = config.get("speed", 1.0)
        if speed != 1.0:
            out = video_transform.retime(out, speed)
        recrop = config.get("recrop", 0.0)
        if recrop > 0:
            out = video_transform.recrop(out, recrop)
        if config.get("regrade"):
            out = video_transform.regrade(
                out,
                rng,
                strength=config.get("regrade_strength", 0.03),
                brightness=config.get("regrade_brightness", 0.02),
            )
        if config.get("sharpen"):
            out = video_transform.sharpen(out, amount=0.25, radius=1.2, threshold=0.01)
        if config.get("denoise"):
            out = spatial.wiener_denoise(out, size=5)
        length = min(len(frames), len(out))
        similarity = embedding.similarity_report(frames[:length], out[:length])
        return name, {
            "frames": len(out),
            "content_cosine": round(similarity["content_cosine"], 4),
            "motion_cosine": round(similarity["motion_cosine"], 4),
            "temporal_consistency": round(metrics.adjacent_cosine(out), 4),
            "order_disruption": round(metrics.order_disruption(frames, out), 4),
            "aligned_psnr_db": round(metrics.temporal_aligned_psnr(frames, out), 2),
            "psnr_db": round(metrics.psnr(frames[:length], out[:length]), 2),
        }

    report = dict(parallel.map_items(_config_one, list(configs.items()), workers))
    return report


def _extract_ber(frames: np.ndarray, extract_fn, bits: list[int], segment: bool = False) -> float:
    if segment:
        return metrics.ber(common.SYNC + bits, extract_fn(frames, len(bits)))
    out_bits: list[int] = []
    for frame in frames:
        out_bits.extend(extract_fn(frame))
    reference = (common.SYNC + bits) * len(frames)
    return metrics.ber(reference, out_bits)


def run_cleanse_matrix(
    frames: np.ndarray,
    variants: dict[str, dict],
    levels: dict[str, dict],
    bits: list[int] | None = None,
    seed: int = 0,
    fps: float = 30.0,
    crf: int = 23,
    workers: int | None = None,
) -> dict:
    """通杀验收矩阵：各水印变体经编码与各档清洗后的误码率与画质。

    variants 每个条目需提供 embed(frame, bits) 与 extract(frame)。
    levels 每个条目为 run_desensitize 的关键字参数（不含 path/output）。
    variant 可选 content_kind 标签（simple/textured/natural），报告按内容
    分层输出 FNR（BER<0.6 视为失效，0.5 为随机猜测）。返回结构向后兼容：
    levels 仍是 BER 浮点数，画质与分层汇总放在新增的 quality / summary 键。
    """
    bits = bits or common.payload_bits(seed + 1, 64)
    report: dict = {}
    with tempfile.TemporaryDirectory() as tmp:
        prepared = []
        for variant_name, variant in variants.items():
            segment = variant.get("segment", False)
            if segment:
                watermarked = variant["embed"](frames, bits)
            else:
                watermarked = np.stack([variant["embed"](frame, bits) for frame in frames])
            wm_path = os.path.join(tmp, f"{variant_name}-wm.mp4")
            ffmpeg.encode_video(watermarked, wm_path, fps=fps, crf=crf)
            coded, _ = ffmpeg.decode_video(wm_path)
            row: dict = {
                "encoded_ber": round(_extract_ber(coded, variant["extract"], bits, segment), 4),
                "content_kind": variant.get("content_kind", "unknown"),
                "levels": {},
                "quality": {},
            }
            report[variant_name] = row
            for level_name, params in levels.items():
                prepared.append(
                    (variant_name, level_name, variant, params, wm_path, coded)
                )

        def _level_one(
            item: tuple[str, str, dict, dict, str, np.ndarray],
        ) -> tuple[str, str, float, float, float]:
            variant_name, level_name, variant, params, wm_path, reference = item
            out_path = os.path.join(tmp, f"{variant_name}-{level_name}.mp4")
            services.run_desensitize(wm_path, out_path, **params)
            cleaned, _ = ffmpeg.decode_video(out_path)
            aligned = min(len(reference), len(cleaned))
            return (
                variant_name,
                level_name,
                round(
                    _extract_ber(cleaned, variant["extract"], bits, variant.get("segment", False)),
                    4,
                ),
                round(metrics.psnr(reference[:aligned], cleaned[:aligned]), 2),
                round(metrics.ssim(reference[:aligned], cleaned[:aligned]), 4),
            )

        for variant_name, level_name, ber, psnr, ssim in parallel.map_items(
            _level_one, prepared, workers
        ):
            report[variant_name]["levels"][level_name] = ber
            report[variant_name]["quality"][level_name] = {
                "psnr_db": psnr,
                "ssim": ssim,
            }

    # FNR 分层汇总：命中阈值 0.6（与报告一致），按内容复杂度分列。
    all_hits = [0, 0]
    by_kind: dict[str, list[int]] = {}
    for row in report.values():
        if not isinstance(row, dict) or "levels" not in row:
            continue
        kind = row["content_kind"]
        by_kind.setdefault(kind, [0, 0])
        for ber in row["levels"].values():
            failed = 1 if ber < 0.6 else 0
            all_hits[0] += failed
            all_hits[1] += 1
            by_kind[kind][0] += failed
            by_kind[kind][1] += 1
    report["summary"] = {
        "fnr": round(all_hits[0] / max(1, all_hits[1]), 4),
        "fnr_by_content": {
            kind: round(hits[0] / max(1, hits[1]), 4) for kind, hits in by_kind.items()
        },
    }
    return report


def run_audio_cleanse(
    signal: np.ndarray,
    bits: list[int],
    sample_rate: int = 16000,
    segment: float = 0.05,
    speed_factor: float = 0.97,
    seed: int = 0,
) -> dict:
    """音频通杀验收：回声水印经变速重混后的误码率。"""
    watermarked = echo.embed(signal, bits, sample_rate, segment=segment)
    rng = np.random.default_rng(seed)
    remixed = audio_transform.remix(watermarked, sample_rate, rng, speed_factor=speed_factor)

    def extract_ber(target: np.ndarray) -> float:
        out = echo.extract(target, len(bits), sample_rate, segment=segment)
        return metrics.ber(common.SYNC + bits, out)

    return {
        "before_ber": round(extract_ber(watermarked), 4),
        "after_ber": round(extract_ber(remixed), 4),
    }
