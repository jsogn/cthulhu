"""内置预置 × 基准库水印的通杀矩阵复测（真实 720p 素材前几秒）。

旧文档的 0.44~0.50 对应「重排+变速+裁剪+调光」旧管线；当前预置已不含
重排/变速/裁剪，本脚本用 services.run_desensitize 走生产路径重测。

用法：
  CALIBRATE=1 uv run --project backend python backend/scripts/bench_preset_matrix.py
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import time

import numpy as np

from cthulhu_backend import db, services
from cthulhu_backend.evaluate import metrics
from cthulhu_backend.media import ffmpeg
from cthulhu_backend.watermark import common, dft, dwt, qim, ss

DEFAULT_CLIPS = [
    "/Users/alone/Downloads/暗水印测试/AD-下载8.mp4",
    "/Users/alone/Downloads/暗水印测试/好心邻居：熬百碗粥只为抓替身-B3.mp4",
    "/Users/alone/Downloads/暗水印测试/好心邻居：熬百碗粥只为抓替身-C6.mp4",
]
SECONDS = 2.0
BITS = common.payload_bits(1, 64)
REF = common.SYNC + BITS


def decode(path: str, frames: int) -> np.ndarray:
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", path],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    width, height = (int(x) for x in probe.split(","))
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-frames:v", str(frames),
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        check=True, capture_output=True,
    ).stdout
    return np.frombuffer(raw, np.uint8).reshape(frames, height, width, 3)


def payload_to_options(payload: dict) -> dict:
    """TemplatePayload → run_desensitize 参数，对齐前端 makeCleanOptions 语义。"""
    opts: dict = {
        "reorder": False,
        "speed": 1.0,
        "recrop": 0.0,
        "perturb": 0.15,
        "regrade": payload.get("regradeOn", False),
        "audio_remix": payload.get("audioRemix", False),
        "echo_defeat": payload.get("echoDefeat", False),
        "audio_strong": payload.get("audioStrong", False),
        "sharpness": payload.get("sharpness", False),
        "color_restore": payload.get("colorRestore", False),
        "denoise": payload.get("denoise", False),
        "rotate": payload.get("rotate", 0.0),
        "requant": payload.get("requant", 0),
        "noise": payload.get("noise", 0.0),
        "skip_vmaf": True,
        "seed": 0,
    }
    if payload.get("regradeOn"):
        opts["hsv_jitter"] = 6
        opts["chroma_levels"] = 32
    if payload.get("dctStep", 0) > 0:
        opts["dct_step"] = 12.0
        opts["anti_reembed"] = True
    if payload.get("hashAttack"):
        mode = payload.get("hashMode", "phash")
        if mode == "phash":
            opts["phash_attack"] = True
        elif mode == "dhash":
            opts["dhash_attack"] = True
        else:
            opts["multi_hash_attack"] = True
        opts["phash_epsilon"] = payload.get("hashEpsilon", 0.08)
    simple = [
        ("temporalSub", "temporal_sub"), ("fftPhase", "fft_phase"),
        ("dwtDetail", "dwt_detail"),
        ("facePerturb", "face_perturb"), ("lpcAttack", "lpc_attack"),
        ("copyAttack", "copy_attack"),
    ]
    for source, target in simple:
        if payload.get(source, 0):
            opts[target] = payload[source]
    if payload.get("nativeTemporal"):
        opts["native_temporal"] = True
    if payload.get("spoof"):
        opts["spoof"] = True
    if payload.get("qualityProtect"):
        opts["quality_protect"] = True
        opts["psnr_target"] = payload.get("psnrTarget", 38.0)
        opts["ssim_target"] = payload.get("ssimTarget", 0.94)
    return opts


VARIANT_FACTORY = {
    "ss-a0.25": (lambda f: ss.embed(f, BITS, seed=0, alpha=0.25),
                 lambda f: ss.extract(f, 64, seed=0)),
    "qim-d20": (lambda f: qim.embed(f, BITS, delta=20.0),
                lambda f: qim.extract(f, 64, delta=20.0)),
    "qim-d40": (lambda f: qim.embed(f, BITS, delta=40.0),
                lambda f: qim.extract(f, 64, delta=40.0)),
    "dwt": (lambda f: dwt.embed(f, BITS, seed=0),
            lambda f: dwt.extract(f, 64, seed=0)),
    "dft": (lambda f: dft.embed(f, BITS, seed=0, alpha=4.0),
            lambda f: dft.extract(f, 64, seed=0)),
}


def ber(frames: np.ndarray, extract) -> float:
    bits = [bit for frame in frames for bit in extract(frame)]
    return metrics.ber(REF * len(frames), bits)


def luma(rgb: np.ndarray) -> np.ndarray:
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def main() -> None:
    count = int(SECONDS * 30)
    clips = os.environ.get("CLIPS", ",".join(DEFAULT_CLIPS)).split(",")
    preset_filter = set(filter(None, os.environ.get("PRESET_FILTER", "").split(",")))
    variant_filter = set(filter(None, os.environ.get("VARIANT_FILTER", "").split(",")))
    gate_off = os.environ.get("GATE_OFF") == "1"
    temporal_beta = os.environ.get("TEMPORAL_BETA")
    beta_sweep = os.environ.get("BETA_SWEEP") == "1"
    rotate_override = os.environ.get("ROTATE")
    gate_psnr_override = os.environ.get("GATE_PSNR")
    no_transcode = os.environ.get("NO_TRANSCODE") == "1"
    force_transcode = os.environ.get("TRANSCODE") == "1"
    no_native = os.environ.get("NO_NATIVE") == "1"
    hash_eps_override = os.environ.get("HASH_EPS")
    requant_override = os.environ.get("REQUANT")
    denoise_override = os.environ.get("DENOISE")
    presets = [(p["name"], payload_to_options(p["payload"])) for p in db.PRESET_TEMPLATES]
    print(f"帧数 {count}，素材 {len(clips)} 段，预置 {len(presets)} 档", flush=True)
    with tempfile.TemporaryDirectory() as tmp:
        for clip_path in clips:
            clean_u8 = decode(clip_path, count)
            clean = clean_u8.astype(np.float32) / 255.0
            clip_name = os.path.basename(clip_path)[:18]
            for variant_name, (embed, extract) in VARIANT_FACTORY.items():
                if variant_filter and variant_name not in variant_filter:
                    continue
                if beta_sweep and variant_name != "ss-a0.25":
                    continue
                base_luma = luma(clean)
                marked_luma = np.stack([embed(f) for f in base_luma])
                watermarked = np.clip(
                    clean + (marked_luma - base_luma)[..., None], 0.0, 1.0
                ).astype(np.float32)
                wm_path = os.path.join(tmp, f"{clip_name}-{variant_name}-wm.mp4")
                ffmpeg.encode_video(watermarked, wm_path, fps=30, crf=23)
                coded, _ = ffmpeg.decode_video(wm_path)
                row = f"{clip_name:20s} {variant_name:9s} 编码后={ber(coded, extract):.3f}"
                if beta_sweep:
                    base_opts = next(
                        opts for name, opts in presets if name == "均衡 · 推荐"
                    )
                    for beta in (0.5, 0.6, 0.7, 0.8, 1.0):
                        opts = {**base_opts, "temporal_sub": beta}
                        start = time.perf_counter()
                        out_path = os.path.join(tmp, f"{clip_name}-ss-beta{beta}.mp4")
                        services.run_desensitize(wm_path, out_path, **opts)
                        cleaned, _ = ffmpeg.decode_video(out_path)
                        row += f" | β{beta}={ber(cleaned, extract):.3f}"
                        print(f"{clip_name} β{beta} 完成 {time.perf_counter()-start:.0f}s", flush=True)
                else:
                    for preset_name, opts in presets:
                        if preset_filter and preset_name not in preset_filter:
                            continue
                        if rotate_override:
                            opts["rotate"] = float(rotate_override)
                        if gate_psnr_override:
                            opts["psnr_target"] = float(gate_psnr_override)
                        if no_transcode:
                            opts.pop("transcode_chain", None)
                        if force_transcode:
                            opts["transcode_chain"] = True
                        if no_native:
                            opts["native_temporal"] = False
                        if hash_eps_override:
                            opts["phash_epsilon"] = float(hash_eps_override)
                        if requant_override:
                            opts["requant"] = int(requant_override)
                        if denoise_override:
                            opts["denoise"] = denoise_override == "1"
                        if temporal_beta:
                            opts["temporal_sub"] = float(temporal_beta)
                        if gate_off:
                            opts.pop("quality_protect", None)
                        start = time.perf_counter()
                        out_path = os.path.join(
                            tmp, f"{clip_name}-{variant_name}-{preset_name}.mp4"
                        )
                        services.run_desensitize(wm_path, out_path, **opts)
                        cleaned, _ = ffmpeg.decode_video(out_path)
                        elapsed = time.perf_counter() - start
                        row += f" | {preset_name}={ber(cleaned, extract):.3f}({elapsed:.0f}s)"
                        print(
                            f"{clip_name} {variant_name} {preset_name} 完成 {elapsed:.0f}s",
                            flush=True,
                        )
                print(row, flush=True)


if __name__ == "__main__":
    main()
