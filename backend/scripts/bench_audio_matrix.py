"""音频回声水印 × 各预置音频栈的通杀矩阵（真实 run_desensitize 音频路径）。

口径：合成音频嵌入回声水印 → 与色块视频 mux 成测试片 → 逐档走
run_desensitize（视频参数最小化，音频参数取预置真实值，echo_defeat 的
atempo 0.97 保调拉伸走真实 mux）→ 解码输出音轨提取 BER。
"""

from __future__ import annotations

import os
import subprocess
import tempfile

import numpy as np

from cthulhu_backend import db, samples, services
from cthulhu_backend.evaluate import metrics
from cthulhu_backend.media import ffmpeg
from cthulhu_backend.watermark import common, echo

BITS = common.payload_bits(1, 64)
REF = common.SYNC + BITS
SECONDS = 20.0


def ber(signal: np.ndarray) -> float:
    out = echo.extract(signal, len(BITS), 16000)
    return metrics.ber(REF, out)


def build_clip(watermarked: np.ndarray, path: str) -> None:
    wav = os.path.join(os.path.dirname(path), "wm.wav")
    payload = (np.clip(watermarked, -1, 1) * 32767).round().astype(np.int16).tobytes()
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "s16le", "-ar", "16000", "-ac", "1",
         "-i", "-", wav],
        input=payload, check=True,
    )
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "lavfi", "-i", f"color=size=320x240:rate=30:duration={SECONDS}",
         "-i", wav, "-map", "0:v:0", "-map", "1:a:0",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "30",
         "-c:a", "aac", "-b:a", "128k", "-shortest", path],
        check=True,
    )


def audio_opts(payload: dict) -> dict:
    return {
        "audio_remix": payload.get("audioRemix", False),
        "echo_defeat": payload.get("echoDefeat", False),
        "audio_strong": payload.get("audioStrong", False),
        "lpc_attack": payload.get("lpcAttack", 0.0),
    }


def main() -> None:
    signal = samples.make_audio(SECONDS, sample_rate=16000, seed=3)
    watermarked = echo.embed(signal, BITS, 16000)
    print(f"嵌入后 BER={ber(watermarked):.3f}", flush=True)
    with tempfile.TemporaryDirectory() as tmp:
        clip = os.path.join(tmp, "clip.mp4")
        build_clip(watermarked, clip)
        # 基线：音频原样透传（AAC 往返自身损耗）。
        base_out = os.path.join(tmp, "base.mp4")
        services.run_desensitize(
            clip, base_out, reorder=False, speed=1.0, regrade=False,
            sharpness=False, color_restore=False, denoise=False, rotate=0.0,
            requant=0, noise=0.0, perturb=0.0, quality_protect=False,
            skip_vmaf=True, seed=0,
        )
        base_audio, _ = ffmpeg.decode_audio(base_out)
        print(f"透传基线(AAC 往返) BER={ber(base_audio):.3f}", flush=True)
        for preset in db.PRESET_TEMPLATES:
            opts = {
                "reorder": False, "speed": 1.0, "regrade": False,
                "sharpness": False, "color_restore": False, "denoise": False,
                "rotate": 0.0, "requant": 0, "noise": 0.0, "perturb": 0.0,
                "quality_protect": False, "skip_vmaf": True, "seed": 0,
                **audio_opts(preset["payload"]),
            }
            out_path = os.path.join(tmp, f"{preset['name']}.mp4")
            services.run_desensitize(clip, out_path, **opts)
            audio, _ = ffmpeg.decode_audio(out_path)
            print(f"{preset['name']:14s} BER={ber(audio):.3f}", flush=True)


if __name__ == "__main__":
    main()
