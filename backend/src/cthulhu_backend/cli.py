"""研究 CLI：样本生成与攻击评估闭环（比 GUI 更早落地，支撑对照实验）。"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Annotated

import numpy as np
import typer
from scipy.io import wavfile

from cthulhu_backend import samples
from cthulhu_backend.bitstream import analyze as bitstream_analyze
from cthulhu_backend.evaluate import (
    attack_matrix,
    calibrate,
    content_matrix,
    dedup_harness,
    harness,
)
from cthulhu_backend.media import container, ffmpeg
from cthulhu_backend.sample_prep import diff as diff_module
from cthulhu_backend.similarity import embedding
from cthulhu_backend.transform import shots
from cthulhu_backend.transform import video as video_transform
from cthulhu_backend.watermark import common, detect, dwt, lsb, qim, qim_rep, ss, temporal

app = typer.Typer(help="暗水印研究工具：样本生成与对抗评估")


@app.command()
def gen(
    frames: int = typer.Option(16, help="视频帧数"),
    width: int = typer.Option(320),
    height: int = typer.Option(240),
    seconds: float = typer.Option(2.0, help="音频时长（秒）"),
    seed: int = typer.Option(0),
    out_prefix: str = typer.Option("data/sample", help="输出前缀"),
) -> None:
    """生成合成干净样本（帧 numpy 数组 + 音频 WAV）。"""
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    video = samples.make_video_frames(frames, width, height, seed)
    np.save(f"{out_prefix}_video.npy", video)
    audio = samples.make_audio(seconds, seed=seed)
    wavfile.write(f"{out_prefix}_audio.wav", 16000, (audio * 32767).astype(np.int16))
    typer.echo(f"已生成：{out_prefix}_video.npy / {out_prefix}_audio.wav")


@app.command()
def harness_video(
    methods: str = typer.Option("lsb,ss,qim", help="逗号分隔：lsb,ss,qim"),
    attacks: str = typer.Option(
        "median,gaussian,wiener,requant,requant-dct,lsb-randomize,geometric,temporal",
        help="逗号分隔的攻击名",
    ),
    frames: int = typer.Option(16),
    width: int = typer.Option(320),
    height: int = typer.Option(240),
    payload_bits: int = typer.Option(64),
    seed: int = typer.Option(0),
    out: str = typer.Option("data/report-video.json"),
) -> None:
    """视频攻击评估矩阵：每种水印 × 每种攻击输出 BER / PSNR / SSIM。"""
    clean = samples.make_video_frames(frames, width, height, seed)
    bits = common.payload_bits(seed + 1, payload_bits)
    report = {}
    for method in methods.split(","):
        method = method.strip()
        result = harness.run_video_harness(method, clean, bits, [a.strip() for a in attacks.split(",")], seed)
        report[method] = result["attacks"]
        for attack, r in result["attacks"].items():
            typer.echo(
                f"{method:>4} × {attack:<14} BER={r['ber']:.3f}  PSNR={r['psnr_db']:.1f}dB  SSIM={r['ssim']:.4f}"
            )
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    typer.echo(f"报告已写入：{out}")


@app.command()
def harness_audio(
    attacks: str = typer.Option("speed,noise"),
    seconds: float = typer.Option(3.0),
    payload_bits: int = typer.Option(32),
    seed: int = typer.Option(0),
    out: str = typer.Option("data/report-audio.json"),
) -> None:
    """音频（回声隐藏）攻击评估。"""
    clean = samples.make_audio(seconds, seed=seed)
    bits = common.payload_bits(seed + 1, payload_bits)
    result = harness.run_audio_harness(clean, bits, [a.strip() for a in attacks.split(",")], seed=seed)
    for attack, r in result["attacks"].items():
        typer.echo(f"echo × {attack:<6} BER={r['ber']:.3f}  PSNR={r['psnr_db']:.1f}dB")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    typer.echo(f"报告已写入：{out}")


@app.command()
def probe(file: str = typer.Argument(..., help="视频文件路径")) -> None:
    """探测视频元信息、容器异常 box 与 SEI 数量。"""
    info = ffmpeg.video_info(file)
    scan = container.scan_mp4(file)
    sei = container.count_sei(file)
    report = {
        "file": file,
        "container": info["format"],
        "codec": info["codec"],
        "resolution": f"{info['width']}x{info['height']}",
        "fps": info["fps"],
        "duration": info["duration"],
        "udta_boxes": scan["udta_boxes"],
        "meta_boxes": scan["meta_boxes"],
        "has_xmp": scan["has_xmp"],
        "n_trak": scan["n_trak"],
        "suspicious": scan["suspicious"],
        "sei_count": sei,
    }
    typer.echo(json.dumps(report, ensure_ascii=False, indent=2))


@app.command()
def dedup_fingerprint(file: str = typer.Argument(..., help="视频文件路径")) -> None:
    """输出判重代理指纹：图像哈希族（逐帧位图）。"""
    typer.echo(json.dumps(dedup_harness.frame_hashes(file), ensure_ascii=False, indent=2))


@app.command()
def dedup_compare(
    original: str = typer.Argument(..., help="原片路径"),
    candidate: str = typer.Argument(..., help="待比较视频路径"),
    out_dir: str = typer.Option("docs/baselines", help="报告输出目录"),
    tag: str = typer.Option("compare", help="报告文件名后缀"),
) -> None:
    """代理判重栈对比：图像哈希族 + 镜头结构 + 音频梅尔谱指纹。"""
    report = dedup_harness.compare(original, candidate)
    path = dedup_harness.save_report(report, out_dir, tag)
    typer.echo(json.dumps(report, ensure_ascii=False, indent=2))
    typer.echo(f"报告已写入：{path}")


@app.command("attack-matrix")
def attack_matrix_run(
    clip: str = typer.Argument(..., help="短片段输入（≤60s）"),
    out_dir: str = typer.Option("/tmp/cthulhu-attack-matrix", help="工作目录"),
    seed: int = typer.Option(0),
    reuse: bool = typer.Option(False, help="复用已生成的变体文件"),
) -> None:
    """对抗原语效果矩阵：判重风险 × VMAF 双目标打分。"""
    report = attack_matrix.run_matrix(clip, out_dir, seed=seed, reuse=reuse)
    typer.echo(json.dumps(report, ensure_ascii=False, indent=2))


@app.command("calibrate-presets")
def calibrate_presets(
    clips: Annotated[list[str], typer.Argument(help="多个短片段输入（≤60s）")],
    out_dir: str = typer.Option("/tmp/cthulhu-calibration", help="工作目录"),
    seeds: str = typer.Option("0,1", help="逗号分隔的种子列表"),
) -> None:
    """多片段 × 多种子标定三档对抗参数，输出均值与波动。"""
    seed_list = tuple(int(part.strip()) for part in seeds.split(",") if part.strip())
    report = calibrate.run_calibration(clips, out_dir, seeds=seed_list)
    typer.echo(json.dumps(report["summary"], ensure_ascii=False, indent=2))


@app.command("content-matrix")
def content_matrix_run(
    clip: str = typer.Argument(..., help="短片段输入（≤60s）"),
    out_dir: str = typer.Option("/tmp/cthulhu-content-matrix", help="报告输出目录"),
) -> None:
    """内容级变换矩阵：CLIP 距离 × 画质 × 叙事连续性。"""
    report = content_matrix.evaluate(clip, out_dir)
    typer.echo(json.dumps(report, ensure_ascii=False, indent=2))


@app.command()
def sample_diff(
    clean: str = typer.Argument(..., help="干净基准视频"),
    watermarked: str = typer.Argument(..., help="带水印视频"),
    out_dir: str = typer.Option("data/samples", help="输出目录"),
    name: str = typer.Option("sample", help="样本名"),
) -> None:
    """样本制备：解码 → 相位相关对齐 → 差分报告与 DCT 热图。"""
    clean_frames, clean_info = ffmpeg.decode_video(clean)
    wm_frames, _ = ffmpeg.decode_video(watermarked)
    if clean_frames.shape[1:] != wm_frames.shape[1:]:
        typer.echo("错误：两段视频分辨率不一致", err=True)
        raise typer.Exit(code=1)
    report = diff_module.build_report(clean_frames, wm_frames, out_dir, name)
    report["clean_frames"] = len(clean_frames)
    report["resolution"] = f"{clean_info['width']}x{clean_info['height']}"
    typer.echo(json.dumps(report, ensure_ascii=False, indent=2))
    typer.echo(f"输出目录：{out_dir}")


@app.command()
def bitstream(
    file: str = typer.Argument(..., help="视频文件路径"),
    reference: str = typer.Option(None, "--reference", help="干净基准视频（差分判定）"),
) -> None:
    """压缩域（码流层）检测：QP 图、码量分配、GOP/SEI 与启发式评分。"""
    report = bitstream_analyze.analyze(file, reference)
    typer.echo(json.dumps(report, ensure_ascii=False, indent=2))


@app.command()
def detect_watermark(
    file: str = typer.Argument(..., help="待检视频文件路径"),
) -> None:
    """盲检测置信度：对视频各方案输出 0~1 的启发式分数。"""
    frames, info = ffmpeg.decode_video(file)
    scores = detect.video_scores(frames)
    typer.echo(json.dumps(
        {
            "file": file,
            "resolution": f"{info['width']}x{info['height']}",
            "frames": len(frames),
            "scores": scores,
            "note": (
                "像素域启发式置信度：有损压缩视频会抬高 LSB/QIM 基线，"
                "建议与干净同源基准做差分判定（bitstream --reference）"
            ),
        },
        ensure_ascii=False,
        indent=2,
    ))


@app.command()
def detect_harness(
    methods: str = typer.Option("lsb,ss,qim", help="逗号分隔：lsb,ss,qim"),
    attacks: str = typer.Option("median,gaussian,wiener,requant,requant-dct,geometric,temporal"),
    frames: int = typer.Option(16),
    width: int = typer.Option(320),
    height: int = typer.Option(240),
    payload_bits: int = typer.Option(64),
    seed: int = typer.Option(0),
    out: str = typer.Option("data/report-detect.json"),
) -> None:
    """盲检测评估矩阵：干净 vs 水印 vs 攻击后的置信度对比。"""
    clean = samples.make_video_frames(frames, width, height, seed)
    bits = common.payload_bits(seed + 1, payload_bits)
    method_list = [m.strip() for m in methods.split(",")]
    attack_list = [a.strip() for a in attacks.split(",")]
    report = harness.run_detection_video_harness(
        method_list, clean, bits, attack_list, seed=seed,
    )
    for method in method_list:
        clean_score = report["clean"][method]
        wm_score = report[method]["watermarked"]
        typer.echo(
            f"{method:>4} 干净={clean_score:.3f} → 水印={wm_score:.3f} → 攻击后={report[method]['attacked']}"
        )
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    typer.echo(f"报告已写入：{out}")


@app.command()
def detect_baseline(
    methods: str = typer.Option("lsb,ss,qim", help="逗号分隔：lsb,ss,qim"),
    attacks: str = typer.Option("requant,wiener,geometric"),
    seeds: str = typer.Option("0,1,2", help="逗号分隔的随机种子列表"),
    frames: int = typer.Option(16),
    width: int = typer.Option(320),
    height: int = typer.Option(240),
    payload_bits: int = typer.Option(64),
    crf: int = typer.Option(23, help="编码质量（越小越接近无损）"),
    out: str = typer.Option("data/report-compressed-baseline.json"),
) -> None:
    """压缩域差分基线：各方案经 H.264 编码往返后的检出置信度统计。"""
    method_list = [m.strip() for m in methods.split(",")]
    attack_list = [a.strip() for a in attacks.split(",")]
    seed_list = [int(s.strip()) for s in seeds.split(",")]
    report = harness.run_compressed_detection_baseline(
        method_list, width, height, frames, payload_bits, seed_list,
        attack_list, crf=crf,
    )
    for method, entry in report["methods"].items():
        typer.echo(
            f"{method:>4} 干净={entry['clean']['mean']:.3f}±{entry['clean']['std']:.3f} "
            f"→ 水印={entry['watermarked']['mean']:.3f}±{entry['watermarked']['std']:.3f} "
            f"(差分 {entry['diff_mean']:+.3f})"
        )
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    typer.echo(f"报告已写入：{out}")


@app.command()
def desensitize_matrix(
    segments: int = typer.Option(6, help="带硬切镜头的段数"),
    frames_per_segment: int = typer.Option(10),
    width: int = typer.Option(320),
    height: int = typer.Option(240),
    seed: int = typer.Option(0),
    out: str = typer.Option("data/report-desensitize.json"),
) -> None:
    """内容脱敏评估矩阵：GUI 三档与单变量消融的内容/运动相似度。"""
    clean = samples.make_cut_video(segments, frames_per_segment, width, height, seed=seed)
    configs = {
        "轻度档": {"reorder": True, "speed": 0.9625, "recrop": 0.0263, "regrade": True, "regrade_strength": 0.06},
        "平衡档": {"reorder": True, "speed": 0.955, "recrop": 0.03, "regrade": True, "regrade_strength": 0.07},
        "深度档": {"reorder": True, "speed": 0.94, "recrop": 0.0375, "regrade": True, "regrade_strength": 0.09},
        "仅重排": {"reorder": True},
        "仅变速": {"speed": 0.955},
        "仅裁剪": {"recrop": 0.03},
        "仅调光": {"regrade": True, "regrade_strength": 0.07},
    }
    report = harness.run_desensitize_harness(clean, configs, seed=seed)
    for name, metrics in report.items():
        typer.echo(
            f"{name:<6} content={metrics['content_cosine']:.4f} "
            f"motion={metrics['motion_cosine']:.3f} "
            f"乱序度={metrics['order_disruption']:.3f} "
            f"对齐PSNR={metrics['aligned_psnr_db']:.1f}dB"
        )
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    typer.echo(f"报告已写入：{out}")


@app.command()
def cleanse_matrix(
    segments: int = typer.Option(4),
    frames_per_segment: int = typer.Option(15),
    width: int = typer.Option(320),
    height: int = typer.Option(240),
    seed: int = typer.Option(0),
    out: str = typer.Option("data/report-cleanse.json"),
) -> None:
    """通杀验收矩阵：各水印方案/强度经各档清洗后的误码率（越接近 0.5 越干净）。"""
    clean = samples.make_cut_video(segments, frames_per_segment, width, height, seed=seed)
    bits = common.payload_bits(seed + 1, 64)
    variants = {
        "ss-a0.08": {
            "embed": lambda f, b: ss.embed(f, b, seed=0, alpha=0.08),
            "extract": lambda f: ss.extract(f, 64, seed=0),
        },
        "ss-a0.25": {
            "embed": lambda f, b: ss.embed(f, b, seed=0, alpha=0.25),
            "extract": lambda f: ss.extract(f, 64, seed=0),
        },
        "qim-d20": {
            "embed": lambda f, b: qim.embed(f, b, delta=20),
            "extract": lambda f: qim.extract(f, 64, delta=20),
        },
        "qim-d40": {
            "embed": lambda f, b: qim.embed(f, b, delta=40),
            "extract": lambda f: qim.extract(f, 64, delta=40),
        },
        "qim-rep3": {
            "embed": lambda f, b: qim_rep.embed(f, b, delta=20, rep=3),
            "extract": lambda f: qim_rep.extract(f, 64, delta=20, rep=3),
        },
        "qim-rep7": {
            "embed": lambda f, b: qim_rep.embed(f, b, delta=20, rep=7),
            "extract": lambda f: qim_rep.extract(f, 64, delta=20, rep=7),
        },
        "dwt": {
            "embed": lambda f, b: dwt.embed(f, b, seed=0),
            "extract": lambda f: dwt.extract(f, 64, seed=0),
        },
        "temporal": {
            "embed": lambda f, b: temporal.embed_frames(f, b),
            "extract": lambda f, n: temporal.extract_frames(f, n),
            "segment": True,
        },
        "lsb": {
            "embed": lambda f, b: lsb.embed(f, b, seed=0),
            "extract": lambda f: lsb.extract(f, 64, seed=0),
        },
    }
    levels = {
        "轻度": {
            "reorder": True, "speed": 0.9625, "recrop": 0.0263, "regrade": True,
            "perturb": 0.15, "audio_remix": False, "denoise": False, "seed": 0,
        },
        "平衡": {
            "reorder": True, "speed": 0.955, "recrop": 0.03, "regrade": True,
            "perturb": 0.2, "audio_remix": False, "denoise": True, "seed": 0,
        },
        "深度": {
            "reorder": True, "speed": 0.94, "recrop": 0.0375, "regrade": True,
            "perturb": 0.3, "audio_remix": False, "denoise": True, "seed": 0,
        },
    }
    report = harness.run_cleanse_matrix(clean, variants, levels, bits=bits, seed=seed)
    for name, entry in report.items():
        level_text = " ".join(f"{k}={v:.3f}" for k, v in entry["levels"].items())
        typer.echo(f"{name:<10} 编码后={entry['encoded_ber']:.3f} | {level_text}")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    typer.echo(f"报告已写入：{out}")


@app.command()
def benchmark(
    path: str = typer.Option(None, help="待测视频；缺省时生成 240 帧合成视频"),
) -> None:
    """软/硬件编解码耗时对比（macOS 使用 VideoToolbox）。"""
    target = path
    if not target:
        from cthulhu_backend import samples

        video = samples.make_video_frames(240, 640, 360, seed=200, motion_speed=1.2)
        target = os.path.join(tempfile.gettempdir(), "cthulhu-bench-src.mp4")
        ffmpeg.encode_video(video, target, fps=30)
    typer.echo(json.dumps(ffmpeg.benchmark(target), ensure_ascii=False, indent=2))


@app.command()
def similarity(a: str = typer.Argument(..., help="参考视频"), b: str = typer.Argument(..., help="待比较视频")) -> None:
    """量化两段视频的内容/运动/哈希/画质相似度（自建 embedding）。"""
    frames_a, _ = ffmpeg.decode_video(a)
    frames_b, _ = ffmpeg.decode_video(b)
    report = embedding.similarity_report(frames_a, frames_b)
    typer.echo(json.dumps(report, ensure_ascii=False, indent=2))


@app.command()
def desensitize(
    input: str = typer.Argument(..., help="输入视频"),
    output: str = typer.Argument(..., help="输出视频"),
    reorder: bool = typer.Option(True, help="分镜重排"),
    speed: float = typer.Option(1.0, help="变速倍率（>1 加速）"),
    recrop: float = typer.Option(0.0, help="重新构图裁剪比例（0~0.2）"),
    regrade: bool = typer.Option(True, help="逐帧重调光"),
    banner: str = typer.Option("", help="叠加贴纸文字（可选）"),
    seed: int = typer.Option(0),
) -> None:
    """内容脱敏流水线：变换画面并量化与源内容的相似度下降。"""
    frames, info = ffmpeg.decode_video(input)
    original = frames
    rng = np.random.default_rng(seed)
    if reorder:
        frames = video_transform.reorder_shots(frames, shots.detect_cuts(frames), rng)
    if speed != 1.0:
        frames = video_transform.retime(frames, speed)
    if recrop > 0:
        frames = video_transform.recrop(frames, recrop)
    if regrade:
        frames = video_transform.regrade(frames, rng)
    if banner:
        frames = video_transform.overlay_banner(frames, banner, seed)
    ffmpeg.encode_video(frames, output, fps=info["fps"])

    report = {
        "input": input,
        "output": output,
        "frames": len(original),
        "similarity_before": embedding.similarity_report(original, original),
        "similarity_after": embedding.similarity_report(original, frames),
        "vmaf": ffmpeg.vmaf_score(output, input),
        "note": "画面层变换；音频重混待接入音画联合管线",
    }
    typer.echo(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    app()
