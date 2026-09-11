#!/usr/bin/env python3
"""白盒定向清洗 worker（研究运行时侧）。

由后端主进程以子进程方式调用：读取输入视频，加载已知公开方案的 decoder，
做 EOT-PGD / mask PGD / message PGD，输出带原音轨的无损 FFV1 中间视频，
并通过 stdout 的 JSON 行汇报进度与自一致性指标。

边界：只支持公开权重方案（VideoSeal / PixelSeal / WAM）；仅研究学习与授权测试使用。
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent


def _find_backend_src() -> Path:
    for candidate in (
        _SCRIPT_DIR.parents[1] / "backend" / "src",
        _SCRIPT_DIR.parents[2] / "backend" / "src",
        Path.cwd() / "backend" / "src",
    ):
        if (candidate / "cthulhu_backend").is_dir():
            return candidate
    return _SCRIPT_DIR


SCRIPTS = _SCRIPT_DIR
BACKEND_SRC = _find_backend_src()


def _vendor_root() -> Path:
    raw = os.environ.get("CTHULHU_TARGETED_VENDOR")
    if raw:
        return Path(raw).expanduser()
    if getattr(sys, "frozen", False):
        base = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
        return base / "vendor"
    for candidate in (
        _SCRIPT_DIR.parent / "vendor",
        _SCRIPT_DIR.parents[1] / "research" / "vendor",
    ):
        if candidate.is_dir():
            return candidate
    return _SCRIPT_DIR.parent / "vendor"


VENDOR_ROOT = _vendor_root()
VSEAL_REPO = VENDOR_ROOT / "videoseal"
WAM_REPO = VENDOR_ROOT / "watermark-anything"
for extra in (SCRIPTS, BACKEND_SRC):
    sys.path.insert(0, str(extra))

_STOP = False


def _on_stop(signum, frame) -> None:
    global _STOP
    _STOP = True


def emit(**payload) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def _device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.is_file():
            return path
    return paths[0]


def read_video(path: Path, max_frames: int) -> tuple[np.ndarray, float, int, int]:
    """读取全片并校验白盒模式的帧数上限（解码口径见 _harness.read_video）。"""
    from cthulhu_backend.media import ffmpeg as backend_ffmpeg
    from _harness import read_video as harness_read_video

    info = backend_ffmpeg.video_info(str(path))
    width, height, fps = int(info["width"]), int(info["height"]), float(info["fps"])
    frames = harness_read_video(path)
    if len(frames) > max_frames:
        raise RuntimeError(
            f"白盒模式仅支持 ≤{max_frames} 帧的片段（当前 {len(frames)} 帧）；"
            f"约 {max_frames / max(fps, 1):.0f} 秒（按当前帧率）；长视频请使用黑盒净化"
        )
    return frames, fps, width, height


def write_video(
    frames: np.ndarray,
    path: Path,
    fps: float,
    audio_source: Path,
) -> None:
    """写 FFV1 无损中间视频，并保留原音轨供主流程复用。"""
    from _harness import write_video as harness_write_video

    harness_write_video(frames, path, fps=fps, audio_source=audio_source)


def _quality(reference: np.ndarray, attacked: np.ndarray) -> tuple[float, float]:
    from _harness import quality as harness_quality

    return harness_quality(reference, attacked)


def _self_ba(bits_before: np.ndarray, bits_after: np.ndarray) -> float:
    """与攻击前自身的比特一致率；0.5 表示比特已被打成随机。"""
    count = min(len(bits_before), len(bits_after))
    if count == 0:
        return 1.0
    return float(np.mean(bits_before[:count] == bits_after[:count]))


def run_videoseal(
    frames: np.ndarray,
    *,
    scheme: str = "videoseal",
    eps: float,
    steps: int,
    chunk: int,
    eot: bool,
    seed: int,
    progress,
) -> tuple[np.ndarray, dict]:
    import eot_core
    import torch
    import torch.nn.functional as F

    os.chdir(VSEAL_REPO)
    sys.path.insert(0, str(VSEAL_REPO))
    import videoseal

    device = _device()
    model = videoseal.load(scheme).to(device).eval()

    def decode_logits(tensor: torch.Tensor) -> torch.Tensor:
        resized = F.interpolate(
            tensor,
            size=(model.img_size, model.img_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        return model.detector(resized)[:, 1:]

    tensor = torch.from_numpy(frames.astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(device)
    with torch.no_grad():
        before = decode_logits(tensor).detach().cpu().numpy()

    def on_step(step: int, total: int) -> None:
        if _STOP:
            raise InterruptedError("白盒攻击已取消")
        progress(0.1 + 0.8 * step / max(1, total), f"EOT-PGD {step}/{total}")

    attacked = eot_core.run(
        frames,
        decode_logits,
        eot_core.EotOptions(
            eps=eps,
            steps=steps,
            chunk=chunk,
            mode="modulated" if eot else "static",
            eot=eot,
        ),
        seed=seed,
        progress=on_step,
        should_stop=lambda: _STOP,
    )
    attacked_t = torch.from_numpy(attacked.astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(device)
    with torch.no_grad():
        after = decode_logits(attacked_t).detach().cpu().numpy()
    psnr, ssim = _quality(frames, attacked)
    metrics = {
        "scheme": scheme,
        "attack": "eot_pgd",
        "frames": len(frames),
        "self_ba": round(_self_ba(before > 0, after > 0), 4),
        "bit_change_rate": round(1.0 - _self_ba(before > 0, after > 0), 4),
        "logits_abs_before": round(float(np.abs(before).mean()), 4),
        "logits_abs_after": round(float(np.abs(after).mean()), 4),
        "psnr": round(psnr, 2),
        "ssim": round(ssim, 4),
    }
    return attacked, metrics


def run_wam(
    frames: np.ndarray,
    *,
    attack: str,
    eps: float,
    steps: int,
    seed: int,
    progress,
) -> tuple[np.ndarray, dict]:
    import torch
    import torch.nn.functional as F

    os.chdir(WAM_REPO)
    sys.path.insert(0, str(WAM_REPO))
    from notebooks.inference_utils import load_model_from_checkpoint
    from watermark_anything.data.transforms import normalize_img

    device = _device()
    model = (
        load_model_from_checkpoint(
            str(WAM_REPO / "checkpoints" / "params.json"),
            str(WAM_REPO / "checkpoints" / "wam_mit.pth"),
        )
        .to(device)
        .eval()
    )
    torch.manual_seed(seed)

    def decode(pixel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        small = F.interpolate(
            pixel, size=(256, 256), mode="bilinear", align_corners=False, antialias=True
        )
        preds = model.detect(normalize_img(small))["preds"]
        return preds[:, 0:1], preds[:, 1:]

    out: list[np.ndarray] = []
    mask_before: list[float] = []
    mask_after: list[float] = []
    bits_before: list[np.ndarray] = []
    bits_after: list[np.ndarray] = []
    previous_delta = None
    total = len(frames)
    for index, frame in enumerate(frames):
        if _STOP:
            raise InterruptedError("白盒攻击已取消")
        original = (
            torch.from_numpy(frame.astype(np.float32) / 255.0)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device)
        )
        with torch.no_grad():
            mask_logits, bit_logits = decode(original)
            mask_before.append(float(torch.sigmoid(mask_logits).mean().item()))
            bits_before.append((bit_logits > 0).cpu().numpy().reshape(-1))
        if previous_delta is None:
            delta = torch.zeros_like(original, requires_grad=True)
        else:
            delta = previous_delta.clone().detach().requires_grad_(True)
        alpha = eps / 5.0
        for step in range(steps):
            if _STOP:
                raise InterruptedError("白盒攻击已取消")
            adv = (original + delta).clamp(0.0, 1.0)
            mask_logits, bit_logits = decode(adv)
            mask_loss = F.binary_cross_entropy_with_logits(
                mask_logits, torch.zeros_like(mask_logits)
            )
            message_loss = bit_logits.pow(2).mean()
            if attack == "mask_pgd":
                loss = mask_loss
            elif attack == "message_pgd":
                loss = message_loss
            else:
                loss = mask_loss + message_loss
            grad = torch.autograd.grad(loss, delta)[0]
            delta = (delta - alpha * grad.sign()).detach().clamp(-eps, eps).requires_grad_(True)
            progress(
                0.1 + 0.8 * (index + (step + 1) / max(1, steps)) / max(1, total),
                f"WAM {attack} {index + 1}/{total} · step {step + 1}/{steps}",
            )
        attacked = (original + delta).clamp(0.0, 1.0)
        with torch.no_grad():
            mask_logits, bit_logits = decode(attacked)
            mask_after.append(float(torch.sigmoid(mask_logits).mean().item()))
            bits_after.append((bit_logits > 0).cpu().numpy().reshape(-1))
        out.append(
            (attacked.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255.0)
            .round()
            .astype(np.uint8)
        )
        previous_delta = delta.detach()
        if device == "mps":
            torch.mps.empty_cache()
    attacked_frames = np.stack(out)
    before = np.concatenate(bits_before) if bits_before else np.zeros(0)
    after = np.concatenate(bits_after) if bits_after else np.zeros(0)
    psnr, ssim = _quality(frames, attacked_frames)
    metrics = {
        "scheme": "wam",
        "attack": attack,
        "frames": int(total),
        "mask_score_before": round(float(np.mean(mask_before)), 4),
        "mask_score_after": round(float(np.mean(mask_after)), 4),
        "self_ba": round(_self_ba(before > 0, after > 0), 4),
        "bit_change_rate": round(1.0 - _self_ba(before > 0, after > 0), 4),
        "psnr": round(psnr, 2),
        "ssim": round(ssim, 4),
    }
    return attacked_frames, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="只检查方案/权重可用性并退出")
    parser.add_argument("--scheme", choices=["videoseal", "pixelseal", "wam"])
    parser.add_argument(
        "--attack",
        choices=["eot_pgd", "mask_pgd", "message_pgd", "both_pgd"],
    )
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--eps", type=float, default=4.0 / 255.0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--chunk", type=int, default=16)
    parser.add_argument("--eot", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=600)
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, _on_stop)
    signal.signal(signal.SIGINT, _on_stop)

    if args.check:
        import torch

        schemes = {
            "videoseal": {
                "vendor": str(VSEAL_REPO / "videoseal"),
                "weights": str(
                    _first_existing(
                        VSEAL_REPO / "ckpts" / "y_256b_img.pth",
                        VSEAL_REPO / "ckpts" / "videoseal_y_256b_img.pth",
                    )
                ),
                "attacks": ["eot_pgd"],
            },
            "pixelseal": {
                "vendor": str(VSEAL_REPO / "videoseal"),
                "weights": str(
                    _first_existing(
                        VSEAL_REPO / "ckpts" / "pixelseal_checkpoint.pth",
                        VSEAL_REPO / "ckpts" / "checkpoint.pth",
                    )
                ),
                "attacks": ["eot_pgd"],
            },
            "wam": {
                "vendor": str(WAM_REPO / "watermark_anything"),
                "weights": str(WAM_REPO / "checkpoints" / "wam_mit.pth"),
                "attacks": ["mask_pgd", "message_pgd", "both_pgd"],
            },
        }
        for item in schemes.values():
            item["available"] = Path(item["vendor"]).is_dir() and Path(item["weights"]).is_file()
        emit(type="check", ok=True, torch=torch.__version__, device=_device(), schemes=schemes)
        return

    if not (args.scheme and args.attack and args.input and args.output):
        parser.error("--scheme/--attack/--input/--output 为必填（--check 除外）")

    started = time.time()
    progress_seen = [0.0]

    def progress(fraction: float, note: str) -> None:
        fraction = max(0.0, min(1.0, fraction))
        if fraction - progress_seen[0] < 0.01 and fraction < 1.0:
            return
        progress_seen[0] = fraction
        emit(type="progress", percent=round(fraction * 100, 2), note=note)

    try:
        progress(0.02, "读取视频")
        frames, fps, _, _ = read_video(args.input, args.max_frames)
        progress(0.05, f"已读取 {len(frames)} 帧")
        if args.scheme in ("videoseal", "pixelseal"):
            attacked, metrics = run_videoseal(
                frames,
                scheme=args.scheme,
                eps=args.eps,
                steps=args.steps,
                chunk=args.chunk,
                eot=args.eot,
                seed=args.seed,
                progress=progress,
            )
        else:
            attacked, metrics = run_wam(
                frames,
                attack=args.attack,
                eps=args.eps,
                steps=args.steps,
                seed=args.seed,
                progress=progress,
            )
        progress(0.95, "写入无损中间视频")
        write_video(attacked, args.output, fps, args.input)
        metrics["seconds"] = round(time.time() - started, 1)
        progress(1.0, "白盒攻击完成")
        emit(type="result", metrics=metrics)
    except InterruptedError:
        emit(type="error", error="白盒攻击已取消")
        raise SystemExit(130) from None
    except Exception as exc:  # noqa: BLE001 - 统一回传给主进程
        traceback.print_exc()
        emit(type="error", error=str(exc))
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
