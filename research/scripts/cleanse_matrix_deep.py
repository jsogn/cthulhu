#!/usr/bin/env python3
"""公开深度水印方案的清洗验收矩阵（研究用途）。

把后端新接入的净化原语（cthulhu_backend.transform.purify）作为清洗手段，
对 TrustMark / MBRS / WAM / VideoSeal 四个公开方案跑内容分层验收：

- 指标：BA（0.5 为随机）、FNR（BA<0.6 即失效）、PSNR/SSIM（vs 带水印帧）；
- 内容：simple（合成）/ natural（公开许可自然图或 Ken Burns 运镜）；
- 档位：扩散净化 strength 0.15 / 0.25 / 0.35；视频额外验收净化后过真实
  H.264 CRF23 的组合链路。

边界：公开预训练模型、合成/公开许可素材、本地离线环境，不接触任何平台
线上系统。复用现有迁移脚本的加载器，不重复实现模型 I/O。
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "research" / "scripts"
MBRS_REPO = ROOT / "research" / "vendor" / "MBRS"
WAM_REPO = ROOT / "research" / "vendor" / "watermark-anything"
VSEAL_REPO = ROOT / "research" / "vendor" / "videoseal"
for extra in (SCRIPTS, MBRS_REPO, WAM_REPO, VSEAL_REPO, ROOT / "backend" / "src"):
    sys.path.insert(0, str(extra))

import eot_core as eot
from baseline_trustmark import bit_accuracy, random_bits, synthetic_image
from cthulhu_backend.transform import purify

FFMPEG = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"


def device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _to_tensor(img: Image.Image, dev: str) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(dev)


def _to_pil(tensor: torch.Tensor) -> Image.Image:
    arr = (
        (tensor.clamp(0.0, 1.0).squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0)
        .round()
        .astype(np.uint8)
    )
    return Image.fromarray(arr)


def _resize(img: Image.Image, size: int) -> Image.Image:
    return img.resize((size, size), Image.BICUBIC)


def build_trustmark(rng: np.random.Generator):
    from trustmark import TrustMark

    tm = TrustMark(use_ECC=False, verbose=False, model_type="Q", loadRemover=False)
    payload = random_bits(rng)

    def embed(img: Image.Image) -> Image.Image:
        return tm.encode(_resize(img, 512), payload, MODE="binary")

    def ba(img: Image.Image) -> float:
        pred, _, _ = tm.decode(_resize(img, 512), MODE="binary")
        return bit_accuracy(pred, payload)

    return "trustmark", embed, ba, 512, "image"


def build_mbrs(rng: np.random.Generator, dev: str):
    from network.Encoder_MP_Decoder import EncoderDecoder

    model = EncoderDecoder(256, 256, 256, noise_layers=[])
    state = torch.load(
        MBRS_REPO / "results" / "MBRS_256_256" / "EC_42.pth", map_location="cpu"
    )
    model.load_state_dict(state, strict=False)
    model.eval().to(dev)
    message = torch.from_numpy(
        np.array([int(x) for x in random_bits(rng, 256)], dtype=np.float32)
    ).unsqueeze(0).to(dev)

    def embed(img: Image.Image) -> Image.Image:
        cover = _to_tensor(_resize(img, 256), dev)
        with torch.no_grad():
            encoded = model.encoder(cover, message).clamp(0.0, 1.0)
        return _to_pil(encoded)

    def ba(img: Image.Image) -> float:
        with torch.no_grad():
            logits = model.decoder(_to_tensor(_resize(img, 256), dev))
        pred = (logits > 0.5).float()
        return float((pred == message).float().mean().item())

    return "mbrs", embed, ba, 512, "image"


def build_wam(rng: np.random.Generator, dev: str):
    os.chdir(WAM_REPO)
    from notebooks.inference_utils import (
        default_transform,
        load_model_from_checkpoint,
        unnormalize_img,
    )
    from watermark_anything.data.metrics import msg_predict_inference

    wam = (
        load_model_from_checkpoint(
            str(WAM_REPO / "checkpoints" / "params.json"),
            str(WAM_REPO / "checkpoints" / "wam_mit.pth"),
        )
        .to(dev)
        .eval()
    )
    torch.manual_seed(int(rng.integers(0, 2**31)))
    message = torch.randint(0, 2, (1, 32)).float().to(dev)

    def to_pil(tensor: torch.Tensor) -> Image.Image:
        arr = unnormalize_img(tensor).clamp(0.0, 1.0).squeeze(0).permute(1, 2, 0).cpu().numpy()
        return Image.fromarray((arr * 255.0).round().astype(np.uint8))

    def embed(img: Image.Image) -> Image.Image:
        cover = default_transform(img).unsqueeze(0).to(dev)
        with torch.no_grad():
            watermarked = wam.embed(cover, message)["imgs_w"]
        return to_pil(watermarked)

    def ba(img: Image.Image) -> float:
        attacked = default_transform(img).unsqueeze(0).to(dev)
        with torch.no_grad():
            preds = wam.detect(attacked)["preds"].cpu()
        mask = torch.sigmoid(preds[:, 0, :, :])
        bits = preds[:, 1:, :, :]
        pred = msg_predict_inference(bits, mask)
        return float((pred == message.cpu()).float().mean().item())

    return "wam", embed, ba, 256, "image"


def build_videoseal(dev: str):
    os.chdir(VSEAL_REPO)
    import videoseal

    model = videoseal.load("videoseal").to(dev).eval()

    def decode_logits(tensor: torch.Tensor) -> torch.Tensor:
        """逐帧 logits（可微调用：EOT-PGD 需回传梯度；评测处外层自行 no_grad）。"""
        import torch.nn.functional as F

        resized = F.interpolate(
            tensor,
            size=(model.img_size, model.img_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        return model.detector(resized)[:, 1:]

    def embed_video(frames_np: np.ndarray) -> tuple[np.ndarray, torch.Tensor]:
        tensor = (
            torch.from_numpy(frames_np.astype(np.float32) / 255.0)
            .permute(0, 3, 1, 2)
            .to(dev)
        )
        with torch.no_grad():
            out = model.embed(tensor, is_video=True)
        watermarked = (
            (out["imgs_w"].clamp(0.0, 1.0).permute(0, 2, 3, 1).cpu().numpy() * 255.0)
            .round()
            .astype(np.uint8)
        )
        return watermarked, out["msgs"][0:1].float().to(dev)

    def ba(frames_np: np.ndarray, message: torch.Tensor) -> tuple[float, float]:
        tensor = (
            torch.from_numpy(frames_np.astype(np.float32) / 255.0)
            .permute(0, 3, 1, 2)
            .to(dev)
        )
        with torch.no_grad():
            preds = model.detect(tensor, is_video=True)["preds"]
        bits = (preds[:, 1:] > 0).float()
        per_frame = float((bits == message).float().mean().item())
        aggregated = (bits.mean(dim=0, keepdim=True) > 0).float()
        agg = float((aggregated == message).float().mean().item())
        return agg, per_frame

    eot.register_scheme(
        "videoseal",
        embed=lambda frames_np, bits: embed_video(frames_np)[0],
        decode=decode_logits,
    )
    return "videoseal", embed_video, ba, 512, "video", decode_logits


def natural_still() -> Image.Image:
    return Image.open(VSEAL_REPO / "assets" / "imgs" / "1.jpg").convert("RGB")


def simple_video(frames: int) -> np.ndarray:
    size = 512
    yy, xx = np.mgrid[0:size, 0:size]
    out = []
    for t in range(frames):
        arr = np.zeros((size, size, 3), dtype=np.uint8)
        arr[..., 0] = (60 + 120 * xx / size).astype(np.uint8)
        arr[..., 1] = (50 + 100 * yy / size).astype(np.uint8)
        arr[..., 2] = 140
        x0 = 30 + t * (size - 180) // max(frames - 1, 1)
        arr[size // 3 : size // 3 + 150, x0 : x0 + 120] = [230, 90, 60]
        out.append(arr)
    return np.stack(out)


def natural_video(frames: int, size: int = 512) -> np.ndarray:
    """Ken Burns 运镜（内联实现，避免研究脚本导入期的 chdir 副作用）。"""
    image = Image.open(VSEAL_REPO / "assets" / "imgs" / "1.jpg").convert("RGB")
    width, height = image.size
    out = []
    for t in range(frames):
        frac = t / max(frames - 1, 1)
        crop_w = int(width * (0.55 - 0.15 * frac))
        crop_h = int(height * (0.55 - 0.15 * frac))
        x0 = int((width - crop_w) * frac)
        y0 = int((height - crop_h) * (0.5 - 0.5 * frac))
        crop = image.crop((x0, y0, x0 + crop_w, y0 + crop_h)).resize(
            (size, size), Image.BICUBIC
        )
        out.append(np.asarray(crop))
    return np.stack(out)


def h264_crf23(frames_np: np.ndarray) -> np.ndarray:
    """真实 H.264 CRF23 链路验收：净化扰动能否穿过真实编码。"""
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        src = tmpdir / "src.mkv"
        dst = tmpdir / "out.mp4"
        subprocess.run(
            [
                FFMPEG, "-y", "-v", "error", "-f", "rawvideo",
                "-pix_fmt", "rgb24", "-s", f"{frames_np.shape[2]}x{frames_np.shape[1]}",
                "-r", "30", "-i", "-", "-c:v", "ffv1", str(src),
            ],
            input=frames_np.tobytes(),
            capture_output=True,
            check=True,
        )
        subprocess.run(
            [
                FFMPEG, "-y", "-v", "error", "-i", str(src),
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", str(dst),
            ],
            capture_output=True,
            check=True,
        )
        probe = subprocess.run(
            [FFMPEG, "-v", "error", "-i", str(dst), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
            capture_output=True,
            check=True,
        )
        width, height = frames_np.shape[2], frames_np.shape[1]
        raw = np.frombuffer(probe.stdout, dtype=np.uint8)
        raw = raw[: len(raw) // (width * height * 3) * (width * height * 3)]
        return raw.reshape(-1, height, width, 3)


def _quality(reference: np.ndarray, attacked: np.ndarray) -> tuple[float, float]:
    length = min(len(reference), len(attacked))
    ref = reference[:length].astype(np.float64)
    atk = attacked[:length].astype(np.float64)
    psnr = psnr_metric(ref, atk, data_range=255)
    if length > 1:
        ssim = float(
            np.mean(
                [
                    ssim_metric(ref[i], atk[i], channel_axis=2, data_range=255)
                    for i in range(length)
                ]
            )
        )
    else:
        ssim = ssim_metric(ref[0], atk[0], channel_axis=2, data_range=255)
    return float(psnr), float(ssim)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schemes", nargs="+", default=["trustmark", "mbrs", "wam", "videoseal"])
    parser.add_argument("--images", type=int, default=2)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--strengths", default="0.15,0.25,0.35")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--eot", action="store_true", help="VideoSeal 额外跑白盒 EOT-PGD")
    parser.add_argument("--eot-eps", type=float, default=4.0 / 255.0)
    parser.add_argument("--eot-steps", type=int, default=50)
    parser.add_argument("--out", type=Path, default=ROOT / "research" / "output" / "cleanse_matrix_deep")
    args = parser.parse_args()
    if not args.out.is_absolute():
        args.out = ROOT / args.out
    strengths = [float(x) for x in args.strengths.split(",")]
    dev = device()
    rng = np.random.default_rng(20260909)

    builders = {
        "trustmark": lambda: build_trustmark(rng),
        "mbrs": lambda: build_mbrs(rng, dev),
        "wam": lambda: build_wam(rng, dev),
        "videoseal": lambda: build_videoseal(dev),
    }
    rows: list[dict[str, object]] = []
    scheme_fnr: dict[str, list[int]] = {}
    for scheme_name in args.schemes:
        if scheme_name not in builders:
            print(f"跳过未知方案：{scheme_name}")
            continue
        try:
            built = builders[scheme_name]()
            name, embed, decode, purify_size, kind = built[:5]
            decode_logits = built[5] if len(built) > 5 else None
        except Exception as exc:  # noqa: BLE001 - 缺权重/仓库时单方案跳过
            print(f"[{scheme_name}] 加载失败，跳过：{exc}")
            continue
        scheme_fnr.setdefault(name, [0, 0])
        if kind == "image":
            covers = {
                "simple": [synthetic_image(i) for i in range(args.images)],
                "natural": [natural_still()],
            }
            for content_kind, images in covers.items():
                for image_id, cover in enumerate(images):
                    watermarked = embed(cover)
                    clean = decode(watermarked)
                    wm_arr = np.asarray(_resize(watermarked, purify_size))
                    for strength in strengths:
                        arr = purify.purify_frames(
                            wm_arr[None],
                            strength=strength,
                            steps=args.steps,
                            seed=20260909 + image_id,
                        )[0]
                        attacked = Image.fromarray(arr).resize(
                            (watermarked.width, watermarked.height), Image.BICUBIC
                        )
                        ba = decode(attacked)
                        psnr, ssim = _quality(
                            np.asarray(watermarked)[None], np.asarray(attacked)[None]
                        )
                        scheme_fnr[name][0] += int(ba < 0.6)
                        scheme_fnr[name][1] += 1
                        rows.append(
                            {
                                "scheme": name,
                                "content": content_kind,
                                "sample": image_id,
                                "level": f"purify_{strength:.2f}",
                                "clean_ba": round(clean, 4),
                                "ba": round(ba, 4),
                                "psnr": round(psnr, 2),
                                "ssim": round(ssim, 4),
                            }
                        )
        else:
            videos = {
                "simple": simple_video(args.frames),
                "natural": natural_video(args.frames),
            }
            for content_kind, cover in videos.items():
                watermarked, message = embed(cover)
                clean_agg, _clean_pf = decode(watermarked, message)
                for strength in strengths:
                    arr = purify.purify_frames(
                        watermarked,
                        strength=strength,
                        steps=args.steps,
                        seed=20260909,
                    )
                    agg, pf = decode(arr, message)
                    psnr, ssim = _quality(watermarked, arr)
                    scheme_fnr[name][0] += int(agg < 0.6)
                    scheme_fnr[name][1] += 1
                    rows.append(
                        {
                            "scheme": name,
                            "content": content_kind,
                            "sample": 0,
                            "level": f"purify_{strength:.2f}",
                            "clean_ba": round(clean_agg, 4),
                            "ba": round(agg, 4),
                            "per_frame_ba": round(pf, 4),
                            "psnr": round(psnr, 2),
                            "ssim": round(ssim, 4),
                        }
                    )
                # 组合链路：净化 + 真实 H.264 CRF23（报告关键验收项）。
                arr = purify.purify_frames(
                    watermarked, strength=0.15, steps=args.steps, seed=20260909
                )
                coded = h264_crf23(arr)
                agg, pf = decode(coded, message)
                psnr, ssim = _quality(watermarked, coded)
                rows.append(
                    {
                        "scheme": name,
                        "content": content_kind,
                        "sample": 0,
                        "level": "purify_0.15+CRF23",
                        "clean_ba": round(clean_agg, 4),
                        "ba": round(agg, 4),
                        "per_frame_ba": round(pf, 4),
                        "psnr": round(psnr, 2),
                        "ssim": round(ssim, 4),
                    }
                )
                if args.eot and decode_logits is not None:
                    adv = eot.run(
                        watermarked,
                        decode_logits,
                        eot.EotOptions(
                            eps=args.eot_eps,
                            steps=args.eot_steps,
                            chunk=16,
                            mode="modulated",
                            eot=True,
                        ),
                        seed=20260909,
                    )
                    coded = h264_crf23(adv)
                    agg, pf = decode(coded, message)
                    psnr, ssim = _quality(watermarked, coded)
                    scheme_fnr[name][0] += int(agg < 0.6)
                    scheme_fnr[name][1] += 1
                    rows.append(
                        {
                            "scheme": name,
                            "content": content_kind,
                            "sample": 0,
                            "level": "eot_modulated+CRF23",
                            "clean_ba": round(clean_agg, 4),
                            "ba": round(agg, 4),
                            "per_frame_ba": round(pf, 4),
                            "psnr": round(psnr, 2),
                            "ssim": round(ssim, 4),
                        }
                    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.out.with_suffix(".csv")
    if rows:
        with csv_path.open("w", newline="") as handle:
            fieldnames = list(dict.fromkeys(key for row in rows for key in row))
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    md_path = args.out.with_suffix(".md")
    with md_path.open("w") as handle:
        handle.write("# 公开深度水印方案清洗验收矩阵（研究）\n\n")
        handle.write(f"- 档位：扩散净化 strength {strengths}；视频含净化 0.15+CRF23 组合\n")
        handle.write("- BA=比特准确率（0.5 为随机）；FNR=BA<0.6 的样本比例\n\n")
        handle.write("| 方案 | FNR |\n| --- | ---: |\n")
        for name, (hits, total) in scheme_fnr.items():
            handle.write(f"| {name} | {hits}/{total} = {hits / max(1, total):.3f} |\n")
    print(f"CSV 已写入：{csv_path}")
    print(f"报告已写入：{md_path}")
    for name, (hits, total) in scheme_fnr.items():
        print(f"{name:<10} FNR={hits}/{total} = {hits / max(1, total):.3f}")


if __name__ == "__main__":
    main()
