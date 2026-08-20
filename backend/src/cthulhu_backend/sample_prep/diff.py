"""对齐后的差分分析：空域残差 + DCT 中频能量差，定位水印活跃区域。"""

from __future__ import annotations

import json
import os

import numpy as np
from scipy.fftpack import dctn

from cthulhu_backend.sample_prep.align import align_videos
from cthulhu_backend.watermark.qim import MID_BAND


def _dct_midband_heatmap(frame: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """逐 8×8 块的中频系数能量差热图。"""
    h, w = frame.shape
    frame8, ref8 = frame * 255.0, reference * 255.0
    padded_f = np.pad(frame8, ((0, -h % 8), (0, -w % 8)), mode="edge")
    padded_r = np.pad(ref8, ((0, -h % 8), (0, -w % 8)), mode="edge")
    blocks_f = padded_f.reshape(padded_f.shape[0] // 8, 8, padded_f.shape[1] // 8, 8)
    blocks_r = padded_r.reshape(padded_r.shape[0] // 8, 8, padded_r.shape[1] // 8, 8)
    cf = dctn(blocks_f, axes=(1, 3), norm="ortho")
    cr = dctn(blocks_r, axes=(1, 3), norm="ortho")
    bf = cf.transpose(0, 2, 1, 3).reshape(cf.shape[0], cf.shape[2], 64)
    br = cr.transpose(0, 2, 1, 3).reshape(cr.shape[0], cr.shape[2], 64)
    return np.mean(np.abs(bf[:, :, MID_BAND] - br[:, :, MID_BAND]), axis=2)


def build_report(
    clean_frames: np.ndarray,
    watermarked_frames: np.ndarray,
    out_dir: str,
    name: str = "sample",
) -> dict:
    """对齐 → 差分 → 输出 JSON 报告、热图 PNG 与对齐后帧。"""
    os.makedirs(out_dir, exist_ok=True)
    ref, mov, shift = align_videos(clean_frames, watermarked_frames)
    residual = np.abs(mov - ref)
    per_frame = float(np.mean(residual))
    heatmap = _dct_midband_heatmap(np.median(mov, axis=0), np.median(ref, axis=0))

    np.save(os.path.join(out_dir, f"{name}_aligned_clean.npy"), ref)
    np.save(os.path.join(out_dir, f"{name}_aligned_watermarked.npy"), mov)
    np.save(os.path.join(out_dir, f"{name}_dct_heatmap.npy"), heatmap)
    try:
        from PIL import Image

        scaled = np.clip(heatmap / max(float(heatmap.max()), 1e-9) * 255, 0, 255).astype(np.uint8)
        Image.fromarray(scaled).save(os.path.join(out_dir, f"{name}_dct_heatmap.png"))
    except ImportError:
        pass

    report = {
        "shift": {"dy": int(shift[0]), "dx": int(shift[1])},
        "mean_abs_diff": per_frame,
        "dct_midband_diff_mean": float(np.mean(heatmap)),
        "dct_midband_diff_max": float(np.max(heatmap)),
        "aligned_frames": len(ref),
    }
    with open(os.path.join(out_dir, f"{name}_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    return report
