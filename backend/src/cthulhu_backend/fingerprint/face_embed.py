"""人脸识别代理：ArcFace ONNX 逼近平台的人脸 embedding 判定。

输入为 RGB 人脸裁剪（NHWC 112×112，归一化 (x-127.5)/128），输出 512 维
embedding；用于验证人脸抗AI扰动的破坏力，并作为 face_perturb 的 SPSA 靶。
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

_DEFAULT_MODEL = str(
    Path(__file__).resolve().parents[3] / "models" / "arcface.onnx"
)

_session = None
_model_path = os.environ.get("CTHULHU_FACE_MODEL") or _DEFAULT_MODEL


def available() -> bool:
    if not Path(_model_path).is_file():
        return False
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    return True


def _get_session():
    global _session
    if _session is None:
        import onnxruntime as ort

        _session = ort.InferenceSession(_model_path, providers=["CPUExecutionProvider"])
    return _session


def embed_crop(crop: np.ndarray) -> np.ndarray:
    """人脸裁剪 → 512 维 L2 归一 embedding（crop 为 uint8 RGB HWC）。"""
    session = _get_session()
    rgb = np.asarray(crop, dtype=np.uint8)
    if rgb.ndim == 2:
        rgb = np.repeat(rgb[..., None], 3, axis=-1)
    resized = cv2.resize(rgb, (112, 112), interpolation=cv2.INTER_LINEAR)
    tensor = ((resized.astype(np.float32) - 127.5) / 128.0)[None, ...]
    vector = session.run(None, {session.get_inputs()[0].name: tensor})[0][0].astype(np.float32)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else vector


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b)) / denom if denom > 0 else 0.0
