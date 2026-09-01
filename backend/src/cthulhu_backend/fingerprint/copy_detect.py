"""拷贝检测代理：DINOv2-small 视觉描述子逼近平台判重的 embedding 层。

CLIP 是对齐文本的语义 embedding，搬运判重用的是拷贝检测类描述子
（SSCD/ISC 族）。DINOv2 是自监督视觉匹配特征，比 CLIP 更接近该判据；
SSCD 官方只提供 torchscript，需 torch 运行时不引入，本模块先用
DINOv2-small（onnx-community，fp16 权重）作为本地可验证的近似代理。
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_SIZE = 224
_DIM = 384

_DEFAULT_MODEL = str(
    Path(__file__).resolve().parents[3] / "models" / "dinov2-small.onnx"
)

_session = None
_model_path = os.environ.get("CTHULHU_COPY_MODEL") or _DEFAULT_MODEL


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


def _preprocess(frame: np.ndarray) -> np.ndarray:
    array = np.clip(np.asarray(frame, dtype=np.float32), 0.0, 1.0)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    image = Image.fromarray((array * 255.0).round().astype(np.uint8), mode="RGB")
    resized = np.asarray(image.resize((_SIZE, _SIZE), Image.BILINEAR), dtype=np.float32) / 255.0
    normalized = (resized - _MEAN) / _STD
    return normalized.transpose(2, 0, 1)[None, ...]


def embed_frame(frame: np.ndarray) -> np.ndarray:
    """单帧拷贝检测描述子（DINOv2 CLS，384 维，L2 归一）。"""
    session = _get_session()
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: _preprocess(frame)})[0]
    vector = output[0, 0, :].astype(np.float32)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else vector


def video_embedding(frames: np.ndarray, max_frames: int = 40) -> np.ndarray:
    array = np.asarray(frames)
    if len(array) > max_frames:
        indices = np.linspace(0, len(array) - 1, max_frames).astype(int)
        array = array[indices]
    pooled = np.mean([embed_frame(frame) for frame in array], axis=0)
    norm = float(np.linalg.norm(pooled))
    return pooled / norm if norm > 0 else pooled


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b)) / denom if denom > 0 else 0.0
