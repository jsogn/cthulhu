"""预训练视觉模型代理：用 CLIP ViT-B/32 逼近平台的深度 embedding 判定。

手工特征（边缘直方图/关键点网格）低估 CNN 对几何与像素扰动的鲁棒性，
本模块把平台判重的 embedding 层换成真实预训练视觉编码器，作为更可信的
代理。模型经 ONNX Runtime 推理，惰性加载；模型缺失时可用性检查返回 False，
代理套件自动降级。
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image

_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
_SIZE = 224
_DIM = 512

_DEFAULT_MODEL = str(
    Path(__file__).resolve().parents[3] / "models" / "clip-vit-b32-vision-fp16.onnx"
)

_session = None
_model_path = os.environ.get("CTHULHU_DEEP_MODEL") or _DEFAULT_MODEL


def available() -> bool:
    """模型文件存在且 onnxruntime 可导入。"""
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

        _session = ort.InferenceSession(
            _model_path, providers=["CPUExecutionProvider"]
        )
    return _session


def _preprocess(frame: np.ndarray) -> np.ndarray:
    """单帧 → (1,3,224,224) 归一化张量；灰度帧复制为三通道。"""
    array = np.clip(np.asarray(frame, dtype=np.float32), 0.0, 1.0)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    image = Image.fromarray((array * 255.0).round().astype(np.uint8), mode="RGB")
    resized = np.asarray(image.resize((_SIZE, _SIZE), Image.BILINEAR), dtype=np.float32) / 255.0
    normalized = (resized - _MEAN) / _STD
    return normalized.transpose(2, 0, 1)[None, ...]


def embed_frame(frame: np.ndarray) -> np.ndarray:
    """单帧 CLIP 视觉 embedding（512 维，L2 归一）。"""
    session = _get_session()
    inputs = session.get_inputs()
    input_name = inputs[0].name
    tensor = _preprocess(frame)
    outputs = session.run(None, {input_name: tensor})
    # Xenova 导出的 vision_model 通常 image_embeds 在输出 0 或 1。
    vector = outputs[0].reshape(-1)
    if vector.shape[0] != _DIM and len(outputs) > 1:
        vector = outputs[1].reshape(-1)
    vector = vector[: _DIM].astype(np.float32)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else vector


def video_embedding(frames: np.ndarray, max_frames: int = 40) -> np.ndarray:
    """视频 embedding：抽样帧逐帧嵌入后均值池化并归一（CLIP 标准做法）。"""
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
