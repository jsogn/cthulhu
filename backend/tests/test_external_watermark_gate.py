"""外部水印方案回归门：第三方嵌入 → 我方攻击 → 第三方解码验证破坏力。

依赖均为可选：缺少对应库/权重时跳过，不阻塞常规测试。
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from cthulhu_backend.attacks import spatial

blind_watermark = pytest.importorskip("blind_watermark")


def _text_embed_extract(image: np.ndarray, text: str) -> tuple[np.ndarray, int, str]:
    bwm = blind_watermark.WaterMark(password_wm=1, password_img=1)
    bwm.read_img(img=image)
    bwm.read_wm(text, mode="str")
    marked = bwm.embed()
    n_bits = len(bwm.wm_bit)
    extracted = bwm.extract(embed_img=marked, wm_shape=n_bits, mode="str")
    return marked, n_bits, extracted


def _extract_text(image: np.ndarray, n_bits: int) -> str:
    bwm = blind_watermark.WaterMark(password_wm=1, password_img=1)
    return bwm.extract(embed_img=image, wm_shape=n_bits, mode="str")


def test_guofei_svd_roundtrip_and_attack_degrades() -> None:
    rng = np.random.default_rng(0)
    image = (rng.random((128, 128, 3)) * 255).astype(np.uint8)
    text = "cthulhu-gate-2026"
    marked, n_bits, extracted_clean = _text_embed_extract(image, text)
    assert extracted_clean == text

    attacked = spatial.median(marked[None, ...].astype(np.float32) / 255.0, size=3)[0]
    attacked = (np.clip(attacked, 0, 1) * 255).round().astype(np.uint8)
    extracted_attacked = _extract_text(attacked, n_bits)
    mismatches = sum(a != b for a, b in zip(text, extracted_attacked, strict=False))
    assert mismatches >= 1, "中值滤波应显著破坏 DWT-DCT-SVD 文本水印"


def _rivagan_dir() -> Path:
    return Path(os.environ.get("CTHULHU_RIVAGAN_DIR", Path(__file__).resolve().parents[1] / "data" / "rivagan"))


def test_rivagan_onnx_roundtrip_and_attack_degrades() -> None:
    import onnxruntime as ort

    directory = _rivagan_dir()
    encoder_path = directory / "rivagan_encoder.onnx"
    decoder_path = directory / "rivagan_decoder.onnx"
    if not encoder_path.is_file() or not decoder_path.is_file():
        pytest.skip("缺少 RivaGAN ONNX 权重（CTHULHU_RIVAGAN_DIR 未就绪）")

    enc = ort.InferenceSession(str(encoder_path), providers=["CPUExecutionProvider"])
    dec = ort.InferenceSession(str(decoder_path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    smooth = gaussian_filter(rng.random((256, 256)), 6.0)
    luma = np.clip(0.35 + 0.2 * (smooth - smooth.mean()), 0, 1)
    frame = (np.repeat(luma[..., None], 3, axis=-1) * 255).astype(np.uint8)
    frame[..., 1] = np.clip(frame[..., 1].astype(np.float32) * 0.95 + 8, 0, 255).astype(np.uint8)
    frame[..., 2] = np.clip(frame[..., 2].astype(np.float32) * 1.05 - 8, 0, 255).astype(np.uint8)
    payload = np.zeros((1, 32), dtype=np.float32)
    payload[0, :16] = 1.0

    def preprocess(image: np.ndarray) -> np.ndarray:
        tensor = (image.astype(np.float32) / 127.5 - 1.0).transpose(2, 0, 1)
        return tensor[None, :, None, ...]  # (1,C,1,H,W)

    marked = enc.run(None, {"frame": preprocess(frame), "data": payload})[0]
    marked = np.clip(marked, -1.0, 1.0)[0, :, 0]
    marked = ((marked.transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8)

    def decode(image: np.ndarray, session) -> np.ndarray:
        data = session.run(None, {"frame": preprocess(image)})[0][0]
        return (data > 0.5).astype(np.uint8)

    clean_bits = decode(marked, dec)
    clean_ber = float(np.mean(clean_bits != payload[0].astype(np.uint8)))
    if clean_ber > 0.15:
        # 提前释放 ONNX 会话：会话滞留到解释器退出会触发 macOS 上的析构竞态。
        del enc, dec
        import gc

        gc.collect()
        pytest.skip(f"该 RivaGAN ONNX 导出不可用（干净解码 BER={clean_ber:.3f}），需上游 torch 模型另行评估")

    noise = rng.normal(0, 8.0, marked.shape)
    attacked = np.clip(marked.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    attacked_ber = float(np.mean(decode(attacked, dec) != payload[0].astype(np.uint8)))
    assert attacked_ber > clean_ber, "噪声攻击应提高 RivaGAN 解码 BER"
    del enc, dec
