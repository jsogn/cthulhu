"""帧级感知哈希与浅层特征代理（numpy/scipy 原生实现）。

对应平台的候选机制：
- aHash/pHash/dHash：经典感知哈希，平台查重最底层组件；
- DCT 符号哈希：频域结构签名，对亮度平移鲁棒、对空间结构敏感；
- 边缘直方图：浅层 CNN（边缘/纹理滤波器组）的近似代理；
- 关键点网格：SIFT/关键点匹配的空间布局代理。
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from scipy.fft import dctn
from scipy.ndimage import maximum_filter, sobel


def resize_gray(frame: np.ndarray, size: int) -> np.ndarray:
    """PIL 双线性缩放到 size×size 的 float64 灰度图。"""
    image = Image.fromarray((np.clip(frame, 0, 1) * 255).round().astype(np.uint8), mode="L")
    return np.asarray(image.resize((size, size), Image.BILINEAR), dtype=np.float64) / 255.0


def ahash(frame: np.ndarray, size: int = 16) -> np.ndarray:
    """均值哈希：低分辨率亮度均值阈值。"""
    small = resize_gray(frame, size)
    return small.ravel() > small.mean()


def phash(frame: np.ndarray, size: int = 32, low: int = 8) -> np.ndarray:
    """感知哈希：DCT 低频系数按（去直流）均值阈值。"""
    img = resize_gray(frame, size)
    coeffs = dctn(img, norm="ortho")
    low_freq = coeffs[:low, :low].ravel()
    threshold = low_freq[1:].mean() if low_freq.size > 1 else low_freq.mean()
    return low_freq > threshold


def dct_sign(frame: np.ndarray, size: int = 32, low: int = 24) -> np.ndarray:
    """DCT 符号哈希：低频系数正负号组成的结构签名。"""
    img = resize_gray(frame, size)
    coeffs = dctn(img, norm="ortho")
    return (coeffs[:low, :low] >= 0).ravel()


def edge_embedding(frame: np.ndarray, size: int = 16) -> np.ndarray:
    """边缘直方图 embedding：梯度幅值的块均值池化，L2 归一。"""
    img = resize_gray(frame, 64)
    gx = sobel(img, axis=1, mode="reflect")
    gy = sobel(img, axis=0, mode="reflect")
    magnitude = np.sqrt(gx**2 + gy**2)
    vector = block_mean(magnitude, size).ravel().astype(np.float64)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else vector


def block_mean(array: np.ndarray, size: int) -> np.ndarray:
    """把方阵块均值降采样到 size×size。"""
    h, w = array.shape
    bh, bw = h // size, w // size
    cropped = array[: bh * size, : bw * size]
    return cropped.reshape(size, bh, size, bw).mean(axis=(1, 3))


def interest_grid(frame: np.ndarray, cells: int = 16, topk: int = 48) -> np.ndarray:
    """关键点网格：梯度幅值局部极值点落到网格单元后的归一化直方图。

    不逐点做描述子匹配，只比较关键点的空间布局重叠（SIFT 匹配的空间代理）。
    """
    img = resize_gray(frame, 128)
    gx = sobel(img, axis=1, mode="reflect")
    gy = sobel(img, axis=0, mode="reflect")
    magnitude = np.sqrt(gx**2 + gy**2)
    local_max = magnitude == maximum_filter(magnitude, size=5, mode="reflect")
    threshold = np.percentile(magnitude, 92)
    ys, xs = np.nonzero(local_max & (magnitude >= threshold))
    order = np.argsort(-magnitude[ys, xs])[:topk]
    ys, xs = ys[order], xs[order]
    h, w = img.shape
    cell_y = np.minimum(ys * cells // h, cells - 1)
    cell_x = np.minimum(xs * cells // w, cells - 1)
    counts = np.zeros(cells * cells, dtype=np.float64)
    np.add.at(counts, cell_y * cells + cell_x, 1.0)
    total = counts.sum()
    return counts / total if total > 0 else counts


def hamming_bits(a: np.ndarray, b: np.ndarray) -> int:
    """两个等长布尔哈希的汉明距离。"""
    return int(np.count_nonzero(np.asarray(a) != np.asarray(b)))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """向量余弦相似度，空向量按 0 处理。"""
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b)) / denom if denom > 0 else 0.0
