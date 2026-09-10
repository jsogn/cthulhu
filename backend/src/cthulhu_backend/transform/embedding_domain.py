"""嵌入域定向攻击：按水印画像选择表示域做低频重写。

依据 PUBLIC_SCHEME_REPORT.md 2.8 的嵌入域画像：
- VideoSeal：Y 通道低频、跨帧一致 → luma 低频重写；
- WAM：Cb/Cr 色度低频、局部 mask 内 → chroma 低频重写。

实现分两代：
- legacy：8×8 分块 DCT 低频带（zigzag 1..15，不含 DC）高斯噪声替换；
- v2（默认）：在 legacy 低频带之外补入报告 Top-10 频带，改用随机量化/
  抖动重写，并可选随机分块相位、多尺度 DCT 与 4:2:0 色度对齐。随机量化
  保留更多系数能量，观感更稳；随机相位/多尺度进一步破坏固定分块对齐。

纯 numpy 实现、确定性（同一 rng 结果一致）、灰度/彩色通用，不引入新依赖。
"""

from __future__ import annotations

import numpy as np

_DCT_CACHE: dict[int, np.ndarray] = {}

MODES = ("luma", "chroma", "both")
VARIANTS = ("legacy", "v2")

# 报告 §2.8 的 Top-10 频带（8×8 行优先索引），剔除 DC 0 避免整帧亮度漂移。
_TOP_BANDS = (1, 2, 3, 8, 9, 10, 16, 17, 24)
_MULTISCALE_BANDS = ((0, 1), (1, 0), (1, 1), (0, 2), (2, 0))


def _dct(n: int) -> np.ndarray:
    """n×n 正交 DCT-II 矩阵（与 scipy norm="ortho" 一致），模块级缓存。"""
    matrix = _DCT_CACHE.get(n)
    if matrix is None:
        frequency = np.arange(n)[:, None]
        sample = np.arange(n)[None, :]
        basis = np.cos(np.pi / n * (sample + 0.5) * frequency)
        basis *= np.sqrt(2.0 / n)
        basis[0, :] /= np.sqrt(2.0)
        matrix = basis.astype(np.float32)
        _DCT_CACHE[n] = matrix
    return matrix


def _dct8() -> np.ndarray:
    """8×8 正交 DCT-II 矩阵（保留旧测试/研究脚本入口）。"""
    return _dct(8)


def _zigzag(n: int = 8) -> list[tuple[int, int]]:
    """标准 zigzag 扫描顺序：低频在前，第 0 项为 DC。"""
    return sorted(
        ((i, j) for i in range(n) for j in range(n)),
        key=lambda ij: (ij[0] + ij[1], -ij[0] if (ij[0] + ij[1]) % 2 else ij[0]),
    )


def _low_band() -> tuple[list[int], list[int]]:
    """zigzag 索引 1..15（剔除 DC），对应报告画像里的 DCT 低频 0-15 主带。"""
    order = _zigzag(8)
    rows, cols = zip(*order[1:16])
    return list(rows), list(cols)


def _top_band() -> tuple[list[int], list[int]]:
    """报告 §2.8 Top-10 频带中除 DC 外的精确索引（行优先）。"""
    rows = [index // 8 for index in _TOP_BANDS]
    cols = [index % 8 for index in _TOP_BANDS]
    return rows, cols


def _attack_bands() -> tuple[list[int], list[int]]:
    """v2 攻击频带：legacy 1..15 与报告 Top-10 的并集，按低频优先排序。"""
    rows, cols = _low_band()
    top_rows, top_cols = _top_band()
    pairs = list(dict.fromkeys(zip(rows, cols, strict=True)))
    pairs.extend(
        pair
        for pair in zip(top_rows, top_cols, strict=True)
        if pair not in pairs
    )
    return [row for row, _ in pairs], [col for _, col in pairs]


_RGB_TO_YCBCR = np.array(
    [
        [0.299, 0.587, 0.114],
        [-0.168736, -0.331264, 0.5],
        [0.5, -0.418688, -0.081312],
    ],
    dtype=np.float32,
)

_YCBCR_TO_RGB = np.array(
    [
        [1.0, 0.0, 1.402],
        [1.0, -0.344136, -0.714136],
        [1.0, 1.772, 0.0],
    ],
    dtype=np.float32,
)


def _rgb_to_ycbcr(rgb: np.ndarray) -> np.ndarray:
    ycbcr = np.einsum("...c,dc->...d", rgb, _RGB_TO_YCBCR)
    ycbcr[..., 1:] += 0.5
    return ycbcr


def _ycbcr_to_rgb(ycbcr: np.ndarray) -> np.ndarray:
    centered = ycbcr.copy()
    centered[..., 1:] -= 0.5
    return np.einsum("...c,dc->...d", centered, _YCBCR_TO_RGB)


# 供 temporal.py 复用同一套颜色变换，避免两处矩阵漂移。
rgb_to_ycbcr = _rgb_to_ycbcr
ycbcr_to_rgb = _ycbcr_to_rgb


def _rewrite_plane(
    plane: np.ndarray,
    strength: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """对单个通道做分块 DCT 低频带重写，返回与原平面同尺寸的新平面。"""
    h, w = plane.shape[1:3]
    pad_h, pad_w = -h % 8, -w % 8
    padded = np.pad(plane, ((0, 0), (0, pad_h), (0, pad_w)), mode="edge")
    bh, bw = padded.shape[1] // 8, padded.shape[2] // 8
    blocks = padded.reshape(len(padded), bh, 8, bw, 8).transpose(0, 1, 3, 2, 4)
    matrix = _dct8()
    coeffs = matrix @ blocks @ matrix.T
    rows, cols = _low_band()
    band = coeffs[..., rows, cols]
    scale = band.std(axis=(1, 2, 3), keepdims=True)
    scale = np.maximum(scale, 1e-6)
    noise = rng.standard_normal(band.shape).astype(np.float32)
    coeffs[..., rows, cols] = (1.0 - strength) * band + strength * noise * scale
    recon = matrix.T @ coeffs @ matrix
    recon = recon.transpose(0, 1, 3, 2, 4).reshape(len(padded), padded.shape[1], padded.shape[2])
    return recon[:, :h, :w]


def _downsample2(plane: np.ndarray) -> np.ndarray:
    """2×2 均值下采样；奇数尺寸先边缘补齐，模拟 4:2:0 色度网格。"""
    h, w = plane.shape[1:3]
    pad_h, pad_w = h % 2, w % 2
    padded = np.pad(plane, ((0, 0), (0, pad_h), (0, pad_w)), mode="edge")
    return padded.reshape(len(padded), padded.shape[1] // 2, 2, padded.shape[2] // 2, 2).mean(
        axis=(2, 4)
    )


def _upsample2(plane: np.ndarray, height: int, width: int) -> np.ndarray:
    """最近邻 2× 上采样并裁回原尺寸；与 4:2:0 解码器口径一致。"""
    up = np.repeat(np.repeat(plane, 2, axis=1), 2, axis=2)
    return up[:, :height, :width]


def _rewrite_frame_dct(
    frame: np.ndarray,
    strength: float,
    rng: np.random.Generator,
    *,
    size: int,
    rows: list[int] | tuple[int, ...],
    cols: list[int] | tuple[int, ...],
    offset_y: int = 0,
    offset_x: int = 0,
) -> np.ndarray:
    """单帧分块 DCT 随机量化重写；返回与原帧同尺寸的 float32 平面。"""
    h, w = frame.shape
    pad_y = -(h + offset_y) % size
    pad_x = -(w + offset_x) % size
    padded = np.pad(
        frame,
        ((offset_y, pad_y), (offset_x, pad_x)),
        mode="edge",
    )
    bh, bw = padded.shape[0] // size, padded.shape[1] // size
    blocks = padded.reshape(bh, size, bw, size).transpose(0, 2, 1, 3)
    matrix = _dct(size)
    coeffs = matrix @ blocks @ matrix.T
    band = coeffs[..., rows, cols]
    spread = band.std(axis=(0, 1, 2), keepdims=True)
    magnitude = np.mean(np.abs(band), axis=(0, 1, 2), keepdims=True)
    # 平滑纹理的块间 std 很小，若只按 std 取量化步长会保留强相关；用
    # 平均幅值兜底，保证量化网格尺度与系数本身同量级。
    scale = np.maximum(np.maximum(spread, magnitude), 1e-6)
    step = np.maximum(scale * strength, 1e-6)
    dither = rng.uniform(-0.5, 0.5, band.shape).astype(np.float32)
    quantized = np.round(band / step + dither) * step
    # 量化后叠加小幅度随机项：保留能量分布，同时确保与原始系数去相关。
    quantized += (
        rng.normal(0.0, 2.0, band.shape).astype(np.float32) * scale * strength
    )
    coeffs[..., rows, cols] = (1.0 - strength) * band + strength * quantized
    recon = matrix.T @ coeffs @ matrix
    recon = recon.transpose(0, 2, 1, 3).reshape(padded.shape)
    return recon[offset_y : offset_y + h, offset_x : offset_x + w].astype(np.float32)


def _rewrite_plane_v2(
    plane: np.ndarray,
    strength: float,
    rng: np.random.Generator,
    *,
    block_jitter: bool,
    multiscale: bool,
) -> np.ndarray:
    """v2 逐帧重写：精确频带 + 随机量化；可选随机相位与 16×16 多尺度。"""
    rows, cols = _attack_bands()
    out = np.empty_like(plane, dtype=np.float32)
    for index in range(len(plane)):
        offset_y = int(rng.integers(0, 8)) if block_jitter else 0
        offset_x = int(rng.integers(0, 8)) if block_jitter else 0
        rewritten = _rewrite_frame_dct(
            plane[index],
            strength,
            rng,
            size=8,
            rows=rows,
            cols=cols,
            offset_y=offset_y,
            offset_x=offset_x,
        )
        if multiscale:
            rewritten = _rewrite_frame_dct(
                rewritten,
                strength * 0.5,
                rng,
                size=16,
                rows=[row for row, _ in _MULTISCALE_BANDS],
                cols=[col for _, col in _MULTISCALE_BANDS],
            )
        out[index] = rewritten
    return out


def attack(
    frames: np.ndarray,
    mode: str = "chroma",
    strength: float = 0.5,
    rng: np.random.Generator | None = None,
    *,
    variant: str = "v2",
    block_jitter: bool = False,
    multiscale: bool = False,
    chroma_subsample: bool = False,
) -> np.ndarray:
    """按嵌入域画像对帧块做低频重写；strength<=0 或不可用模式时原样返回。

    variant="legacy" 保持旧版高斯噪声替换；默认 v2 使用精确频带随机量化。
    block_jitter/multiscale/chroma_subsample 仅作用于 v2，属于增强档。
    """
    if strength <= 0:
        return frames
    if mode not in MODES:
        raise ValueError(f"未知嵌入域模式：{mode}（可选 {' / '.join(MODES)}）")
    if variant not in VARIANTS:
        raise ValueError(f"未知嵌入域变体：{variant}（可选 {' / '.join(VARIANTS)}）")
    if rng is None:
        rng = np.random.default_rng(0)
    is_u8 = frames.dtype == np.uint8
    work = frames.astype(np.float32)
    if is_u8:
        work /= 255.0
    color = work.ndim == 4
    rgb = work if color else np.repeat(work[..., None], 3, axis=-1)
    ycbcr = _rgb_to_ycbcr(rgb)
    channels = {"luma": (0,), "chroma": (1, 2), "both": (0, 1, 2)}[mode]
    for channel in channels:
        plane = ycbcr[..., channel]
        if variant == "legacy":
            ycbcr[..., channel] = _rewrite_plane(plane, strength, rng)
            continue
        if chroma_subsample and channel in (1, 2):
            height, width = plane.shape[1:3]
            down = _downsample2(plane)
            down = _rewrite_plane_v2(
                down,
                strength,
                rng,
                block_jitter=block_jitter,
                multiscale=multiscale,
            )
            ycbcr[..., channel] = _upsample2(down, height, width)
        else:
            ycbcr[..., channel] = _rewrite_plane_v2(
                plane,
                strength,
                rng,
                block_jitter=block_jitter,
                multiscale=multiscale,
            )
    out = np.clip(_ycbcr_to_rgb(ycbcr), 0.0, 1.0)
    if not color:
        out = out[..., 0]
    if is_u8:
        return (out * 255.0).round().astype(np.uint8)
    return out.astype(np.float32)
