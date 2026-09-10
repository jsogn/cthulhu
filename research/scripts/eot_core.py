"""白盒 EOT-PGD 核心（研究侧）：对已知公开深度水印方案的抗编码扰动。

依据 PUBLIC_SCHEME_REPORT.md 2.3 的结论：dual-tail 方向（最小化 logits²，
把比特推向随机）、正均值时变调制、ε=3~4/255、50~100 步、16 帧分块梯度
累积可穿过真实 H.264/H.265 CRF23/28；零均值时变无效，预算/步数随视频
长度缩放。

本模块只提供算法核心，不绑定具体模型：调用方把目标方案 decoder 注入
（decode_fn 输入 (N,3,H,W) 的 [0,1] float 张量，输出 (N,bits) 的逐帧
logits），scheme 注册表仅做命名管理。torch 为可选依赖。

**边界**：这是研究专用原语，不能进入生产包——白盒攻击必须拿到目标方案
decoder，产品侧不存在该条件。生产包只保留黑盒原语（purify/embedding_domain）。
自检：`research/.venv/bin/python research/scripts/check_eot_core.py`。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

DecodeFn = Callable[[object], object]
EmbedFn = Callable[[np.ndarray, list[int]], np.ndarray]


@dataclass
class EotOptions:
    """EOT-PGD 预算；alpha 缺省按 2ε/steps 推导，保证总步进不超预算。"""

    eps: float = 4.0 / 255.0
    steps: int = 50
    chunk: int = 16
    mode: str = "modulated"  # static / per_frame / modulated（正均值时变）
    alpha: float | None = None
    eot: bool = True
    scale_min: float = 0.4
    scale_max: float = 0.9
    blur_max: float = 2.0
    drop_min: float = 0.4
    drop_max: float = 1.0


@dataclass
class Scheme:
    """已知方案注册条目：embed 返回带水印帧，decode 返回逐帧 logits。"""

    name: str
    embed: EmbedFn
    decode: DecodeFn


_SCHEMES: dict[str, Scheme] = {}


def register_scheme(name: str, embed: EmbedFn, decode: DecodeFn) -> None:
    """注册已知方案（同名覆盖视为显式替换）。"""
    _SCHEMES[name] = Scheme(name=name, embed=embed, decode=decode)


def get_scheme(name: str) -> Scheme:
    if name not in _SCHEMES:
        raise KeyError(f"未注册的目标方案：{name}")
    return _SCHEMES[name]


def _device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _gaussian_blur(x: object, sigma: float) -> object:
    import torch
    import torch.nn.functional as F

    if sigma <= 0:
        return x
    radius = int(np.ceil(3.0 * sigma))
    kernel_size = 2 * radius + 1
    grid = torch.arange(kernel_size, device=x.device, dtype=x.dtype) - radius
    weights = torch.exp(-(grid**2) / (2.0 * sigma**2))
    weights = weights / weights.sum()
    kernel = weights[None, None, :] * weights[None, None, :, None]
    kernel = kernel.expand(x.shape[1], 1, kernel_size, kernel_size)
    return F.conv2d(x, kernel, padding=radius, groups=x.shape[1])


def _eot_forward(x: object, options: EotOptions, rng: np.random.Generator) -> object:
    """可微 EOT 代理：随机缩放往返 + 高斯模糊 + 随机抽帧（JPEG 近似见研究脚本）。"""
    import torch
    import torch.nn.functional as F

    if not options.eot:
        return x
    frames, _, height, width = x.shape
    scale = float(rng.uniform(options.scale_min, options.scale_max))
    small = (max(2, round(height * scale)), max(2, round(width * scale)))
    x = F.interpolate(x, size=small, mode="bilinear", align_corners=False)
    x = F.interpolate(x, size=(height, width), mode="bilinear", align_corners=False)
    sigma = float(rng.uniform(0.0, options.blur_max))
    x = _gaussian_blur(x, sigma)
    keep = float(rng.uniform(options.drop_min, options.drop_max))
    drop = rng.random(frames) > keep
    if drop.any():
        mask = torch.from_numpy(~drop).to(x.device).to(x.dtype)[:, None, None, None]
        x = x * mask
    return x


def _positive_mean_modulation(frames: int, device: object, dtype: object) -> object:
    """正均值时变标量序列：跨帧均值精确为 1（零均值时变已被报告证伪）。"""
    import torch

    period = max(2, frames // 4)
    time = torch.arange(frames, device=device, dtype=torch.float32)
    m = 1.0 + 0.45 * torch.sin(2.0 * np.pi * time / period)
    m = m - (m.mean() - 1.0)
    return m.to(dtype)[:, None, None, None]


def run(
    frames: np.ndarray,
    decode_fn: DecodeFn,
    options: EotOptions | None = None,
    seed: int = 0,
    progress: Callable[[int, int], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> np.ndarray:
    """对帧块做白盒 EOT-PGD，返回 [0,1] float32（uint8 输入返回 uint8）。"""
    import torch

    opts = options or EotOptions()
    if opts.mode not in ("static", "per_frame", "modulated"):
        raise ValueError(f"未知扰动模式：{opts.mode}")
    is_u8 = frames.dtype == np.uint8
    gray = frames.ndim == 3
    arr = frames.astype(np.float32)
    if is_u8:
        arr = arr / 255.0
    if gray:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    device = _device()
    original = torch.from_numpy(arr).permute(0, 3, 1, 2).to(device)
    count, _, height, width = original.shape
    shared = opts.mode in ("static", "modulated")
    shape = (1 if shared else count, 3, height, width)
    rng = np.random.default_rng(seed)
    perturb = (torch.from_numpy(rng.uniform(-1.0, 1.0, shape)).float().to(device) * 0.1 * opts.eps)
    perturb.requires_grad_(True)
    alpha = opts.alpha if opts.alpha is not None else opts.eps / 5.0

    def expand(p: torch.Tensor) -> torch.Tensor:
        base = p if not shared else p.expand(count, 3, height, width)
        if opts.mode == "modulated":
            base = base * _positive_mean_modulation(count, p.device, p.dtype)
        return torch.clamp(original + base, 0.0, 1.0)

    for step in range(opts.steps):
        if should_stop is not None and should_stop():
            raise InterruptedError("EOT-PGD 已取消")
        if perturb.grad is not None:
            perturb.grad.zero_()
        step_rng = np.random.default_rng(seed ^ (step + 1))
        for start in range(0, count, opts.chunk):
            window = expand(perturb)[start : start + opts.chunk]
            logits = decode_fn(_eot_forward(window, opts, step_rng))
            loss = (logits**2).mean()
            loss.backward()
            del logits, loss
            if device == "mps":
                torch.mps.empty_cache()
        with torch.no_grad():
            # dual-tail descend：最小化 logits²，把比特推向随机（报告 2.3 口径）。
            perturb.sub_(alpha * perturb.grad.sign())
            perturb.clamp_(-opts.eps, opts.eps)
        if progress is not None:
            progress(step + 1, opts.steps)

    out = expand(perturb).permute(0, 2, 3, 1).detach().cpu().numpy()
    if gray:
        out = 0.299 * out[..., 0] + 0.587 * out[..., 1] + 0.114 * out[..., 2]
    if is_u8:
        return (np.clip(out, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    return np.clip(out, 0.0, 1.0).astype(np.float32)
