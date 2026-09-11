"""再生族攻击：把画面从「变换」升级为「重估计/重采样」，攻击更鲁棒的水印。

与经典攻击（滤波/重量化/几何）互补，覆盖四类盲区：
- temporal_subtract：跨帧估计帧间固定水印并过减（针对空域扩频协同检测）；
- fft_phase：保留频谱幅值、随机化中高频相位（针对 DFT 扩频相位相关）；
- dwt_detail：随机化 Haar 对角线细节子带（针对小波细节子带嵌入）；

全部原语确定性（同一 rng 下结果一致），灰度/彩色帧通用，不做几何改变，
不引入新依赖（numpy/scipy/PIL/cv2 均为既有依赖）。
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import zoom

_FIELD_CACHE: dict[tuple[str, int], np.ndarray] = {}
_ULTRAFACE_SESSION = None


def _as_float(frames: np.ndarray) -> tuple[np.ndarray, np.dtype]:
    original_dtype = frames.dtype
    if original_dtype == np.float32:
        return frames, original_dtype
    work = frames.astype(np.float32)
    if original_dtype == np.uint8:
        work /= 255.0
    return work, original_dtype


def _as_original(work: np.ndarray, original_dtype: np.dtype) -> np.ndarray:
    if original_dtype == np.uint8:
        return (np.clip(work, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    if original_dtype == np.float32:
        return np.clip(work, 0.0, 1.0)
    return np.clip(work, 0.0, 1.0).astype(original_dtype)


def _spatial_filter_size(ndim: int, size: int) -> tuple[int, ...]:
    if ndim == 3:  # (F,H,W)
        return (1, size, size)
    if ndim == 4:  # (F,H,W,3)
        return (1, size, size, 1)
    return (size, size)


def _radial_mask(h: int, w: int, radius_frac: float) -> np.ndarray:
    fx = np.fft.fftfreq(w)[: w // 2 + 1]
    fy = np.fft.fftfreq(h)
    radius = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    return (radius > radius_frac * 0.5).astype(np.float32)


def temporal_subtract(frames: np.ndarray, beta: float, size: int = 3) -> np.ndarray:
    """跨帧估计相减：残差的时域均值近似帧间固定水印，按 beta 倍过减。

    水印跨帧不变而内容随时间变化时，E[frame - median(frame)] 收敛到水印
    图案；对静态画面（字幕/静态背景）会连带削弱其边缘对比，故 beta 建议
    取 0.6~1.0，由档位按素材类型调节。亮度域计算 + 帧抽样估计（长片只取
    至多 24 帧），把中值滤波成本降到约 1/10。
    """
    if beta <= 0:
        return frames
    work, original_dtype = _as_float(frames)
    if work.ndim == 4:
        luma = 0.299 * work[..., 0] + 0.587 * work[..., 1] + 0.114 * work[..., 2]
    else:
        luma = work
    if len(luma) > 24:
        indices = np.linspace(0, len(luma) - 1, 24).astype(int)
        sampled = luma[indices]
    else:
        sampled = luma
    import cv2

    sampled_u8 = (np.clip(sampled, 0, 1) * 255).round().astype(np.uint8)
    smoothed = np.stack([cv2.medianBlur(frame, size) for frame in sampled_u8]).astype(
        np.float32
    ) / 255.0
    residual = sampled - smoothed
    estimate = residual.mean(axis=0)
    attacked = work - beta * (estimate[..., None] if work.ndim == 4 else estimate)
    return _as_original(attacked, original_dtype)


def fft_phase(
    frames: np.ndarray,
    strength: float,
    rng: np.random.Generator,
    radius_frac: float = 0.35,
) -> np.ndarray:
    """FFT 相位随机化：亮度通道中高频相位按强度打散，幅值保持不变。

    DFT 扩频水印依赖相位/系数相关，打散相位即破坏相关；只动中高频
    （半径大于 radius_frac×Nyquist），低频与大结构观感保持稳定。
    只处理亮度、色差不动，降低色彩伪影与计算量。
    """
    if strength <= 0:
        return frames
    from cthulhu_backend.parallel import map_frames

    work, original_dtype = _as_float(frames)
    if work.ndim == 4:
        luma = 0.299 * work[..., 0] + 0.587 * work[..., 1] + 0.114 * work[..., 2]
    else:
        luma = work

    def one(frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape
        spectrum = np.fft.rfft2(frame)
        magnitude = np.abs(spectrum)
        phase = np.angle(spectrum)
        mask = _radial_mask(h, w, radius_frac)
        random_phase = rng.uniform(-np.pi, np.pi, spectrum.shape).astype(np.float32)
        new_phase = (1.0 - strength * mask) * phase + strength * mask * random_phase
        return np.fft.irfft2(magnitude * np.exp(1j * new_phase), s=(h, w))

    new_luma = np.stack(map_frames(one, [luma[i] for i in range(len(luma))]))
    if work.ndim == 4:
        delta = (new_luma - luma)[..., None]
        return _as_original(work + delta, original_dtype)
    return _as_original(new_luma, original_dtype)


def dwt_detail(
    frames: np.ndarray,
    strength: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Haar 对角线细节子带随机化：破坏小波细节子带嵌入的相关结构。"""
    if strength <= 0:
        return frames
    from cthulhu_backend.parallel import map_frames
    from cthulhu_backend.watermark import dwt as dwt_module

    work, original_dtype = _as_float(frames)

    def one_plane(frame: np.ndarray) -> np.ndarray:
        coeffs, (h2, w2) = dwt_module._decompose(frame)
        band = coeffs[coeffs.shape[0] // 2 :, coeffs.shape[1] // 2 :]
        scale = float(np.std(band))
        noise = rng.standard_normal(band.shape).astype(band.dtype) * scale
        coeffs[coeffs.shape[0] // 2 :, coeffs.shape[1] // 2 :] = (
            (1.0 - strength) * band + strength * noise
        )
        out = frame.copy()
        out[:h2, :w2] = dwt_module._reconstruct(coeffs)
        return out

    def one(frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 3:
            luma = 0.299 * frame[..., 0] + 0.587 * frame[..., 1] + 0.114 * frame[..., 2]
            new_luma = one_plane(luma)
            return np.clip(frame + (new_luma - luma)[..., None], 0.0, 1.0)
        return one_plane(frame)

    attacked = np.stack(map_frames(one, [work[i] for i in range(len(work))]))
    return _as_original(attacked, original_dtype)


def hsv_jitter(
    frames: np.ndarray,
    strength: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """HSV 抖动：逐帧微移色相、微调饱和度，破坏色度/色彩描述子。"""
    if strength <= 0 or frames.ndim != 4:
        return frames
    import cv2


    is_u8 = frames.dtype == np.uint8
    hue_deltas = rng.uniform(-strength, strength, len(frames))
    sat_factors = 1.0 + rng.uniform(-1.0, 1.0, len(frames)) * min(0.25, strength * 0.02)

    def one(pair: tuple[np.ndarray, float, float]) -> np.ndarray:
        frame, hue_delta, sat_factor = pair
        base = frame if is_u8 else (np.clip(frame, 0, 1) * 255).round().astype(np.uint8)
        hsv = cv2.cvtColor(base, cv2.COLOR_RGB2HSV).astype(np.int16)
        hsv[..., 0] = (hsv[..., 0] + round(hue_delta * 0.5)) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] * sat_factor, 0, 255)
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        if is_u8:
            return result
        return result.astype(frames.dtype) / 255.0

    # OpenCV 在多线程析构时存在 TLS 崩溃风险，色相抖动顺序执行。
    return np.stack([one(item) for item in zip(frames, hue_deltas, sat_factors)])


def _spsa_field(
    frames: np.ndarray,
    strength: float,
    rng: np.random.Generator,
    embed_fn,
    cache_name: str,
    iters: int = 40,
    grid: tuple[int, int] = (24, 16),
    step: float = 0.05,
    probe: float = 0.02,
    warm: bool = False,
) -> np.ndarray:
    """SPSA 黑盒优化：低分辨率平滑场参数化，攻击给定 embedding 函数。

    扰动场按任务缓存（同一 rng 与同一嵌入器跨分块复用），避免逐块重算与
    块间闪烁；余弦相似度经 L2 归一后的点积计算。
    """
    original_dtype = frames.dtype
    work = frames.astype(np.float32)
    if original_dtype == np.uint8:
        work /= 255.0
    h, w = work.shape[1:3]
    key = (cache_name, id(rng))
    field = _FIELD_CACHE.get(key)
    if field is None or field.shape != (h, w):
        anchor = work[len(work) // 2]
        base = embed_fn(anchor)
        theta = np.zeros(grid, dtype=np.float32)
        if warm:
            theta = rng.uniform(-0.6 * strength, 0.6 * strength, grid).astype(np.float32)

        def evaluate(t: np.ndarray) -> float:
            smooth = zoom(
                np.clip(t, -strength, strength),
                (h / grid[0], w / grid[1]),
                order=1,
            )
            candidate = anchor + smooth[..., None] if anchor.ndim == 3 else anchor + smooth
            return 1.0 - float(np.dot(base, embed_fn(np.clip(candidate, 0.0, 1.0))))

        for _ in range(iters):
            delta = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=grid)
            positive = np.clip(theta + probe * delta, -strength, strength)
            negative = np.clip(theta - probe * delta, -strength, strength)
            gradient = (evaluate(positive) - evaluate(negative)) / (2.0 * probe) * delta
            theta = np.clip(theta + step * gradient, -strength, strength)
        field = zoom(theta, (h / grid[0], w / grid[1]), order=1)
        if len(_FIELD_CACHE) >= 8:
            _FIELD_CACHE.pop(next(iter(_FIELD_CACHE)))
        _FIELD_CACHE[key] = field
    return field


def _apply_field(frames: np.ndarray, field: np.ndarray) -> np.ndarray:
    original_dtype = frames.dtype
    work = frames.astype(np.float32)
    if original_dtype == np.uint8:
        work /= 255.0
    attacked = work + field[..., None] if work.ndim == 4 else work + field
    return _as_original(attacked, original_dtype)


def copy_attack(
    frames: np.ndarray,
    strength: float,
    rng: np.random.Generator,
    iters: int = 30,
    grid: tuple[int, int] = (24, 16),
    step: float = 0.05,
    probe: float = 0.02,
    warm: bool = True,
    field: np.ndarray | None = None,
) -> np.ndarray:
    """判重代理对抗：SPSA 黑盒攻击 DINOv2 拷贝检测描述子。"""
    if strength <= 0:
        return frames
    if field is not None and field.shape == frames.shape[1:3]:
        return _apply_field(frames, field)
    from cthulhu_backend.fingerprint import copy_detect

    if not copy_detect.available():
        return frames
    field = _spsa_field(
        frames, strength, rng, copy_detect.embed_frame, "copy",
        iters=iters, grid=grid, step=step, probe=probe, warm=warm,
    )
    return _apply_field(frames, field)


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.3) -> list[int]:
    order = np.argsort(scores)[::-1]
    keep: list[int] = []
    while len(order):
        current = order[0]
        keep.append(current)
        if len(order) == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[current, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[current, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[current, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[current, 3], boxes[rest, 3])
        intersection = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        area_current = (boxes[current, 2] - boxes[current, 0]) * (
            boxes[current, 3] - boxes[current, 1]
        )
        area_rest = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = intersection / (area_current + area_rest - intersection + 1e-6)
        order = rest[iou <= iou_threshold]
    return keep


def _face_boxes(base_u8: np.ndarray, min_size: int = 24) -> list[tuple[int, int, int, int]]:
    """UltraFace ONNX 人脸检测：320×240 推理 + numpy NMS，框映射回原分辨率。"""
    global _ULTRAFACE_SESSION

    from pathlib import Path

    import cv2

    model_path = Path(__file__).resolve().parents[3] / "models" / "ultraface.onnx"
    if not model_path.is_file():
        return []
    if _ULTRAFACE_SESSION is None:
        import onnxruntime as ort

        _ULTRAFACE_SESSION = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    h, w = base_u8.shape[:2]
    rgb = base_u8 if base_u8.ndim == 3 else np.repeat(base_u8[..., None], 3, axis=-1)
    resized = cv2.resize(rgb, (320, 240), interpolation=cv2.INTER_LINEAR)
    tensor = ((resized.astype(np.float32) - 127.5) / 128.0).transpose(2, 0, 1)[None, ...]
    scores, boxes = _ULTRAFACE_SESSION.run(
        None, {_ULTRAFACE_SESSION.get_inputs()[0].name: tensor}
    )
    face_scores = scores[0, :, 1]
    candidates = np.where(face_scores > 0.7)[0]
    if len(candidates) == 0:
        return []
    detected: list[tuple[int, int, int, int]] = []
    for index in _nms(boxes[0][candidates], face_scores[candidates]):
        x1, y1, x2, y2 = boxes[0][candidates[index]]
        bw = max(min_size, round((x2 - x1) * w))
        bh = max(min_size, round((y2 - y1) * h))
        bx = max(0, round(x1 * w))
        by = max(0, round(y1 * h))
        detected.append((bx, by, min(bw, w - bx), min(bh, h - by)))
    return detected


def _perturb_regions(
    frame: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    strength: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """向指定区域注入带通高频扰动：区域外像素原样保留。"""
    from scipy.ndimage import gaussian_filter

    h, w = frame.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    for x, y, bw, bh in boxes:
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(w, x + bw), min(h, y + bh)
        mask[y0:y1, x0:x1] = 1.0
    mask = gaussian_filter(mask, sigma=3.0)
    grain = rng.standard_normal((h, w)).astype(np.float32)
    grain = grain - gaussian_filter(grain, sigma=0.8)
    grain /= max(float(grain.std()), 1e-6)
    injected = strength * grain * mask
    if frame.ndim == 3:
        return np.clip(frame + injected[..., None], 0.0, 1.0)
    return np.clip(frame + injected, 0.0, 1.0)


def face_perturb(
    frames: np.ndarray,
    strength: float,
    rng: np.random.Generator,
    min_size: int = 24,
    refine_iters: int = 20,
    field: np.ndarray | None = None,
) -> np.ndarray:
    """人脸抗AI扰动：只在人脸区域注入带通扰动，破坏人脸识别/检测特征。

    本产品口径为攻脸、不保脸。无脸帧原样返回；区域外像素保持逐位一致。
    ArcFace 代理可用时，以锚帧最大脸框做 SPSA 精修（攻击真实人脸
    embedding），否则退化为逐帧带通扰动。
    """
    if strength <= 0:
        return frames
    original_dtype = frames.dtype
    is_u8 = original_dtype == np.uint8
    work = frames.astype(np.float32)
    if is_u8:
        work /= 255.0
    h, w = work.shape[1:3]
    if field is not None and field.shape == (h, w):
        attacked = work + field[..., None] if work.ndim == 4 else work + field
        return _as_original(attacked, original_dtype)
    anchor_index = len(work) // 2
    anchor_u8 = (np.clip(work[anchor_index], 0, 1) * 255).round().astype(np.uint8)
    boxes = _face_boxes(anchor_u8, min_size)
    if boxes:
        from cthulhu_backend.fingerprint import face_embed

        if face_embed.available():
            key = ("facefield", id(rng))
            field = _FIELD_CACHE.get(key)
            if field is None or field.shape != (h, w):
                x, y, bw, bh = max(boxes, key=lambda box: box[2] * box[3])
                crop = anchor_u8[y : y + bh, x : x + bw]
                base = face_embed.embed_crop(crop)
                grid = (10, 10)
                theta = rng.uniform(-0.6 * strength, 0.6 * strength, grid).astype(np.float32)

                def evaluate(t: np.ndarray) -> float:
                    smooth = zoom(
                        np.clip(t, -strength, strength),
                        (bh / grid[0], bw / grid[1]),
                        order=1,
                    )
                    candidate = np.clip(crop.astype(np.float32) / 255.0 + smooth[..., None], 0, 1)
                    candidate_u8 = (candidate * 255).round().astype(np.uint8)
                    return 1.0 - float(np.dot(base, face_embed.embed_crop(candidate_u8)))

                for _ in range(refine_iters):
                    delta = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=grid)
                    positive = np.clip(theta + 0.015 * delta, -strength, strength)
                    negative = np.clip(theta - 0.015 * delta, -strength, strength)
                    gradient = (evaluate(positive) - evaluate(negative)) / 0.03 * delta
                    theta = np.clip(theta + 0.04 * gradient, -strength, strength)
                crop_field = zoom(theta, (bh / grid[0], bw / grid[1]), order=1)
                field = np.zeros((h, w), dtype=np.float32)
                field[y : y + bh, x : x + bw] = crop_field
                if len(_FIELD_CACHE) >= 8:
                    _FIELD_CACHE.pop(next(iter(_FIELD_CACHE)))
                _FIELD_CACHE[key] = field
            attacked = work + field[..., None] if work.ndim == 4 else work + field
            return _as_original(attacked, original_dtype)
    out = []
    for frame in work:
        base_u8 = frame if is_u8 else (np.clip(frame, 0, 1) * 255).round().astype(np.uint8)
        boxes = _face_boxes(base_u8, min_size)
        if not boxes:
            out.append(frame)
            continue
        out.append(_perturb_regions(frame, boxes, strength, rng))
    attacked = np.stack(out)
    return _as_original(attacked, original_dtype)

