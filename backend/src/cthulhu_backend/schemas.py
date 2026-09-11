"""API 与队列共享的请求契约。

DesensitizeOptions 是清洗选项的唯一权威定义：直接端点、任务队列与
前端生成类型都从这里派生，避免 Python/TS 两侧手写副本漂移。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from cthulhu_backend.tiers import (
    SCHEMA_DEFAULT_TIER_ID,
    TIER_BY_ID,
    TIER_DETAIL,
    TIER_DETAIL_SIGMA,
)

# 净化字段默认值取自「平衡」档：与历史行为一致，且与预置/面板共用同一份表（tiers.py）。
_SCHEMA_TIER = TIER_BY_ID[SCHEMA_DEFAULT_TIER_ID]

# 已下线：扩散引擎随 sd-turbo（2.4GB）一起移除，strength 现在只作开关，
# 步数与 guidance 不再有消费方。旧模板/旧脚本仍带这些键，入参处丢弃而不是
# 报错，避免存量数据一升级就 422。
_DEPRECATED_INPUT_KEYS = frozenset(
    {
        "purify_engine",
        "purifyEngine",
        "purify_steps",
        "purifySteps",
        "purify_guidance",
        "purifyGuidance",
    }
)


class DesensitizeOptions(BaseModel):
    """清洗选项集：snake_case 为唯一字段名，未知字段直接报错。"""

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def _drop_deprecated(cls, data: object) -> object:
        if isinstance(data, dict):
            return {
                key: value
                for key, value in data.items()
                if key not in _DEPRECATED_INPUT_KEYS
            }
        return data

    reorder: bool = False
    speed: float = Field(1.0, gt=0)
    recrop: float = Field(0.0, ge=0, le=0.2)
    regrade: bool = True
    perturb: float = Field(0.0, ge=0, le=1)
    audio_remix: bool = True
    sharpness: bool = True
    color_restore: bool = True
    denoise: bool = False
    anti_reembed: bool = False
    banner: str = ""
    seed: int = 0
    codec: str = "libx264"
    lossless: bool = False
    preset: str = "veryfast"
    spoof: bool = False
    bitrate_kbps: int | None = Field(None, ge=100, le=100000)
    gop: int | None = Field(None, ge=1, le=600)
    resolution: str | None = None
    fps_out: float | None = Field(None, gt=0, le=240)
    hardware: bool = False
    transform_strategy: str | None = None
    phash_attack: bool = False
    phash_epsilon: float = Field(0.03, gt=0, le=1)
    phash_iters: int = Field(40, ge=1, le=1000)
    multi_hash_attack: bool = False
    dhash_attack: bool = False
    shot_retime: bool = False
    shot_retime_min: float = 0.98
    shot_retime_max: float = 1.04
    cut_jitter: int = 0
    audio_strong: bool = False
    echo_defeat: bool = False
    skip_vmaf: bool = False
    filter_scale: int = Field(0, ge=0, le=1080)
    rotate: float = Field(0.0, ge=0, le=10)
    median: int = Field(0, ge=0, le=9)
    noise: float = Field(0.0, ge=0, le=1)
    requant: int = Field(0, ge=0, le=256)
    dct_step: float = Field(0.0, ge=0, le=256)
    temporal_sub: float = Field(0.0, ge=0, le=4)
    fft_phase: float = Field(0.0, ge=0, le=1)
    dwt_detail: float = Field(0.0, ge=0, le=1)
    face_perturb: float = Field(0.0, ge=0, le=0.2)
    lpc_attack: float = Field(0.0, ge=0, le=1)
    copy_attack: float = Field(0.0, ge=0, le=0.2)
    native_temporal: bool = False
    quality_protect: bool = False
    psnr_target: float = Field(38.0, ge=10, le=60)
    ssim_target: float = Field(0.94, ge=0, le=1)
    jitter: float = Field(0.0, ge=0, le=1)
    perspective: float = Field(0.0, ge=0, le=1)
    warp: float = Field(0.0, ge=0, le=1)
    purify_strength: float = Field(0.0, ge=0, le=1)
    # 潜空间净化：strength 只作开关（>0 启用），实际强度由边缘/带宽决定。
    purify_detail: float = Field(TIER_DETAIL, ge=0, le=1)
    # 细节回注带宽：水印在低频、字幕纹理在中频。0 = 自动（按帧长边换算，
    # 512p→1.3、1080p→2.7），>0 为专家手动指定的像素值。
    purify_detail_sigma: float = Field(TIER_DETAIL_SIGMA, ge=0, le=4.0)
    # B 档画质优先：把回注带宽推到字幕笔画尺度（1080p≈6.5），字幕可读，
    # 代价是水印部分回流。
    purify_detail_wide: bool = _SCHEMA_TIER.detail_wide
    # 已知来源方案（可空）：画像据此限制清晰度档位——亮度类必须压到 192，
    # 色度类 256 就够（research §19.7）。
    known_scheme: Literal[
        "", "luma", "chroma", "videoseal", "pixelseal", "wam", "trustmark", "mbrs"
    ] = ""
    purify_temporal: float = Field(0.0, ge=0, le=1)
    purify_max_edge: int = Field(_SCHEMA_TIER.max_edge, ge=0, le=2048)
    purify_batch: int = Field(_SCHEMA_TIER.batch, ge=1, le=16)
    embedding_attack: Literal["", "auto", "luma", "chroma", "both"] = ""
    embedding_strength: float = Field(0.0, ge=0, le=1)
    embedding_variant: Literal["legacy", "v2"] = "v2"
    embedding_aggressive: bool = False
    auto_profile: bool = False


class DesensitizeRequest(DesensitizeOptions):
    """直接调用 /api/desensitize 的请求体：在选项集之上补充路径。"""

    path: str
    output: str


class CollusionRequest(BaseModel):
    """共谋平均请求：同一内容的多份不同水印副本。"""

    model_config = ConfigDict(extra="forbid")

    paths: list[str] = Field(min_length=2, max_length=32)
    output: str
    mode: Literal["mean", "median"] = "mean"
    max_frames: int = Field(600, ge=2, le=6000)


TemplateCodec = Literal["H.264", "H.265"]


class TemplatePayload(BaseModel):
    """模板持久化 payload：camelCase 是 UI/DB 既有线格式。

    extra=ignore 容忍旧版本迁移遗留字段（level/restruct/perturb/audio 等），
    只约束与校验当前面板真实使用的字段。
    """

    model_config = ConfigDict(extra="ignore")

    audioRemix: bool = False
    echoDefeat: bool = False
    antiReembed: bool = False
    rotate: float = 0.0
    hashAttack: bool = False
    hashEpsilon: float = 0.045
    hashMode: str = "joint"
    requant: int = 0
    noise: float = 0.0
    dctStep: float = 0.0
    audioStrong: bool = False
    regradeOn: bool = False
    recropOn: bool = False
    temporalSub: float = 0.0
    fftPhase: float = 0.0
    dwtDetail: float = 0.0
    jitter: float = 0.0
    perspective: float = 0.0
    warp: float = 0.0
    facePerturb: float = 0.0
    lpcAttack: float = 0.0
    copyAttack: float = 0.0
    nativeTemporal: bool = False
    qualityProtect: bool = False
    psnrTarget: float = 38.0
    ssimTarget: float = 0.94
    purifyStrength: float = Field(0.0, ge=0, le=1)
    purifyDetail: float = Field(TIER_DETAIL, ge=0, le=1)
    purifyDetailSigma: float = Field(TIER_DETAIL_SIGMA, ge=0, le=4.0)
    purifyDetailWide: bool = _SCHEMA_TIER.detail_wide
    knownScheme: Literal[
        "", "luma", "chroma", "videoseal", "pixelseal", "wam", "trustmark", "mbrs"
    ] = ""
    purifyTemporal: float = Field(0.0, ge=0, le=1)
    purifyMaxEdge: int = Field(_SCHEMA_TIER.max_edge, ge=0, le=2048)
    purifyBatch: int = Field(_SCHEMA_TIER.batch, ge=1, le=16)
    embeddingAttack: Literal["", "auto", "luma", "chroma", "both"] = ""
    embeddingStrength: float = Field(0.0, ge=0, le=1)
    embeddingVariant: Literal["legacy", "v2"] = "v2"
    embeddingAggressive: bool = False
    autoProfile: bool = False
    sharpness: bool = False
    colorRestore: bool = False
    denoise: bool = False
    spoof: bool = False
    codec: TemplateCodec = "H.264"
    lossless: bool = False
    resolution: str = "保持原始分辨率"
    bitrate: int | None = None
    gop: int | None = None
    fpsOut: float | None = None
