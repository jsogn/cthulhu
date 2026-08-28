"""API 与队列共享的请求契约。

DesensitizeOptions 是清洗选项的唯一权威定义：直接端点、任务队列与
前端生成类型都从这里派生，避免 Python/TS 两侧手写副本漂移。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class DesensitizeOptions(BaseModel):
    """清洗选项集：snake_case 为唯一字段名，未知字段直接报错。"""

    model_config = ConfigDict(extra="forbid")

    output_mode: str = "reencode"
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
    phash_iters: int = Field(120, ge=1, le=1000)
    multi_hash_attack: bool = False
    shot_retime: bool = False
    shot_retime_min: float = 0.98
    shot_retime_max: float = 1.04
    cut_jitter: int = 0
    audio_strong: bool = False
    echo_defeat: bool = False
    skip_vmaf: bool = False
    filter_scale: int = Field(0, ge=0, le=1080)
    rotate: float = Field(0.0, ge=0, le=10)
    transcode_chain: bool = False
    median: int = Field(0, ge=0, le=9)
    noise: float = Field(0.0, ge=0, le=1)
    requant: int = Field(0, ge=0, le=256)
    dct_step: float = Field(0.0, ge=0, le=256)
    drop_every: int = Field(0, ge=0, le=1000)
    jitter: float = Field(0.0, ge=0, le=1)
    perspective: float = Field(0.0, ge=0, le=1)
    warp: float = Field(0.0, ge=0, le=1)
    mirror: bool = False
    subtract_beta: float = Field(0.0, ge=0, le=4)
    saliency: int = Field(0, ge=0, le=4)
    chroma_levels: int = Field(0, ge=0, le=256)
    native_filters: bool = False
    detail_protect: float = Field(0.0, ge=0, le=1)


class DesensitizeRequest(DesensitizeOptions):
    """直接调用 /api/desensitize 的请求体：在选项集之上补充路径。"""

    path: str
    output: str


AntiLevel = Literal["关闭", "轻度", "标准", "强力", "全兵器"]
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
    anti: AntiLevel = "关闭"
    regradeOn: bool = False
    recropOn: bool = False
    detailProtectOn: bool = False
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
