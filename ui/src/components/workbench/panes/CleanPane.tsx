import { useEffect, useState, type ReactNode } from "react";

import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { TabsContent } from "@/components/ui/tabs";
import {
  WeaponTags,
  type CostKind,
  type LayerKind,
  type QualityKind,
} from "@/components/workbench/WeaponTags";
import {
  installPurify,
  purifyStatus,
  type PurifyStatus,
  type TemplateInfo,
} from "@/lib/backend";
import { FIXED_CROP } from "@/lib/cleanOptions";
import { useCleanPanel } from "@/stores/cleanPanel";
import type { Material } from "@/stores/materials";

interface CleanPaneProps {
  material: Material | null;
  templateList: TemplateInfo[];
  applyTemplateById: (id: string) => void;
  cleanSubmitting: boolean;
  runClean: () => void;
}

const formatBytes = (bytes: number) => {
  if (!bytes) return "0 MB";
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(2)} GB`;
  return `${Math.round(bytes / 1e6)} MB`;
};

export function CleanPane({
  material,
  templateList,
  applyTemplateById,
  cleanSubmitting,
  runClean,
}: CleanPaneProps) {
  const {
    templateId,
    audioClean,
    setAudioClean,
    echoDefeat,
    setEchoDefeat,
    rotateOn,
    setRotateOn,
    rotate,
    setRotate,
    hashOn,
    setHashOn,
    hashEps,
    setHashEps,
    hashMode,
    setHashMode,
    requantOn,
    setRequantOn,
    requant,
    setRequant,
    noiseOn,
    setNoiseOn,
    noise,
    setNoise,
    dctOn,
    setDctOn,
    dctStep,
    setDctStep,
    audioStrongOn,
    setAudioStrongOn,
    recropOn,
    setRecropOn,
    regradeOn,
    setRegradeOn,
    sharpness,
    setSharpness,
    colorFix,
    setColorFix,
    aiDenoise,
    setAiDenoise,
    spoof,
    setSpoof,
    temporalSubOn,
    setTemporalSubOn,
    temporalSub,
    setTemporalSub,
    nativeTemporalOn,
    setNativeTemporalOn,
    fftOn,
    setFftOn,
    fftPhase,
    setFftPhase,
    fftMag,
    setFftMag,
    dwtDetailOn,
    setDwtDetailOn,
    dwtDetail,
    setDwtDetail,
    warpOn,
    setWarpOn,
    warp,
    setWarp,
    shearOn,
    setShearOn,
    shear,
    setShear,
    jitterOn,
    setJitterOn,
    jitter,
    setJitter,
    nonintOn,
    setNonintOn,
    nonintRatio,
    setNonintRatio,
    flowOn,
    setFlowOn,
    flow,
    setFlow,
    textureOn,
    setTextureOn,
    texture,
    setTexture,
    multiscaleOn,
    setMultiscaleOn,
    multiscale,
    setMultiscale,
    faceOn,
    setFaceOn,
    face,
    setFace,
    temporalBlurOn,
    setTemporalBlurOn,
    temporalBlur,
    setTemporalBlur,
    lpcOn,
    setLpcOn,
    lpc,
    setLpc,
    neuralOn,
    setNeuralOn,
    neural,
    setNeural,
    qualityProtectOn,
    setQualityProtectOn,
    psnrTarget,
    setPsnrTarget,
    ssimTarget,
    setSsimTarget,
    purifyOn,
    setPurifyOn,
    purifyTemporal,
    setPurifyTemporal,
    purifyMaxEdge,
    setPurifyMaxEdge,
    purifyDetailWide,
    setPurifyDetailWide,
    setPurifyBatch,
    embeddingOn,
    setEmbeddingOn,
    embeddingAttack,
    setEmbeddingAttack,
    embeddingStrength,
    setEmbeddingStrength,
    embeddingVariant,
    setEmbeddingVariant,
    embeddingAggressive,
    setEmbeddingAggressive,
    autoProfile,
    setAutoProfile,
    codec,
    setCodec,
    lossless,
    setLossless,
    resolution,
    setResolution,
  } = useCleanPanel();

  const [purifyState, setPurifyState] = useState<PurifyStatus | null>(null);
  const [purifyError, setPurifyError] = useState<string | null>(null);
  const [purifyBusy, setPurifyBusy] = useState(false);

  useEffect(() => {
    let active = true;
    void purifyStatus()
      .then((status) => {
        if (active) {
          setPurifyState(status);
          setPurifyError(null);
        }
      })
      .catch((error: unknown) => {
        if (active) {
          setPurifyError(error instanceof Error ? error.message : "净化状态查询失败");
        }
      });
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    if (purifyState?.state !== "downloading") return;
    const timer = window.setInterval(() => {
      void purifyStatus()
        .then(setPurifyState)
        .catch(() => {
          // 轮询失败保持上次进度，下一次继续尝试。
        });
    }, 2000);
    return () => window.clearInterval(timer);
  }, [purifyState?.state]);

  const handleInstallPurify = async () => {
    setPurifyBusy(true);
    setPurifyError(null);
    try {
      setPurifyState(await installPurify());
    } catch (error) {
      setPurifyError(error instanceof Error ? error.message : "模型下载启动失败");
    } finally {
      setPurifyBusy(false);
    }
  };

  const purifyPercent = Math.round((purifyState?.progress ?? 0) * 100);
  const purifyDownloading = purifyState?.state === "downloading";
  // 只有一份权重：内置的 10MB TAESD（sd-turbo 与扩散引擎已下线）。
  const purifyReady = purifyState?.latent?.cached ?? false;
  const purifyPerformance =
    purifyMaxEdge === 192 && !purifyDetailWide
      ? "extreme"
      : purifyMaxEdge === 512 && purifyDetailWide
        ? "quality"
        : "fast";
  const applyPurifyPerformance = (value: string) => {
    if (value === "extreme") {
      setPurifyMaxEdge(192);
      setPurifyBatch(8);
      setPurifyDetailWide(false);
    } else if (value === "quality") {
      setPurifyMaxEdge(512);
      setPurifyBatch(8);
      setPurifyDetailWide(true);
    } else {
      setPurifyMaxEdge(256);
      setPurifyBatch(8);
      setPurifyDetailWide(false);
    }
  };
  const switchCard = (
    label: string,
    desc: ReactNode,
    layer: LayerKind,
    cost: CostKind,
    quality: QualityKind,
    checked: boolean,
    onCheckedChange: (value: boolean) => void,
    body?: ReactNode,
  ) => (
    <div className="switch-card">
      <div className="switch">
        <div>
          <div className="switch-label">{label}</div>
          <div className="switch-desc">{desc}</div>
          <WeaponTags layer={layer} cost={cost} quality={quality} />
        </div>
        <Switch checked={checked} onCheckedChange={onCheckedChange} />
      </div>
      {body ? <div className="switch-body">{body}</div> : null}
    </div>
  );

  return (
    <TabsContent value="清洗去重" className="tab-pane">
      <ScrollArea className="min-h-0 flex-1">
        <div className="flex flex-col gap-2.5">
          {material ? (
            <>
              <div className="section-title">文件信息</div>
              <div className="kv-card">
                <div className="kv-row">
                  <span>文件名</span>
                  <b className="truncate">{material.name}</b>
                </div>
                <div className="kv-row">
                  <span>编码 / 分辨率</span>
                  <b>
                    {material.codec ?? "—"} · {material.res}
                  </b>
                </div>
                <div className="kv-row">
                  <span>帧率 / 时长</span>
                  <b>
                    {material.fps} · {material.dur}
                  </b>
                </div>
                <div className="kv-row">
                  <span>文件大小</span>
                  <b>{material.size}</b>
                </div>
              </div>
            </>
          ) : null}
          <div className="field">
            <span className="field-label">模板</span>
            <Select value={templateId} onValueChange={applyTemplateById}>
              <SelectTrigger className="form-input h-8 w-full">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="manual">不使用模板（手动）</SelectItem>
                {templateList.length === 0 ? (
                  <SelectItem value="__empty__" disabled>
                    还没有模板，可到「去重模板」页创建
                  </SelectItem>
                ) : (
                  templateList.map((item) => (
                    <SelectItem key={item.id} value={item.id}>
                      {item.name}
                    </SelectItem>
                  ))
                )}
              </SelectContent>
            </Select>
            <div className="form-help">选择后自动回填下方参数，可继续调整</div>
          </div>
          <div className="rounded-md bg-muted/50 px-3 py-2">
            <p className="note">
              覆盖空域、DCT 频域、时域、色度与音频回声等常见水印；未知方案建议加强对抗档。
            </p>
          </div>
          <div className="flex min-w-0 flex-col gap-2.5">
            <div className="section-title">去水印 · 画面重建</div>
            {switchCard(
              "画面重建（去暗水印）",
              "按画面内容重新生成一遍，抹掉嵌进去的暗水印；字幕与人脸是否清楚由下方档位决定",
              "wm",
              "fast",
              "mild",
              purifyOn,
              setPurifyOn,
              purifyOn ? (
                <>
                  <div className="field">
                    <span className="field-label">清晰度档位</span>
                    <Select
                      value={purifyPerformance}
                      onValueChange={applyPurifyPerformance}
                    >
                      <SelectTrigger className="form-input h-8 w-full">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="quality">画质优先（默认）</SelectItem>
                        <SelectItem value="fast">平衡</SelectItem>
                        <SelectItem value="extreme">清除优先</SelectItem>
                      </SelectContent>
                    </Select>
                    <div className="form-help">
                      {purifyPerformance === "quality"
                        ? "画面清晰、人脸与字幕正常；水印清除较弱，适合要成片质量的场景"
                        : purifyPerformance === "fast"
                          ? "画面略软，清除率比画质优先更好，是人人都能接受的中间档"
                          : "水印清除最彻底；画面会明显变软、字幕可能难以辨认"}
                    </div>
                  </div>
                  <div className="form-help">
                    {purifyDetailWide
                      ? "细节带宽按分辨率自动放大到字幕笔画尺度（1080p≈6.5），字幕可读、水印会部分回流"
                      : "细节带宽按分辨率自动换算（1080p≈2.6），清除率优先；要更清晰的画面请选「画面优先 · 512」"}
                    ，并自动做一次轻度锐化
                  </div>
                  <div className="field">
                    <span className="field-label">
                      时序一致性减法 {purifyTemporal.toFixed(2)}
                    </span>
                    <Slider
                      value={[purifyTemporal]}
                      min={0}
                      max={1}
                      step={0.05}
                      onValueChange={(values) => setPurifyTemporal(values[0] ?? 0)}
                    />
                  </div>
                </>
              ) : undefined,
            )}
            {/* 开箱即用：模型随包内置，就绪时不显示任何状态提示；
                只有缺失/下载中/出错/运行时不可用等异常状态才展示。 */}
            {!purifyReady ? (
              <div className="rounded-md bg-muted/50 px-3 py-2">
                <div className="flex items-center justify-between gap-2">
                  <span className="text-xs text-muted-foreground">
                    {purifyDownloading
                      ? `模型下载中 ${purifyPercent}% · ${formatBytes(
                          purifyState?.downloaded_bytes ?? 0,
                        )} / ${formatBytes(purifyState?.total_bytes ?? 0)}`
                      : purifyState?.state === "unavailable"
                        ? `运行时不可用：${purifyState.reason}`
                        : purifyState?.state === "error"
                          ? `模型安装失败：${purifyState.error ?? "未知错误"}`
                          : purifyState?.latent?.bundled
                            ? "潜空间模型未就绪"
                            : "潜空间模型未内置（开发环境可下载约 10MB）"}
                  </span>
                  {!purifyDownloading &&
                  purifyState?.available &&
                  purifyState?.allow_download ? (
                    <Button
                      size="sm"
                      variant="secondary"
                      disabled={purifyBusy}
                      onClick={() => void handleInstallPurify()}
                    >
                      {purifyBusy ? "启动中…" : "下载模型"}
                    </Button>
                  ) : null}
                </div>
                {purifyDownloading ? (
                  <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-muted">
                    <div
                      className="h-full bg-primary transition-all"
                      style={{ width: `${purifyPercent}%` }}
                    />
                  </div>
                ) : null}
                {purifyError ? (
                  <div className="mt-1 text-xs text-destructive">{purifyError}</div>
                ) : null}
                {purifyState && !purifyState.allow_download ? (
                  <div className="mt-1 text-xs text-muted-foreground">
                    已禁用自动下载，可用 CTHULHU_PURIFY_MODEL_DIR 预置权重
                  </div>
                ) : null}
              </div>
            ) : null}
            {switchCard(
              "嵌入域定向重写",
              "按水印画像改写低频：VideoSeal 类打 Y 亮度，WAM 类打 Cb/Cr 色度",
              "wm",
              "fast",
              "mild",
              embeddingOn,
              setEmbeddingOn,
              embeddingOn ? (
                <>
                  <div className="field">
                    <span className="field-label">目标域</span>
                    <Select
                      value={embeddingAttack}
                      onValueChange={(value) =>
                        setEmbeddingAttack(value as "luma" | "chroma" | "both")
                      }
                    >
                      <SelectTrigger className="form-input h-8 w-full">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="auto">自动（已知方案按画像，未知 both）</SelectItem>
                        <SelectItem value="luma">Y 亮度低频（VideoSeal 类）</SelectItem>
                        <SelectItem value="chroma">Cb/Cr 色度低频（WAM 类）</SelectItem>
                        <SelectItem value="both">两者同时</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="field">
                    <span className="field-label">重写强度 {embeddingStrength.toFixed(2)}</span>
                    <Slider
                      value={[embeddingStrength]}
                      min={0.1}
                      max={1}
                      step={0.05}
                      onValueChange={(values) => setEmbeddingStrength(values[0] ?? 0.25)}
                    />
                  </div>
                  <div className="switch">
                    <div>
                      <div className="switch-label">精确频带重写</div>
                      <div className="switch-desc">
                        按实测水印频带精确改写低频并加随机量化；关闭则退回粗略的高斯平滑替换
                      </div>
                    </div>
                    <Switch
                      checked={embeddingVariant === "v2"}
                      onCheckedChange={(checked) =>
                        setEmbeddingVariant(checked ? "v2" : "legacy")
                      }
                    />
                  </div>
                  <div className="switch">
                    <div>
                      <div className="switch-label">增强重写（更强，画质代价更高）</div>
                      <div className="switch-desc">
                        破坏固定分块对齐，并覆盖色度子采样网格；画质代价更高
                      </div>
                    </div>
                    <Switch
                      checked={embeddingAggressive}
                      onCheckedChange={setEmbeddingAggressive}
                    />
                  </div>
                </>
              ) : undefined,
            )}
            <div className="switch">
              <div>
                <div className="switch-label">自动画像（内容复杂度自适应）</div>
                <div className="switch-desc">
                  按镜头纹理/运动/时序一致性自适应净化强度与时序减法；画面档位（边缘、
                  细节带宽、字幕增强）按预设执行，不参与自适应
                </div>
                <WeaponTags layer="quality" cost="fast" quality="none" />
              </div>
              <Switch checked={autoProfile} onCheckedChange={setAutoProfile} />
            </div>

            <div className="section-title">画面变换 · 去同步与色彩微扰</div>
            <div className="switch">
              <div>
                <div className="switch-label">重新构图（裁剪回缩）</div>
                <div className="switch-desc">
                  裁左/下各 {((FIXED_CROP * 100).toFixed(1))}%，顶部与右侧保留。对暗水印无效，用于判重指纹；文字位置不固定，可能裁到字幕，仅确认边缘无文字时使用
                </div>
                <WeaponTags layer="fp" cost="fast" quality="heavy" />
              </div>
              <Switch checked={recropOn} onCheckedChange={setRecropOn} />
            </div>
            {switchCard(
              "几何微旋转",
              "低频小幅旋转去同步，破坏哈希与块对齐",
              "dual",
              "fast",
              "mild",
              rotateOn,
              setRotateOn,
              rotateOn ? (
                <div className="field">
                  <span className="field-label">旋转振幅 {rotate.toFixed(1)}°</span>
                  <Slider
                    value={[rotate]}
                    min={0.1}
                    max={2.5}
                    step={0.1}
                    onValueChange={(values) => setRotate(values[0] ?? 0.2)}
                  />
                </div>
              ) : undefined,
            )}
            {switchCard(
              "局部平滑扭曲",
              "粗网格位移场上采样成平滑光流逐帧重采样，破坏空间对齐",
              "wm",
              "mid",
              "mild",
              warpOn,
              setWarpOn,
              warpOn ? (
                <div className="field">
                  <span className="field-label">位移跨度 {(warp * 100).toFixed(1)}% 短边</span>
                  <Slider
                    value={[warp]}
                    min={0.001}
                    max={0.01}
                    step={0.001}
                    onValueChange={(values) => setWarp(values[0] ?? 0.005)}
                  />
                </div>
              ) : undefined,
            )}
            {switchCard(
              "透视剪切",
              "逐帧轻微梯形畸变，破坏块对齐与几何同步",
              "wm",
              "mid",
              "mild",
              shearOn,
              setShearOn,
              shearOn ? (
                <div className="field">
                  <span className="field-label">剪切强度 {shear.toFixed(3)}</span>
                  <Slider
                    value={[shear]}
                    min={0.005}
                    max={0.03}
                    step={0.005}
                    onValueChange={(values) => setShear(values[0] ?? 0.01)}
                  />
                </div>
              ) : undefined,
            )}
            {switchCard(
              "平移抖动",
              "低频正弦漂移（相邻帧 ≤1px），破坏逐帧对齐，观感轻微",
              "wm",
              "fast",
              "mild",
              jitterOn,
              setJitterOn,
              jitterOn ? (
                <div className="field">
                  <span className="field-label">抖动幅度 {(jitter * 100).toFixed(1)}%</span>
                  <Slider
                    value={[jitter]}
                    min={0.005}
                    max={0.03}
                    step={0.005}
                    onValueChange={(values) => setJitter(values[0] ?? 0.005)}
                  />
                </div>
              ) : undefined,
            )}
            <div className="switch">
              <div>
                <div className="switch-label">色彩微扰（伽马/亮度+色相）</div>
                <div className="switch-desc">
                  逐帧伽马/亮度微调并叠加 ±6° 色相抖动，对抗时域与色彩描述子
                </div>
                <WeaponTags layer="dual" cost="fast" quality="mild" />
              </div>
              <Switch checked={regradeOn} onCheckedChange={setRegradeOn} />
            </div>

            <div className="section-title">音频处理</div>
            <div className="switch">
              <div>
                <div className="switch-label">同步处理音频指纹</div>
                <div className="switch-desc">对音轨做等长频谱轻处理，不影响音画同步</div>
                <WeaponTags layer="audio" cost="fast" quality="none" />
              </div>
              <Switch checked={audioClean} onCheckedChange={setAudioClean} />
            </div>
            <div className="switch">
              <div>
                <div className="switch-label">音频回声扰动</div>
                <div className="switch-desc">同步放慢约 3% 并保音调</div>
                <WeaponTags layer="audio" cost="fast" quality="mild" />
              </div>
              <Switch checked={echoDefeat} onCheckedChange={setEchoDefeat} />
            </div>
            {switchCard(
              "LPCAA 音频攻击",
              "LPC 残差白化，破坏语音类指纹 · 高强可听出改变",
              "audio",
              "fast",
              "heavy",
              lpcOn,
              setLpcOn,
              lpcOn ? (
                <div className="field">
                  <span className="field-label">白化强度 {lpc.toFixed(2)}</span>
                  <Slider
                    value={[lpc]}
                    min={0.2}
                    max={1}
                    step={0.05}
                    onValueChange={(values) => setLpc(values[0] ?? 0.85)}
                  />
                </div>
              ) : undefined,
            )}
            <div className="switch">
              <div>
                <div className="switch-label">强音频重混</div>
                <div className="switch-desc">变调+EQ 倾斜+底噪，配合音频指纹开关生效</div>
                <WeaponTags layer="audio" cost="fast" quality="heavy" />
              </div>
              <Switch checked={audioStrongOn} onCheckedChange={setAudioStrongOn} />
            </div>

            <div className="section-title">重编码与信号扰动</div>
            {switchCard(
              "哈希签名对抗（pHash/dHash）",
              "签名域可微扰动，翻转感知哈希符号位（专攻模式效果更强）",
              "fp",
              "mid",
              "heavy",
              hashOn,
              setHashOn,
              hashOn ? (
                <>
                  <div className="field">
                    <span className="field-label">目标模式</span>
                    <Select
                      value={hashMode}
                      onValueChange={(value) =>
                        setHashMode(value as "phash" | "dhash" | "joint")
                      }
                    >
                      <SelectTrigger className="form-input h-8 w-full"><SelectValue /></SelectTrigger>
                      <SelectContent>
                        <SelectItem value="phash">pHash 专攻（推荐）</SelectItem>
                        <SelectItem value="dhash">dHash 专攻</SelectItem>
                        <SelectItem value="joint">联合（pHash+dHash）</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="field">
                    <span className="field-label">扰动预算 ε {hashEps.toFixed(3)}</span>
                    <Slider
                      value={[hashEps]}
                      min={0.02}
                      max={0.12}
                      step={0.01}
                      onValueChange={(values) => setHashEps(values[0] ?? 0.08)}
                    />
                  </div>
                </>
              ) : undefined,
            )}
            {switchCard(
              "像素重量化",
              "把像素压缩到有限级数，破坏低位与细微扰动",
              "wm",
              "fast",
              "none",
              requantOn,
              setRequantOn,
              requantOn ? (
                <div className="field">
                  <span className="field-label">量化级数 {requant}</span>
                  <Slider
                    value={[requant]}
                    min={32}
                    max={96}
                    step={8}
                    onValueChange={(values) => setRequant(values[0] ?? 64)}
                  />
                </div>
              ) : undefined,
            )}
            {switchCard(
              "微噪声",
              "加性微高斯噪声，破坏扩频水印的低位相关",
              "wm",
              "fast",
              "none",
              noiseOn,
              setNoiseOn,
              noiseOn ? (
                <div className="field">
                  <span className="field-label">噪声强度 {noise.toFixed(3)}</span>
                  <Slider
                    value={[noise]}
                    min={0.001}
                    max={0.01}
                    step={0.001}
                    onValueChange={(values) => setNoise(values[0] ?? 0.004)}
                  />
                </div>
              ) : undefined,
            )}
            <div className="switch">
              <div>
                <div className="switch-label">空间降噪</div>
                <div className="switch-desc">removegrain 轻降噪，破坏扩频/QIM/DWT 水印</div>
                <WeaponTags layer="wm" cost="fast" quality="none" />
              </div>
              <Switch checked={aiDenoise} onCheckedChange={setAiDenoise} />
            </div>
            {switchCard(
              "非整缩重采样",
              "先放大再回压，打散像素网格与块对齐",
              "dual",
              "fast",
              "mild",
              nonintOn,
              setNonintOn,
              nonintOn ? (
                <div className="field">
                  <span className="field-label">缩放比例 {(nonintRatio * 100).toFixed(1)}%</span>
                  <Slider
                    value={[nonintRatio]}
                    min={0.005}
                    max={0.03}
                    step={0.005}
                    onValueChange={(values) => setNonintRatio(values[0] ?? 0.01)}
                  />
                </div>
              ) : undefined,
            )}
            {switchCard(
              "跨帧估计相减",
              "估计帧间固定水印并过减，会削弱静态字幕/背景边缘",
              "wm",
              "mid",
              "heavy",
              temporalSubOn,
              setTemporalSubOn,
              temporalSubOn ? (
                <>
                  <div className="field">
                    <span className="field-label">过减强度 β {temporalSub.toFixed(1)}</span>
                    <Slider
                      value={[temporalSub]}
                      min={0.2}
                      max={1.5}
                      step={0.1}
                      onValueChange={(values) => setTemporalSub(values[0] ?? 0.8)}
                    />
                  </div>
                  <div className="switch-inline">
                    <div>
                      <div className="switch-label">原生加速(ffmpeg)</div>
                      <div className="switch-desc">8bit 原生链,破坏力更强,不支持分镜重排</div>
                    </div>
                    <Switch
                      checked={nativeTemporalOn}
                      onCheckedChange={setNativeTemporalOn}
                    />
                  </div>
                </>
              ) : undefined,
            )}
            {switchCard(
              "FFT 扰动（相位+幅度）",
              "打散中高频相位并随机缩放幅值，覆盖 DFT 域两类水印",
              "wm",
              "mid",
              "heavy",
              fftOn,
              setFftOn,
              fftOn ? (
                <>
                  <div className="field">
                    <span className="field-label">相位强度 {fftPhase.toFixed(2)}</span>
                    <Slider
                      value={[fftPhase]}
                      min={0.1}
                      max={1}
                      step={0.05}
                      onValueChange={(values) => setFftPhase(values[0] ?? 0.5)}
                    />
                  </div>
                  <div className="field">
                    <span className="field-label">幅度强度 {fftMag.toFixed(2)}</span>
                    <Slider
                      value={[fftMag]}
                      min={0.1}
                      max={1}
                      step={0.05}
                      onValueChange={(values) => setFftMag(values[0] ?? 0.5)}
                    />
                  </div>
                </>
              ) : undefined,
            )}
            {switchCard(
              "小波细节带随机化",
              "随机化 Haar 对角线细节子带，破坏小波嵌入相关",
              "wm",
              "fast",
              "none",
              dwtDetailOn,
              setDwtDetailOn,
              dwtDetailOn ? (
                <div className="field">
                  <span className="field-label">随机化强度 {dwtDetail.toFixed(2)}</span>
                  <Slider
                    value={[dwtDetail]}
                    min={0.1}
                    max={1}
                    step={0.05}
                    onValueChange={(values) => setDwtDetail(values[0] ?? 0.8)}
                  />
                </div>
              ) : undefined,
            )}
            {switchCard(
              "DCT 系数扰动（重量化+中频）",
              "8×8 DCT 重量化并扰动中频系数",
              "wm",
              "mid",
              "mild",
              dctOn,
              setDctOn,
              dctOn ? (
                <div className="field">
                  <span className="field-label">量化步长 {dctStep}</span>
                  <Slider
                    value={[dctStep]}
                    min={1}
                    max={256}
                    step={1}
                    onValueChange={(values) => setDctStep(values[0] ?? 12)}
                  />
                </div>
              ) : undefined,
            )}
            <div className="switch">
              <div>
                <div className="switch-label">伪水印注入（溯源干扰）</div>
                <div className="switch-desc">注入随机干扰水印，对抗上传后二次嵌入 · 需投流实测</div>
                <WeaponTags layer="wm" cost="mid" quality="mild" />
              </div>
              <Switch checked={spoof} onCheckedChange={setSpoof} />
            </div>
            {switchCard(
              "光流一致性破坏",
              "按运动加权的平滑扰动重采样，改运动指纹",
              "fp",
              "fast",
              "heavy",
              flowOn,
              setFlowOn,
              flowOn ? (
                <div className="field">
                  <span className="field-label">扰动强度 {flow.toFixed(1)}px</span>
                  <Slider
                    value={[flow]}
                    min={0.5}
                    max={2.5}
                    step={0.1}
                    onValueChange={(values) => setFlow(values[0] ?? 2)}
                  />
                </div>
              ) : undefined,
            )}
            {switchCard(
              "纹理/复杂度注入",
              "向低纹理区注入纹理并拉平复杂度分布",
              "fp",
              "slow",
              "heavy",
              textureOn,
              setTextureOn,
              textureOn ? (
                <div className="field">
                  <span className="field-label">注入强度 {texture.toFixed(2)}</span>
                  <Slider
                    value={[texture]}
                    min={0.01}
                    max={0.05}
                    step={0.005}
                    onValueChange={(values) => setTexture(values[0] ?? 0.04)}
                  />
                </div>
              ) : undefined,
            )}
            {switchCard(
              "神经对抗",
              "黑盒攻击 DINOv2 判重描述子，压拷贝检测 embedding · 较慢",
              "fp",
              "slow",
              "heavy",
              neuralOn,
              setNeuralOn,
              neuralOn ? (
                <>
                  <div className="field">
                    <span className="field-label">扰动预算 ±{neural.toFixed(2)}</span>
                    <Slider
                      value={[neural]}
                      min={0.01}
                      max={0.08}
                      step={0.01}
                      onValueChange={(values) => setNeural(values[0] ?? 0.05)}
                    />
                  </div>
                </>
              ) : undefined,
            )}
            {switchCard(
              "多尺度特征扰动（DMFF）",
              "金字塔各尺度带限扰动，8/16/32 签名同步偏移",
              "fp",
              "slow",
              "heavy",
              multiscaleOn,
              setMultiscaleOn,
              multiscaleOn ? (
                <div className="field">
                  <span className="field-label">扰动强度 {multiscale.toFixed(2)}</span>
                  <Slider
                    value={[multiscale]}
                    min={0.005}
                    max={0.04}
                    step={0.005}
                    onValueChange={(values) => setMultiscale(values[0] ?? 0.02)}
                  />
                </div>
              ) : undefined,
            )}
            {switchCard(
              "人脸抗AI扰动",
              "仅在脸部区域注入扰动，破坏人脸识别特征",
              "face",
              "mid",
              "none",
              faceOn,
              setFaceOn,
              faceOn ? (
                <div className="field">
                  <span className="field-label">扰动强度 {face.toFixed(2)}</span>
                  <Slider
                    value={[face]}
                    min={0.01}
                    max={0.08}
                    step={0.01}
                    onValueChange={(values) => setFace(values[0] ?? 0.04)}
                  />
                </div>
              ) : undefined,
            )}
            {switchCard(
              "时序模糊",
              "帧间时域平滑，破坏逐帧匹配与帧差结构",
              "dual",
              "mid",
              "heavy",
              temporalBlurOn,
              setTemporalBlurOn,
              temporalBlurOn ? (
                <div className="field">
                  <span className="field-label">模糊强度 {temporalBlur.toFixed(2)}</span>
                  <Slider
                    value={[temporalBlur]}
                    min={0.05}
                    max={0.5}
                    step={0.05}
                    onValueChange={(values) => setTemporalBlur(values[0] ?? 0.25)}
                  />
                </div>
              ) : undefined,
            )}

            {/* 排序约定：画质优化固定在倒数第二分区，新增分区请插在它之前。 */}
            <div className="section-title">画质优化</div>
            {switchCard(
              "画质保护（PSNR/SSIM 门控）",
              `对攻击/重武器层按目标自动回退，PSNR≥${psnrTarget}dB、SSIM≥${ssimTarget}（原生滤镜链除外）`,
              "quality",
              "mid",
              "none",
              qualityProtectOn,
              setQualityProtectOn,
              qualityProtectOn ? (
                <>
                  <div className="field">
                    <span className="field-label">PSNR 目标 {psnrTarget} dB</span>
                    <Slider
                      value={[psnrTarget]}
                      min={32}
                      max={44}
                      step={1}
                      onValueChange={(values) => setPsnrTarget(values[0] ?? 38)}
                    />
                  </div>
                  <div className="field">
                    <span className="field-label">SSIM 目标 {ssimTarget.toFixed(2)}</span>
                    <Slider
                      value={[ssimTarget]}
                      min={0.8}
                      max={0.98}
                      step={0.01}
                      onValueChange={(values) => setSsimTarget(values[0] ?? 0.94)}
                    />
                  </div>
                </>
              ) : undefined,
            )}
            <div className="switch">
              <div>
                <div className="switch-label">锐度补偿</div>
                <div className="switch-desc">抵消清除算法带来的轻微模糊</div>
                <WeaponTags layer="quality" cost="fast" quality="none" />
              </div>
              <Switch checked={sharpness} onCheckedChange={setSharpness} />
            </div>
            <div className="switch">
              <div>
                <div className="switch-label">色彩还原</div>
                <div className="switch-desc">把处理后亮度均值校准回原片</div>
                <WeaponTags layer="quality" cost="fast" quality="none" />
              </div>
              <Switch checked={colorFix} onCheckedChange={setColorFix} />
            </div>
            <div className="section-title">输出编码 · MP4</div>
            <div className="field">
              <span className="field-label">视频编码</span>
              <Select value={codec} onValueChange={setCodec}>
                <SelectTrigger className="h-9 w-full"><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="H.264">H.264</SelectItem>
                  <SelectItem value="H.265">H.265</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="field">
              <span className="field-label">分辨率策略</span>
              <Select value={resolution} onValueChange={setResolution}>
                <SelectTrigger className="h-9 w-full"><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="保持原始分辨率">保持原始分辨率</SelectItem>
                  <SelectItem value="1920x1080">1920×1080</SelectItem>
                  <SelectItem value="1280x720">1280×720</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="switch">
              <div>
                <div className="switch-label">无损输出</div>
                <div className="switch-desc">极致保留画质，文件体积相应增大</div>
                <WeaponTags layer="quality" cost="fast" quality="none" />
              </div>
              <Switch checked={lossless} onCheckedChange={setLossless} />
            </div>
          </div>
        </div>
      </ScrollArea>
      <div className="shrink-0 pt-3">
        <Button
          variant="secondary"
          className="w-full"
          disabled={cleanSubmitting}
          onClick={runClean}
        >
          {cleanSubmitting ? "已加入队列" : "执行清洗"}
        </Button>
      </div>
    </TabsContent>
  );
}
