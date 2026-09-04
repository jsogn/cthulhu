import type { ReactNode } from "react";

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
import type { TemplateInfo } from "@/lib/backend";
import { FIXED_CROP } from "@/lib/cleanOptions";
import { cn } from "@/lib/utils";
import { useCleanPanel } from "@/stores/cleanPanel";

interface CleanPaneProps {
  templateList: TemplateInfo[];
  applyTemplateById: (id: string) => void;
  cleanSubmitting: boolean;
  runClean: () => void;
}

export function CleanPane({
  templateList,
  applyTemplateById,
  cleanSubmitting,
  runClean,
}: CleanPaneProps) {
  const {
    templateId,
    outputMode,
    setOutputMode,
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
    codec,
    setCodec,
    lossless,
    setLossless,
    resolution,
    setResolution,
  } = useCleanPanel();

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
          <div className="field">
            <span className="field-label">清洗方式</span>
            <div className="inline-flex w-full rounded-lg bg-muted/50 p-0.5">
              <button
                type="button"
                onClick={() => setOutputMode("reencode")}
                aria-pressed={outputMode === "reencode"}
                className={cn(
                  "flex-1 rounded-md px-3 py-1.5 text-sm font-medium transition-colors",
                  outputMode === "reencode"
                    ? "bg-background text-foreground shadow-sm"
                    : "text-muted-foreground hover:text-foreground",
                )}
              >
                重新编码
              </button>
              <button
                type="button"
                onClick={() => setOutputMode("remux")}
                aria-pressed={outputMode === "remux"}
                className={cn(
                  "flex-1 rounded-md px-3 py-1.5 text-sm font-medium transition-colors",
                  outputMode === "remux"
                    ? "bg-background text-foreground shadow-sm"
                    : "text-muted-foreground hover:text-foreground",
                )}
              >
                重新封装
              </button>
            </div>
            <div className="form-help">
              {outputMode === "remux"
                ? "仅重写封装格式与元数据，画面音轨原样保留、画质无损，速度最快，但不会清除内容指纹"
                : "重新解码并压缩画面与音轨，下方对抗设置真正生效，清洗更彻底"}
            </div>
          </div>
          <div
            className={`flex min-w-0 flex-col gap-2.5${
              outputMode === "remux" ? " pointer-events-none select-none opacity-45" : ""
            }`}
          >
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

            <div className="section-title">再生重写</div>
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
            <div className="switch">
              <div>
                <div className="switch-label">DCT 系数扰动（重量化+中频）</div>
                <div className="switch-desc">8×8 DCT 重量化并扰动中频系数</div>
                <WeaponTags layer="wm" cost="mid" quality="mild" />
              </div>
              <Switch checked={dctOn} onCheckedChange={setDctOn} />
            </div>
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
