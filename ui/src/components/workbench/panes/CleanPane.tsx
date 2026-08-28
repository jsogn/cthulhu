import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { TabsContent } from "@/components/ui/tabs";
import type { TemplateInfo } from "@/lib/backend";
import { FIXED_BRIGHTNESS, FIXED_CROP, FIXED_GAMMA } from "@/lib/cleanOptions";
import type { AntiLevel } from "@/lib/templates";
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
    antiLevel,
    setAntiLevel,
    audioClean,
    setAudioClean,
    echoDefeat,
    setEchoDefeat,
    antiReembed,
    setAntiReembed,
    recropOn,
    setRecropOn,
    regradeOn,
    setRegradeOn,
    detailProtectOn,
    setDetailProtectOn,
    sharpness,
    setSharpness,
    colorFix,
    setColorFix,
    aiDenoise,
    setAiDenoise,
    spoof,
    setSpoof,
    codec,
    setCodec,
    lossless,
    setLossless,
    resolution,
    setResolution,
  } = useCleanPanel();

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
            <div className="field">
              <span className="field-label">指纹对抗强度</span>
              <Select
                value={antiLevel}
                onValueChange={(value) => setAntiLevel(value as AntiLevel)}
              >
                <SelectTrigger className="form-input h-9 w-full">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="关闭">关闭 · 不启用指纹对抗</SelectItem>
                  <SelectItem value="轻度">轻度 · 低成本近无损</SelectItem>
                  <SelectItem value="标准">标准 · 均衡，哈希层打满</SelectItem>
                  <SelectItem value="强力">强力 · 轻微模糊，可能有轻微闪烁</SelectItem>
                  <SelectItem value="全兵器">全兵器 · 研究用，明显伪影</SelectItem>
                </SelectContent>
              </Select>
              <div className="form-help">
                档位越高对经典哈希破坏越充分，耗时与模糊逐档上升；语义级指纹各档提升有限。本地代理口径，仅供参考。
              </div>
            </div>
            <div className="switch">
              <div>
                <div className="switch-label">同步处理音频指纹</div>
                <div className="switch-desc">对音轨做等长频谱轻处理，不影响音画同步</div>
              </div>
              <Switch checked={audioClean} onCheckedChange={setAudioClean} />
            </div>
            <div className="switch">
              <div>
                <div className="switch-label">音频回声扰动</div>
                <div className="switch-desc">同步放慢约 3% 并保音调</div>
              </div>
              <Switch checked={echoDefeat} onCheckedChange={setEchoDefeat} />
            </div>
            <div className="switch">
              <div>
                <div className="switch-label">抗二次检测增强</div>
                <div className="switch-desc">扰动中频 DCT 系数，破坏二次嵌入</div>
              </div>
              <Switch checked={antiReembed} onCheckedChange={setAntiReembed} />
            </div>
            <div className="switch">
              <div>
                <div className="switch-label">重新构图（裁剪回缩）</div>
                <div className="switch-desc">
                  裁左/下各 {((FIXED_CROP * 100).toFixed(1))}%，顶部与右侧保留，静态缩放微模糊
                </div>
              </div>
              <Switch checked={recropOn} onCheckedChange={setRecropOn} />
            </div>
            <div className="switch">
              <div>
                <div className="switch-label">调光微扰</div>
                <div className="switch-desc">
                  逐帧伽马/亮度微调（γ±{FIXED_GAMMA.toFixed(2)} · 亮度±{FIXED_BRIGHTNESS.toFixed(3)}），对抗时域相关信号
                </div>
              </div>
              <Switch checked={regradeOn} onCheckedChange={setRegradeOn} />
            </div>
            <div className="switch">
              <div>
                <div className="switch-label">细节保护（人脸/字幕/纹理）</div>
                <div className="switch-desc">保护区域回退原帧，保留部分水印特征</div>
              </div>
              <Switch checked={detailProtectOn} onCheckedChange={setDetailProtectOn} />
            </div>

            <div className="section-title">画质优化</div>
            <div className="switch">
              <div>
                <div className="switch-label">锐度补偿</div>
                <div className="switch-desc">抵消清除算法带来的轻微模糊</div>
              </div>
              <Switch checked={sharpness} onCheckedChange={setSharpness} />
            </div>
            <div className="switch">
              <div>
                <div className="switch-label">色彩还原</div>
                <div className="switch-desc">把处理后亮度均值校准回原片</div>
              </div>
              <Switch checked={colorFix} onCheckedChange={setColorFix} />
            </div>
            <div className="switch">
              <div>
                <div className="switch-label">空间降噪</div>
                <div className="switch-desc">破坏空域扩频水印</div>
              </div>
              <Switch checked={aiDenoise} onCheckedChange={setAiDenoise} />
            </div>
            <div className="switch">
              <div>
                <div className="switch-label">伪水印注入（溯源干扰）</div>
                <div className="switch-desc">注入随机干扰水印，轻微损失画质</div>
              </div>
              <Switch checked={spoof} onCheckedChange={setSpoof} />
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
