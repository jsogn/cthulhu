import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { TabsContent } from "@/components/ui/tabs";
import type { AudioAnalysis, DetectReport } from "@/lib/backend";
import { riskLabel } from "@/lib/workbenchOptions";
import type { AntiLevel } from "@/lib/templates";
import type { Material, RiskLevel } from "@/stores/materials";

interface DetectPaneProps {
  material: Material | null;
  report: DetectReport | null;
  risk: RiskLevel;
  score: number;
  detecting: boolean;
  detectSubmitting: boolean;
  audio: AudioAnalysis | null;
  audioMissing: boolean;
  antiLevel: AntiLevel;
  setAntiLevel: (value: AntiLevel) => void;
  onDetect: () => void;
}

export function DetectPane({
  material,
  report,
  risk,
  score,
  detecting,
  detectSubmitting,
  audio,
  audioMissing,
  antiLevel,
  setAntiLevel,
  onDetect,
}: DetectPaneProps) {
  const pct = risk === "待检测" ? 100 : Math.max(0, Math.min(100, score));
  const ringColor =
    risk === "待检测"
      ? "var(--border)"
      : risk === "有疑似特征"
        ? "var(--warn)"
        : "var(--ok)";
  const scoreSub = !report
    ? "尚未检测，点击上方「检测此素材」开始检测"
    : report.bitstream.flags.length > 0
      ? `码流评分 ${score} · 命中 ${report.bitstream.flags.length} 项疑似特征 · 建议差分复核`
      : `码流评分 ${score} · 未命中明显码流异常 · 建议差分复核`;

  return (
    <TabsContent value="检测参考" className="tab-pane">
      <ScrollArea className="h-full">
        <div className="flex flex-col gap-2.5">
          {material?.path && (
            <Button
              variant="secondary"
              size="sm"
              className="w-full"
              disabled={detecting || detectSubmitting}
              onClick={onDetect}
            >
              {detecting || detectSubmitting ? "检测中…" : report ? "重新检测" : "检测此素材"}
            </Button>
          )}
          {report ? (
            <>
              <Card className="flex items-center gap-3.5 border border-border p-3.5 shadow-none ring-0">
                <div
                  className="ring"
                  style={{
                    background: `conic-gradient(var(--warn) 0 ${Math.min(100, report.bitstream.score)}%, var(--muted) ${Math.min(100, report.bitstream.score)}% 100%)`,
                  }}
                >
                  <div className="ring-inner">
                    <span className="ring-score">{report.bitstream.score}</span>
                  </div>
                </div>
                <div>
                  <div className="score-title">
                    {riskLabel(risk)}
                  </div>
                  <div className="score-sub">码流层启发式检测 · 容器/SEI/压缩域联合</div>
                </div>
              </Card>

              <div className="section-title">文件信息</div>
              <div className="kv-card">
                <div className="kv-row">
                  <span>编码 / 分辨率</span>
                  <b>
                    {report.probe.codec} · {report.probe.width}×{report.probe.height}
                  </b>
                </div>
                <div className="kv-row">
                  <span>帧率 / 时长</span>
                  <b>
                    {report.probe.fps}fps · {report.probe.duration.toFixed(1)}s
                  </b>
                </div>
                <div className="kv-row">
                  <span>SEI 数量</span>
                  <b>{report.sei_count ?? "—"}</b>
                </div>
                <div className="kv-row">
                  <span>可疑元数据</span>
                  <b>
                    {report.container.suspicious?.length
                      ? report.container.suspicious.join("、")
                      : "无"}
                  </b>
                </div>
              </div>

              <div className="section-title">疑似命中项</div>
              {report.bitstream.flags.length > 0 ? (
                <ul className="suggest">
                  {report.bitstream.flags.map((flag) => (
                    <li key={flag}>{flag}</li>
                  ))}
                </ul>
              ) : (
                <p className="note">未发现明显码流异常。</p>
              )}

              {report.blind && (
                <>
                  <div className="section-title">盲检测（启发式 · 标定阈值）</div>
                  <div className="kv-card">
                    {(
                      [
                        ["空域扩频 SS", "ss"],
                        ["DCT-QIM 量化格", "qim"],
                        ["时域帧差", "temporal"],
                        ["小波 DWT", "dwt"],
                        ["色度残差", "chroma"],
                      ] as const
                    ).map(([label, key]) => {
                      const blind = report.blind;
                      if (!blind || typeof blind[key] !== "number") return null;
                      const confidence = report.blind_confidence?.[key];
                      const isHit = report.hits?.includes(key);
                      return (
                        <div className="kv-row" key={key}>
                          <span>
                            {label}
                            {isHit ? " ▲" : ""}
                          </span>
                          <b>
                            {blind[key].toFixed(2)}
                            {confidence !== undefined && (
                              <small> 置信 {confidence.toFixed(2)}</small>
                            )}
                          </b>
                        </div>
                      );
                    })}
                    {report.blind.echo !== undefined && (
                      <div className="kv-row">
                        <span>音频回声{report.hits?.includes("echo") ? " ▲" : ""}</span>
                        <b>{report.blind.echo.toFixed(2)}</b>
                      </div>
                    )}
                    {report.blind_structural && (
                      <>
                        <div className="kv-row">
                          <span>DCT 模运算（辅助）</span>
                          <b>{report.blind_structural.dctmod.toFixed(2)}</b>
                        </div>
                        <div className="kv-row">
                          <span>SVD 格点（辅助）</span>
                          <b>{report.blind_structural.svd.toFixed(2)}</b>
                        </div>
                      </>
                    )}
                  </div>
                  {report.hits && report.hits.length > 0 && (
                    <p className="note">
                      疑似命中（≥标定阈值且窗口置信度≥0.75）：{report.hits.join("、")}
                    </p>
                  )}
                  <p className="note">
                    分数 0~1、括号为窗口一致性置信度，研究口径；判定三要素：分数＋置信度＋干净同源差分。基于文件身份的平台标记（原件字节匹配）不在画面与码流中，本地无法复现。
                  </p>
                </>
              )}

              <div className="section-title">音频分析</div>
              {audio ? (
                <>
                  <div className="kv-row">
                    <span>回声隐藏置信度</span>
                    <b>{audio.echo_score.toFixed(2)}</b>
                  </div>
                  <svg
                    viewBox={`0 0 ${Math.max(1, audio.waveform.length - 1)} 40`}
                    preserveAspectRatio="none"
                    className="h-10 w-full"
                    aria-label="音轨波形"
                  >
                    <polyline
                      points={audio.waveform
                        .map((value, index) => `${index},${20 - value * 18}`)
                        .join(" ")}
                      fill="none"
                      stroke="var(--primary)"
                      strokeWidth="1.2"
                    />
                  </svg>
                  <div className="audio-spectrum" aria-label="对数频谱">
                    {audio.spectrum.map((value, index) => (
                      <span key={index} style={{ height: `${Math.max(4, value * 100)}%` }} />
                    ))}
                  </div>
                  <p className="note">
                    回声置信度 0~1，压缩与转码会抬高基线。
                  </p>
                </>
              ) : audioMissing ? (
                <p className="note">该素材没有音轨或音轨无法解析。</p>
              ) : (
                <p className="note">正在分析音轨…</p>
              )}

              <div className="section-title">清洗建议</div>
              <ul className="suggest">
                {(() => {
                  const suggestions: { text: string; action?: { label: string; level: AntiLevel } }[] = [];
                  if (report.bitstream.flags.length > 0) {
                    suggestions.push(
                      { text: `码流层命中 ${report.bitstream.flags.length} 项疑似特征，建议开启清洗并人工复核` },
                    );
                  }
                  if (report.blind) {
                    const pixelHits = (report.hits ?? []).filter((hit) =>
                      ["ss", "qim", "temporal", "dwt", "dctmod", "svd"].includes(hit),
                    );
                    if (pixelHits.length > 0) {
                      suggestions.push({
                        text: `盲检测命中 ${pixelHits.join("、")}，建议开启空间降噪与 DCT 重量化`,
                      });
                    } else if (!report.hits && (report.blind.ss > 0.5 || report.blind.qim > 0.6)) {
                      suggestions.push(
                        { text: `空域/频域疑似度偏高（ss ${report.blind.ss.toFixed(2)}、qim ${report.blind.qim.toFixed(2)}），建议开启空间降噪与 DCT 重量化` },
                      );
                    }
                    if (report.hits?.includes("echo") || (!report.hits && (report.blind.echo ?? 0) > 0.6)) {
                      suggestions.push(
                        { text: `音频回声置信度 ${(report.blind.echo ?? 0).toFixed(2)}，建议同步音频重混` },
                      );
                    }
                  }
                  if (score >= 60) {
                    suggestions.push({
                      text: `码流评分 ${score}，建议至少开启标准指纹对抗档`,
                      action: { label: "应用标准档", level: "标准" },
                    });
                  }
                  if (score >= 40 && antiLevel === "关闭") {
                    suggestions.push({
                      text: "当前指纹对抗关闭，建议至少开启轻度档",
                      action: { label: "应用轻度档", level: "轻度" },
                    });
                  }
                  if (suggestions.length === 0) {
                    suggestions.push(
                      { text: "未命中明显异常，保持基础清洗即可" },
                    );
                  }
                  return suggestions.map((item) => (
                    <li key={item.text}>
                      {item.text}
                      {item.action ? (
                        <Button
                          variant="outline"
                          size="sm"
                          className="ml-2"
                          onClick={() => setAntiLevel(item.action!.level)}
                        >
                          {item.action.label}
                        </Button>
                      ) : null}
                    </li>
                  ));
                })()}
              </ul>
            </>
          ) : (
            <>
              <Card className="flex items-center gap-3.5 border border-border p-3.5 shadow-none ring-0">
                <div
                  className="ring"
                  style={{ background: `conic-gradient(${ringColor} 0 ${pct}%, var(--muted) ${pct}% 100%)` }}
                >
                  <div className="ring-inner">
                    <span className="ring-score">{risk === "待检测" ? "—" : score}</span>
                  </div>
                </div>
                <div>
                  <div className="score-title">{riskLabel(risk)}</div>
                  <div className="score-sub">{scoreSub}</div>
                </div>
              </Card>

              <Card className="border border-border p-4 shadow-none ring-0">
                <p className="note">
                  尚未检测，检测后此处展示码流异常、元数据与命中维度。
                </p>
              </Card>
            </>
          )}
        </div>
      </ScrollArea>
    </TabsContent>
  );
}
