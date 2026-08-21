import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  fetchOutputs,
  makePreview,
  previewImageUrl,
  runSimilarity,
  type OutputInfo,
  type SimilarityReport,
} from "@/lib/backend";
import { fmtSize } from "@/lib/format";
import { useMaterialsStore } from "@/stores/materials";
import { openInFolder, toast } from "@/stores/toasts";

const KIND_LABEL: Record<OutputInfo["kind"], string> = {
  cleaned: "清洗",
  repaired: "修复",
  candidate: "候选",
};

function timeLabel(mtime: number): string {
  const date = new Date(mtime * 1000);
  const pad = (value: number) => String(value).padStart(2, "0");
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${pad(
    date.getHours(),
  )}:${pad(date.getMinutes())}`;
}

export default function OutputsView() {
  const materials = useMaterialsStore((state) => state.materials);
  const [sourcePath, setSourcePath] = useState<string>("");
  const [outputs, setOutputs] = useState<OutputInfo[]>([]);
  const [sideA, setSideA] = useState<string>("");
  const [sideB, setSideB] = useState<string>("");
  const [compareOpen, setCompareOpen] = useState(false);
  const [comparing, setComparing] = useState(false);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [similarity, setSimilarity] = useState<SimilarityReport | null>(null);

  useEffect(() => {
    if (!sourcePath) {
      setOutputs([]);
      return;
    }
    void fetchOutputs(sourcePath)
      .then((report) => {
        setOutputs(report.outputs);
        setSideA(report.outputs[0]?.path ?? "");
        setSideB(report.outputs[1]?.path ?? report.outputs[0]?.path ?? "");
      })
      .catch(() => setOutputs([]));
  }, [sourcePath]);

  const compare = async () => {
    if (!sideA || !sideB) {
      toast("请先选择两个产物");
      return;
    }
    setComparing(true);
    try {
      const [sim, preview] = await Promise.all([
        runSimilarity(sideA, sideB),
        makePreview(sideA, sideB),
      ]);
      setSimilarity(sim);
      setImageUrl(previewImageUrl(preview.image_path));
      setCompareOpen(true);
    } catch (error) {
      toast(error instanceof Error ? error.message : "对比失败");
    } finally {
      setComparing(false);
    }
  };

  return (
    <>
      <div className="view-head">
        <div>
          <div className="view-title">处理产物</div>
          <div className="view-desc">按源素材查看全部清洗 / 修复 / 候选产物，任选两个并排对比</div>
        </div>
      </div>

      <div className="view-body">
        <div className="flex flex-col gap-3">
          <div className="field">
            <span className="field-label">源素材</span>
            <Select value={sourcePath} onValueChange={setSourcePath}>
              <SelectTrigger className="h-9">
                <SelectValue placeholder="选择素材" />
              </SelectTrigger>
              <SelectContent>
                {materials
                  .filter((material) => material.path)
                  .map((material) => (
                    <SelectItem key={material.path} value={material.path as string}>
                      {material.name}
                    </SelectItem>
                  ))}
              </SelectContent>
            </Select>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div className="field">
              <span className="field-label">对比 A</span>
              <Select value={sideA} onValueChange={setSideA}>
                <SelectTrigger className="h-9">
                  <SelectValue placeholder="选择产物" />
                </SelectTrigger>
                <SelectContent>
                  {outputs.map((output) => (
                    <SelectItem key={output.path} value={output.path}>
                      {output.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="field">
              <span className="field-label">对比 B</span>
              <Select value={sideB} onValueChange={setSideB}>
                <SelectTrigger className="h-9">
                  <SelectValue placeholder="选择产物" />
                </SelectTrigger>
                <SelectContent>
                  {outputs.map((output) => (
                    <SelectItem key={output.path} value={output.path}>
                      {output.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>

          <Button variant="secondary" disabled={comparing || !sideA || !sideB} onClick={compare}>
            {comparing ? "对比中…" : "并排对比"}
          </Button>

          {outputs.length > 0 && (
            <div className="flex flex-col gap-1">
              {outputs.map((output) => (
                <div
                  key={output.path}
                  className="flex items-center justify-between gap-3 rounded-md border border-border px-3 py-2"
                >
                  <span className="mono truncate text-xs text-muted-foreground">
                    {KIND_LABEL[output.kind]} · {output.name} · {fmtSize(output.size)}
                  </span>
                  <span className="shrink-0 text-xs text-muted-foreground">
                    {timeLabel(output.mtime)}
                  </span>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => openInFolder(output.path)}
                  >
                    打开
                  </Button>
                </div>
              ))}
            </div>
          )}
          {sourcePath && outputs.length === 0 && (
            <p className="text-muted-foreground">该素材还没有处理产物。</p>
          )}
        </div>
      </div>

      <Dialog open={compareOpen} onOpenChange={setCompareOpen}>
        <DialogContent className="max-w-[min(90vw,960px)]">
          <DialogHeader>
            <DialogTitle>产物并排对比</DialogTitle>
            <DialogDescription>
              左列为对比 A，右列为对比 B；指标越低差异越大。
            </DialogDescription>
          </DialogHeader>
          {imageUrl && (
            <img
              src={imageUrl}
              alt="产物对比"
              className="max-h-[60vh] w-full rounded-md object-contain"
            />
          )}
          {similarity && (
            <div className="grid grid-cols-3 gap-2 text-xs text-muted-foreground">
              <div className="rounded-md border border-border p-2">
                内容相似度 {similarity.content_cosine.toFixed(3)}
              </div>
              <div className="rounded-md border border-border p-2">
                时序相似度 {similarity.motion_cosine.toFixed(3)}
              </div>
              <div className="rounded-md border border-border p-2">
                SSIM {similarity.ssim_mean.toFixed(3)}
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </>
  );
}
