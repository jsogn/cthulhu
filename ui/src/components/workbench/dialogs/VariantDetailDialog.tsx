import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import type { VariantInfo } from "@/lib/backend";
import { cleanOptionRows } from "@/lib/workbenchOptions";

interface VariantDetailDialogProps {
  variantDetail: VariantInfo | null;
  onClose: () => void;
}

export function VariantDetailDialog({ variantDetail, onClose }: VariantDetailDialogProps) {
  return (
    <Dialog open={!!variantDetail} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="max-w-[min(92vw,520px)]">
        <DialogHeader>
          <DialogTitle>产物参数</DialogTitle>
          <DialogDescription className="break-all text-xs">
            {variantDetail?.output}
          </DialogDescription>
        </DialogHeader>
        {variantDetail && (
          <div className="flex flex-col gap-2 text-sm">
            <div className="flex flex-col gap-1">
              <span className="text-xs text-muted-foreground">源素材</span>
              <span className="mono break-all text-xs">{variantDetail.source}</span>
            </div>
            <div className="kv-row">
              <span>随机种子</span>
              <b className="mono">{variantDetail.seed}</b>
            </div>
            <div className="kv-row">
              <span>创建时间</span>
              <b className="mono">
                {new Date(variantDetail.created_at * 1000).toLocaleString()}
              </b>
            </div>
            {variantDetail.metrics.duplicate_risk != null && (
              <div className="kv-row">
                <span>源片相似度</span>
                <b>
                  {((variantDetail.metrics.duplicate_risk as number) * 100).toFixed(0)}%
                </b>
              </div>
            )}
            {variantDetail.metrics.ssim != null && (
              <div className="kv-row">
                <span>SSIM</span>
                <b>{String(variantDetail.metrics.ssim)}</b>
              </div>
            )}
            {(() => {
              const rows = cleanOptionRows(variantDetail.options);
              return rows.length > 0 ? (
                <>
                  <div className="section-title">清洗参数</div>
                  {rows.map((row) => (
                    <div className="kv-row" key={row.label}>
                      <span>{row.label}</span>
                      <b>{row.value}</b>
                    </div>
                  ))}
                </>
              ) : null;
            })()}
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}
