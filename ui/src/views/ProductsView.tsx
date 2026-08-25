import { useEffect, useMemo, useState } from "react";
import { RefreshCw, Trash2 } from "lucide-react";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { fmtSize } from "@/lib/format";
import { cn } from "@/lib/utils";
import { deleteProducts, listAllProducts, type ProductInfo } from "@/lib/backend";
import { openInFolder, toast } from "@/stores/toasts";

const KIND_LABEL: Record<ProductInfo["kind"], string> = {
  cleaned: "清洗",
  repaired: "修复",
};

type SortKey = "time" | "size";

function basename(path: string): string {
  return path.split(/[\\/]/).pop() || path;
}

function formatTime(timestamp: number): string {
  if (!timestamp) return "—";
  return new Date(timestamp * 1000).toLocaleString();
}

export default function ProductsView() {
  const [products, setProducts] = useState<ProductInfo[]>([]);
  const [summary, setSummary] = useState({ count: 0, total_size: 0 });
  const [loading, setLoading] = useState(false);
  const [sort, setSort] = useState<SortKey>("time");
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [confirmPaths, setConfirmPaths] = useState<string[] | null>(null);

  const load = async () => {
    setLoading(true);
    try {
      const data = await listAllProducts();
      setProducts(data.products);
      setSummary({ count: data.count, total_size: data.total_size });
      setSelected((prev) => {
        const next = new Set([...prev].filter((p) => data.products.some((x) => x.path === p)));
        return next;
      });
    } catch (error) {
      toast(error instanceof Error ? error.message : "读取产物列表失败");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void load();
  }, []);

  const sorted = useMemo(() => {
    const arr = [...products];
    arr.sort((a, b) => (sort === "size" ? b.size - a.size : b.mtime - a.mtime));
    return arr;
  }, [products, sort]);

  const missingCount = products.filter((p) => !p.exists).length;
  const allSelected = products.length > 0 && products.every((p) => selected.has(p.path));

  const toggleOne = (path: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(path)) next.delete(path);
      else next.add(path);
      return next;
    });
  };

  const toggleAll = () => {
    setSelected(allSelected ? new Set() : new Set(products.map((p) => p.path)));
  };

  const confirmDelete = async () => {
    if (!confirmPaths) return;
    try {
      const result = await deleteProducts(confirmPaths);
      toast(`已删除 ${result.removed} 个产物`, "源视频不受影响");
      setSelected(new Set());
      await load();
    } catch (error) {
      toast(error instanceof Error ? error.message : "删除失败");
    } finally {
      setConfirmPaths(null);
    }
  };

  return (
    <>
      <div className="view-head">
        <div>
          <div className="view-title">产物管理</div>
          <div className="view-desc">集中查看清洗产物占用，及时清理不再需要的文件</div>
        </div>
        <div className="flex gap-2">
          <Button
            variant="ghost"
            disabled={selected.size === 0}
            onClick={() => setConfirmPaths([...selected])}
          >
            <Trash2 />
            删除所选
          </Button>
          <Button variant="ghost" disabled={loading} onClick={() => void load()}>
            <RefreshCw />
            刷新
          </Button>
        </div>
      </div>

      <div className="view-body">
        <div className="stats">
          <Card className="stat p-3.5">
            <div className="stat-num">{summary.count}</div>
            <div className="stat-label">产物总数</div>
          </Card>
          <Card className="stat p-3.5">
            <div className="stat-num">{fmtSize(summary.total_size)}</div>
            <div className="stat-label">总占用空间</div>
          </Card>
          <Card className="stat p-3.5">
            <div className="stat-num">{missingCount}</div>
            <div className="stat-label">文件缺失</div>
          </Card>
        </div>

        <div className="products-toolbar flex items-center justify-between gap-3">
          <label className="flex items-center gap-2 text-sm text-muted-foreground">
            <Checkbox
              checked={allSelected}
              onCheckedChange={() => toggleAll()}
              aria-label="全选产物"
            />
            {selected.size > 0 ? `已选 ${selected.size} 项` : "全选"}
          </label>
          <Select value={sort} onValueChange={(value) => setSort(value as SortKey)}>
            <SelectTrigger className="h-8 w-36">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="time">最新优先</SelectItem>
              <SelectItem value="size">体积优先</SelectItem>
            </SelectContent>
          </Select>
        </div>

        {products.length === 0 ? (
          <Card className="p-8 text-center text-muted-foreground">
            暂无处理产物，清洗或修复后产物会集中出现在这里。
          </Card>
        ) : (
          <div className="flex flex-col gap-2">
            {sorted.map((p) => (
              <div
                key={p.path}
                className={cn(
                  "flex items-center gap-3 rounded-md border border-border bg-card p-3",
                  !p.exists && "opacity-60",
                )}
              >
                <Checkbox
                  checked={selected.has(p.path)}
                  onCheckedChange={() => toggleOne(p.path)}
                  aria-label={`勾选${p.name}`}
                />
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2">
                    <span className="shrink-0 rounded border border-border px-1.5 py-0.5 text-xs text-muted-foreground">
                      {KIND_LABEL[p.kind]}
                    </span>
                    <span className="truncate font-medium" title={p.name}>
                      {p.name}
                    </span>
                    {!p.exists && (
                      <span className="shrink-0 text-xs text-destructive">文件缺失</span>
                    )}
                  </div>
                  <div className="mt-0.5 truncate text-xs text-muted-foreground" title={p.source}>
                    来源：{basename(p.source) || "未知"}
                  </div>
                </div>
                <div className="w-20 shrink-0 text-right text-sm">{fmtSize(p.size)}</div>
                <div className="w-40 shrink-0 text-right text-xs text-muted-foreground">
                  {formatTime(p.mtime)}
                </div>
                <div className="flex shrink-0 gap-1">
                  <Button variant="ghost" size="sm" onClick={() => openInFolder(p.path)}>
                    打开
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="text-destructive"
                    onClick={() => setConfirmPaths([p.path])}
                  >
                    删除
                  </Button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      <AlertDialog
        open={!!confirmPaths}
        onOpenChange={(open) => !open && setConfirmPaths(null)}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>确认删除产物</AlertDialogTitle>
            <AlertDialogDescription>
              将删除 {confirmPaths?.length ?? 0} 个产物文件及记录，文件会从磁盘移除，此操作不可恢复；源视频不受影响。
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>取消</AlertDialogCancel>
            <AlertDialogAction onClick={() => void confirmDelete()}>确认删除</AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  );
}
