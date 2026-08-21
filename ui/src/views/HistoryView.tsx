import { useEffect, useState } from "react";
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
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { clearAudit, listAudit } from "@/lib/backend";
import { useHistoryStore } from "@/stores/history";
import { openInFolder, toast } from "@/stores/toasts";

function resultClass(result: string): string {
  if (result === "成功") return "低";
  if (result === "排队中") return "中";
  return "高";
}

export default function HistoryView() {
  const entries = useHistoryStore((state) => state.entries);
  const [confirmClear, setConfirmClear] = useState(false);

  useEffect(() => {
    void listAudit()
      .then((loaded) => useHistoryStore.getState().load(loaded))
      .catch(() => undefined);
  }, []);

  const doClear = async () => {
    try {
      const removed = await clearAudit();
      useHistoryStore.getState().load([]);
      toast(`已清空 ${removed} 条记录`);
    } catch {
      toast("清空失败，请确认引擎在线");
    } finally {
      setConfirmClear(false);
    }
  };

  return (
    <>
      <div className="view-head">
        <div>
          <div className="view-title">处理历史</div>
          <div className="view-desc">每次处理的参数与输出都留在这里</div>
        </div>
        <div className="flex gap-2">
          <Button variant="ghost" disabled={!entries.length} onClick={() => setConfirmClear(true)}>
            清空记录
          </Button>
        </div>
      </div>

      <div className="view-body">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>素材</TableHead>
              <TableHead>时间</TableHead>
              <TableHead>动作</TableHead>
              <TableHead>参数快照</TableHead>
              <TableHead>输出</TableHead>
              <TableHead>结果</TableHead>
              <TableHead />
            </TableRow>
          </TableHeader>
          <TableBody>
            {entries.map((entry, i) => (
              <TableRow key={i}>
                <TableCell className="max-w-52 truncate">{entry.name}</TableCell>
                <TableCell className="mono">{entry.time}</TableCell>
                <TableCell>{entry.action}</TableCell>
                <TableCell>{entry.params}</TableCell>
                <TableCell className="mono max-w-64 truncate">{entry.out}</TableCell>
                <TableCell>
                  <span className={`dim-level ${resultClass(entry.result)}`}>{entry.result}</span>
                </TableCell>
                <TableCell>
                  {entry.out && entry.out !== "—" && (
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => openInFolder(entry.out)}
                    >
                      打开文件夹
                    </Button>
                  )}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>

      <AlertDialog open={confirmClear} onOpenChange={setConfirmClear}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>确认清空处理历史</AlertDialogTitle>
            <AlertDialogDescription>将删除全部处理记录，此操作不可撤销。</AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>取消</AlertDialogCancel>
            <AlertDialogAction onClick={doClear}>确认清空</AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  );
}
