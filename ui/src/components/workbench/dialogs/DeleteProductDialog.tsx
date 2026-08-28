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
import type { OutputInfo } from "@/lib/backend";

interface DeleteProductDialogProps {
  pendingDelete: OutputInfo | null;
  onClose: () => void;
  onConfirm: () => void;
}

export function DeleteProductDialog({
  pendingDelete,
  onClose,
  onConfirm,
}: DeleteProductDialogProps) {
  return (
    <AlertDialog
      open={!!pendingDelete}
      onOpenChange={(open) => !open && onClose()}
    >
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>确认删除产物</AlertDialogTitle>
          <AlertDialogDescription>
            将删除「{pendingDelete?.name}」，文件会从磁盘移除，此操作不可恢复。
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel>取消</AlertDialogCancel>
          <AlertDialogAction onClick={onConfirm}>确认删除</AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}
