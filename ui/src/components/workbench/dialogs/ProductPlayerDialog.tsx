import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { mediaUrl } from "@/lib/backend";

interface ProductPlayerDialogProps {
  playerPath: string | null;
  onClose: () => void;
}

export function ProductPlayerDialog({ playerPath, onClose }: ProductPlayerDialogProps) {
  return (
    <Dialog open={!!playerPath} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="max-w-[min(90vw,720px)]">
        <DialogHeader>
          <DialogTitle>产物预览</DialogTitle>
          <DialogDescription>播放产物视频，确认观感。</DialogDescription>
        </DialogHeader>
        {playerPath && (
          <video
            src={mediaUrl(playerPath)}
            controls
            autoPlay
            className="max-h-[70vh] w-full rounded-md bg-black"
          />
        )}
      </DialogContent>
    </Dialog>
  );
}
