import { useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Input } from "@/components/ui/input";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { fmtSize } from "@/lib/format";
import {
  registerLibrary,
  scanImportPath,
  thumbUrl,
  uploadFile,
  type LibraryVideoInfo,
} from "@/lib/backend";
import { useAppStore } from "@/stores/app";
import { useMaterialsStore, type Material } from "@/stores/materials";
import { toast } from "@/stores/toasts";

interface PendingFile {
  file: File;
  name: string;
  size: string;
  path: string | null;
  dup: boolean;
  invalid: boolean;
}

function fmtDur(sec: number): string {
  sec = Math.round(sec || 0);
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = sec % 60;
  const p = (n: number) => String(n).padStart(2, "0");
  return `${p(h)}:${p(m)}:${p(s)}`;
}

function probeVideo(file: File): Promise<{ dur: string; res: string }> {
  return new Promise((resolve) => {
    const url = URL.createObjectURL(file);
    const video = document.createElement("video");
    video.preload = "metadata";
    video.onloadedmetadata = () => {
      const dur = Number.isFinite(video.duration) ? video.duration : 0;
      const res =
        video.videoWidth && video.videoHeight ? `${video.videoWidth}×${video.videoHeight}` : "—";
      URL.revokeObjectURL(url);
      resolve({ dur: fmtDur(dur), res });
    };
    video.onerror = () => {
      URL.revokeObjectURL(url);
      resolve({ dur: "00:00", res: "—" });
    };
    video.src = url;
  });
}

export default function ImportDialog() {
  const open = useAppStore((state) => state.importOpen);
  const setOpen = useAppStore((state) => state.setImportOpen);
  const materials = useMaterialsStore((state) => state.materials);
  const importOne = useMaterialsStore((state) => state.importOne);
  const select = useMaterialsStore((state) => state.select);

  const [pending, setPending] = useState<PendingFile[]>([]);
  const [localPath, setLocalPath] = useState("");
  const [folderScanning, setFolderScanning] = useState(false);
  const [importing, setImporting] = useState<{
    current: number;
    total: number;
    percent: number;
  } | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const folderInputRef = useRef<HTMLInputElement>(null);

  const addFiles = (list: FileList | null | undefined) => {
    const files = Array.from(list ?? []);
    if (!files.length) return;
    setPending((prev) => [
      ...prev,
      ...files.map((file) => ({
        file,
        name: file.name,
        size: fmtSize(file.size),
        path: window.appEnv?.getPathForFile?.(file) ?? null,
        dup: materials.some((m) => m.name === file.name) || prev.some((p) => p.name === file.name),
        invalid: !/\.(mp4|mov|mkv|avi|webm|flv|ts)$/i.test(file.name),
      })),
    ]);
  };

  const confirm = async () => {
    const fresh = pending.filter((f) => !f.dup && !f.invalid);
    const dupCount = pending.filter((f) => f.dup).length;
    const invalidCount = pending.filter((f) => f.invalid).length;
    if (!fresh.length) {
      setOpen(false);
      toast(
        invalidCount
          ? `${invalidCount} 个文件格式不支持，已跳过`
          : dupCount
            ? "所选文件均重复，已跳过"
            : "尚未选择文件",
      );
      return;
    }
    setImporting({ current: 0, total: fresh.length, percent: 0 });

    let firstId: string | null = null;
    let success = 0;
    let failed = 0;
    for (let i = 0; i < fresh.length; i++) {
      const item = fresh[i];
      setImporting({ current: i, total: fresh.length, percent: 0 });
      try {
        // 桌面端拖入可直接登记本地路径；网页端先把文件上传到引擎素材库。
        // 两个入口都会读取真实 fps/时长/分辨率，保证预览与清洗参数准确。
        let path: string;
        let video: LibraryVideoInfo | null | undefined;
        if (item.path) {
          const record = await registerLibrary(item.path);
          path = record.path;
          video = record.video;
        } else {
          const uploaded = await uploadFile(item.file, (percent) =>
            setImporting({ current: i, total: fresh.length, percent }),
          );
          path = uploaded.path;
          video = uploaded.video;
        }
        let dur = "00:00";
        let res = "—";
        let fps = "—";
        let duration: number | undefined;
        if (video) {
          dur = fmtDur(video.duration);
          res = `${video.width}×${video.height}`;
          fps = `${video.fps}fps`;
          duration = video.duration;
        } else {
          const meta = await probeVideo(item.file);
          dur = meta.dur;
          res = meta.res;
        }
        const material: Material = {
          id: `imp${Date.now()}_${i}`,
          name: item.name,
          dur,
          res,
          fps,
          size: item.size,
          risk: "待检测",
          score: 0,
          tags: ["本地"],
          frame: thumbUrl(path, 320),
          path,
          duration,
        };
        importOne(material);
        success++;
        if (!firstId) firstId = material.id;
      } catch {
        failed++;
      }
    }
    setImporting(null);
    setOpen(false);
    setPending([]);
    if (firstId) select(firstId);
    const parts = [`已导入 ${success} 个素材`];
    if (failed) parts.push(`${failed} 个失败`);
    if (dupCount) parts.push(`跳过 ${dupCount} 个重复`);
    if (invalidCount) parts.push(`${invalidCount} 个格式不支持`);
    toast(parts.join("，"));
  };

  const detectLocal = async () => {
    const path = localPath.trim();
    if (!path) {
      toast("请输入本机视频文件路径");
      return;
    }
    try {
      const record = await registerLibrary(path);
      if (!record.video) {
        toast("无法读取该视频文件，请确认路径与格式");
        return;
      }
      const video = record.video;
      const name = path.split(/[\\/]/).pop() || "本地素材";
      const material: Material = {
        id: `loc${Date.now()}`,
        name,
        dur: fmtDur(video.duration),
        res: `${video.width}×${video.height}`,
        fps: `${video.fps}fps`,
        size: fmtSize(record.size),
        risk: "待检测",
        score: 0,
        tags: ["本地"],
        frame: thumbUrl(path, 320),
        path,
        duration: video.duration,
      };
      importOne(material);
      select(material.id);
      setOpen(false);
      setLocalPath("");
      toast(`已导入：${name}（待检测，可在检测参考页手动检测）`);
    } catch (error) {
      toast(error instanceof Error ? error.message : "导入失败，请确认路径与文件格式");
    }
  };

  const importFolder = async (folderPath: string) => {
    setFolderScanning(true);
    try {
      const scan = await scanImportPath(folderPath);
      let firstId: string | null = null;
      scan.files.forEach((file, i) => {
        if (!file.video) return;
        const material: Material = {
          id: `dir${Date.now()}_${i}`,
          name: file.name,
          dur: fmtDur(file.video.duration),
          res: `${file.video.width}×${file.video.height}`,
          fps: `${file.video.fps}fps`,
          size: fmtSize(file.size),
          risk: "待检测",
          score: 0,
          tags: ["本地", "文件夹"],
          frame: thumbUrl(file.path, 320),
          path: file.path,
          duration: file.video.duration,
        };
        importOne(material);
        if (!firstId) firstId = material.id;
      });
      if (firstId) select(firstId);
      // 登记进持久化素材库，重启后恢复显示（登记失败不阻断导入）。
      scan.files
        .filter((file) => file.video)
        .forEach((file) => registerLibrary(file.path).catch(() => undefined));
      toast(
        `已导入 ${scan.valid} 个素材${scan.invalid ? `，跳过 ${scan.invalid} 个无效文件` : ""}`,
      );
      setOpen(false);
    } catch (error) {
      toast(error instanceof Error ? error.message : "文件夹导入失败");
    } finally {
      setFolderScanning(false);
    }
  };

  const chooseFolder = async () => {
    const picker = window.appEnv?.chooseFolder;
    if (picker) {
      const folderPath = await picker();
      if (folderPath) await importFolder(folderPath);
    } else {
      // 浏览器回退：webkitdirectory 选择目录，走既有文件导入流程。
      folderInputRef.current?.click();
    }
  };

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>批量导入素材</DialogTitle>
          <DialogDescription>
            拖入视频文件或点击选择，支持 MP4 / MOV 等常见格式，自动解析时长、分辨率、大小并去重。
          </DialogDescription>
        </DialogHeader>

        <div
          className="dropzone"
          role="button"
          tabIndex={0}
          onClick={() => inputRef.current?.click()}
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === " ") {
              e.preventDefault();
              inputRef.current?.click();
            }
          }}
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => {
            e.preventDefault();
            addFiles(e.dataTransfer.files);
          }}
        >
          拖入视频文件，或点击选择文件
          <span className="text-xs text-muted-foreground">
            支持 MP4 / MOV / MKV / AVI / FLV / TS；网页端会自动上传到本地素材库
          </span>
        </div>

        <div className="mt-1 flex gap-2">
          <Input
            className="h-8 flex-1"
            placeholder="或输入本机视频绝对路径"
            value={localPath}
            onChange={(e) => setLocalPath(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") detectLocal();
            }}
          />
          <Button size="sm" variant="secondary" disabled={!localPath.trim()} onClick={detectLocal}>
            检测并导入
          </Button>
          <Button size="sm" variant="secondary" disabled={folderScanning} onClick={chooseFolder}>
            {folderScanning ? "扫描中…" : "选择文件夹"}
          </Button>
        </div>

        <input
          ref={inputRef}
          type="file"
          multiple
          hidden
          accept="video/*,.mp4,.mov,.mkv,.avi,.webm,.flv,.ts"
          onChange={(e) => {
            addFiles(e.target.files);
            e.target.value = "";
          }}
        />
        <input
          ref={folderInputRef}
          type="file"
          multiple
          hidden
          {...({ webkitdirectory: "" } as Record<string, string>)}
          onChange={(e) => {
            addFiles(e.target.files);
            e.target.value = "";
          }}
        />

        <div className="import-files">
          {pending.map((f, i) => (
            <div key={`${f.name}-${i}`} className="import-file">
              <span className="fi-name">{f.name}</span>
              <span className="fi-meta">
                {f.path ? `桌面路径就绪 · ${f.size}` : f.size}
              </span>
              <span className={`fi-flag ${f.invalid ? "invalid" : f.dup ? "dup" : "ok"}`}>
                {f.invalid ? "格式不支持" : f.dup ? "重复，已跳过" : "校验通过"}
              </span>
            </div>
          ))}
        </div>

        {importing && (
          <div className="flex flex-col gap-1.5">
            <Progress
              value={
                ((importing.current + importing.percent / 100) /
                  Math.max(1, importing.total)) *
                100
              }
              className="h-1.5"
            />
            <span className="text-xs text-muted-foreground">
              正在导入 {importing.current + 1} / {importing.total}
              {importing.percent > 0
                ? ` · 上传 ${importing.percent}%`
                : " · 解析视频信息…"}
            </span>
          </div>
        )}

        <DialogFooter>
          <Button variant="ghost" disabled={!!importing} onClick={() => setOpen(false)}>
            取消
          </Button>
          <Button disabled={!!importing} onClick={confirm}>
            {importing ? "导入中…" : "开始导入"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
