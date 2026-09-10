import { useEffect, useRef, useState } from "react";
import { MoreVertical } from "lucide-react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
} from "@/components/ui/select";
import { TabsList, TabsTrigger } from "@/components/ui/tabs";
import { cn } from "@/lib/utils";

export const TAB_KEYS = ["清洗去重", "处理产物"] as const;

export type WorkbenchTab = (typeof TAB_KEYS)[number];

interface ContextTabsProps {
  setTab: (tab: WorkbenchTab) => void;
}

/** 自适应标签栏：放得下全部展示，放不下收起尾部并提供“更多”下拉。 */
export function ContextTabs({ setTab }: ContextTabsProps) {
  const tabRowRef = useRef<HTMLDivElement | null>(null);
  const tabItemRefs = useRef<(HTMLButtonElement | null)[]>([]);
  const [visibleTabCount, setVisibleTabCount] = useState<number>(TAB_KEYS.length);

  useEffect(() => {
    const compute = () => {
      const row = tabRowRef.current;
      if (!row) return;
      const widths = TAB_KEYS.map((_, index) => tabItemRefs.current[index]?.offsetWidth ?? 0);
      const total = widths.reduce((sum, width) => sum + width + 4, 0);
      // 全部放得下时不显示“更多”按钮，也无需预留宽度。
      if (total <= row.clientWidth) {
        setVisibleTabCount(TAB_KEYS.length);
        return;
      }
      const available = row.clientWidth - 48; // 溢出时预留“更多”按钮宽度
      let used = 0;
      let count = 0;
      for (let index = 0; index < TAB_KEYS.length; index++) {
        const item = tabItemRefs.current[index];
        if (!item) break;
        used += item.offsetWidth + 4; // 加上相邻标签的间距
        if (used > available) break;
        count = index + 1;
      }
      setVisibleTabCount(Math.max(1, count));
    };
    compute();
    const observer = new ResizeObserver(compute);
    if (tabRowRef.current) observer.observe(tabRowRef.current);
    return () => observer.disconnect();
  }, []);

  return (
    <div ref={tabRowRef} className="flex w-full shrink-0 items-center border-b border-border">
      <TabsList
        variant="line"
        className="flex min-w-0 flex-1 justify-start gap-1 overflow-hidden rounded-none border-0 p-0"
      >
        {TAB_KEYS.map((key, index) => (
          <TabsTrigger
            key={key}
            ref={(el) => {
              tabItemRefs.current[index] = el;
            }}
            value={key}
            className={cn(
              "shrink-0 whitespace-nowrap px-2",
              index >= visibleTabCount && "invisible",
            )}
          >
            {key}
          </TabsTrigger>
        ))}
      </TabsList>
      {visibleTabCount < TAB_KEYS.length && (
        <Select
          value=""
          onValueChange={(value) => setTab(value as WorkbenchTab)}
        >
          <SelectTrigger
            className="mr-1 h-8 w-8 shrink-0 justify-center gap-0 border-0 bg-transparent p-0 text-muted-foreground shadow-none hover:text-foreground [&>svg:last-child]:hidden"
            aria-label="更多标签页"
          >
            <MoreVertical className="size-4" />
          </SelectTrigger>
          <SelectContent align="end">
            {TAB_KEYS.slice(visibleTabCount).map((key) => (
              <SelectItem key={key} value={key}>
                {key}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      )}
    </div>
  );
}
