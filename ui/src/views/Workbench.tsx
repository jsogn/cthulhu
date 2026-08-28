import { useState } from "react";
import { BatchPopup } from "@/components/workbench/BatchPopup";
import { ContextPanel } from "@/components/workbench/ContextPanel";
import { TAB_KEYS } from "@/components/workbench/ContextTabs";
import { MaterialPane } from "@/components/workbench/MaterialPane";
import { PreviewPane } from "@/components/workbench/PreviewPane";
import { useMaterialsStore } from "@/stores/materials";

export default function Workbench() {
  const materials = useMaterialsStore((state) => state.materials);
  const activeId = useMaterialsStore((state) => state.activeId);

  const active = materials.find((m) => m.id === activeId) ?? null;
  const [tab, setTab] = useState<(typeof TAB_KEYS)[number]>("清洗去重");

  const [frame, setFrame] = useState(132);
  const [playing, setPlaying] = useState(false);
  const [comparePct, setComparePct] = useState(50);
  const [compareLeft, setCompareLeft] = useState<string>("");
  const [compareRight, setCompareRight] = useState<string>("");
  const [compareLeftLabel, setCompareLeftLabel] = useState("原片");
  const [compareRightLabel, setCompareRightLabel] = useState("产物");

  const startCompare = (
    left: string,
    right: string,
    leftLabel: string,
    rightLabel: string,
  ) => {
    setCompareLeft(left);
    setCompareRight(right);
    setCompareLeftLabel(leftLabel);
    setCompareRightLabel(rightLabel);
  };

  const exitCompare = () => {
    setCompareLeft("");
    setCompareRight("");
  };

  return (
    <section className="workbench">
      <div className="workbench-main">
        <MaterialPane />
        <PreviewPane
          material={active}
          frame={frame}
          setFrame={setFrame}
          playing={playing}
          setPlaying={setPlaying}
          comparePct={comparePct}
          setComparePct={setComparePct}
          compareLeft={compareLeft}
          compareRight={compareRight}
          compareLeftLabel={compareLeftLabel}
          compareRightLabel={compareRightLabel}
          onExitCompare={exitCompare}
        />
        <ContextPanel
          tab={tab}
          setTab={setTab}
          onStartCompare={startCompare}
        />
      </div>
      <BatchPopup />
    </section>
  );
}
