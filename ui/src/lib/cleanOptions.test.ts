// @vitest-environment jsdom
import { describe, expect, it } from "vitest";

import {
  cleanOutputPath,
  FIXED_CROP,
  FIXED_PERTURB,
  makeCleanOptions,
} from "@/lib/cleanOptions";
import { useCleanPanel, type CleanPanelState } from "@/stores/cleanPanel";

function panelState(overrides: Partial<CleanPanelState> = {}): CleanPanelState {
  return { ...useCleanPanel.getState(), ...overrides };
}

describe("makeCleanOptions", () => {
  it("remux 模式只换壳，不携带任何对抗参数", () => {
    const options = makeCleanOptions(panelState({ outputMode: "remux", antiLevel: "全兵器" }));
    expect(options.output_mode).toBe("remux");
    expect(options.reorder).toBe(false);
    expect(options.speed).toBe(1);
    expect(options.recrop).toBe(0);
    expect(options.regrade).toBe(false);
    expect(options.audio_remix).toBe(false);
    expect(options.echo_defeat).toBe(false);
    expect(options.sharpness).toBe(false);
    expect(options.color_restore).toBe(false);
    expect(options.denoise).toBe(false);
    expect(options.anti_reembed).toBe(false);
    expect(typeof options.seed).toBe("number");
  });

  it("标准档映射到静态几何与签名域参数", () => {
    const options = makeCleanOptions(panelState({ antiLevel: "标准" }));
    expect(options.rotate).toBe(0.2);
    expect(options.requant).toBe(96);
    expect(options.noise).toBe(0.003);
    expect(options.audio_strong).toBe(true);
  });

  it("全兵器档同时启用频域、色度与联合攻击", () => {
    const options = makeCleanOptions(panelState({ antiLevel: "全兵器" }));
    expect(options.dct_step).toBe(12);
    expect(options.chroma_levels).toBe(32);
    expect(options.transcode_chain).toBe(true);
    expect(options.multi_hash_attack).toBe(true);
    expect(options.requant).toBe(32);
  });

  it("H.265 映射 libx265，H.264 映射 libx264", () => {
    expect(makeCleanOptions(panelState({ codec: "H.265" })).codec).toBe("libx265");
    expect(makeCleanOptions(panelState({ codec: "H.264" })).codec).toBe("libx264");
  });

  it("重新构图与调光开关映射为固定幅度", () => {
    const options = makeCleanOptions(panelState({ recropOn: true, regradeOn: true }));
    expect(options.recrop).toBe(FIXED_CROP);
    expect(options.perturb).toBe(FIXED_PERTURB);
    expect(options.regrade).toBe(true);
  });

  it("默认跳过 VMAF 质量评估，保留 PSNR/SSIM", () => {
    expect(makeCleanOptions(panelState()).skip_vmaf).toBe(true);
  });
});

describe("cleanOutputPath", () => {
  it("按命名模板拼接输出路径并剥离源后缀", () => {
    expect(cleanOutputPath("/素材/视频.MP4", "原文件名 + 时间戳", "/导出", "20240101")).toBe(
      "/导出/视频_清洗_20240101.mp4",
    );
    expect(
      cleanOutputPath("/素材/视频.mov", "时间戳 + 原文件名", "/导出/", "20240101", 2),
    ).toBe("/导出/20240101_视频_清洗_2.mp4");
  });

  it("未配置导出目录时回退默认目录", () => {
    expect(cleanOutputPath("/素材/a.mp4", "原文件名 + 时间戳", "", "20240101")).toBe(
      "~/Documents/Cthulhu/a_清洗_20240101.mp4",
    );
  });
});
