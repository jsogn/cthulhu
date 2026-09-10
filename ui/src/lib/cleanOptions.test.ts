// @vitest-environment jsdom
import { describe, expect, it } from "vitest";

import {
  cleanOutputPath,
  collusionOutputPath,
  FIXED_CROP,
  FIXED_PERTURB,
  makeCleanOptions,
} from "@/lib/cleanOptions";
import { useCleanPanel, type CleanPanelState } from "@/stores/cleanPanel";

function panelState(overrides: Partial<CleanPanelState> = {}): CleanPanelState {
  return { ...useCleanPanel.getState(), ...overrides };
}

describe("makeCleanOptions", () => {
  it("经典指纹原语按开关映射为后端参数", () => {
    const options = makeCleanOptions(
      panelState({
        rotateOn: true,
        rotate: 1.2,
        hashOn: true,
        hashEps: 0.05,
        requantOn: true,
        requant: 64,
        noiseOn: true,
        noise: 0.004,
        dctOn: true,
        audioStrongOn: true,
      }),
    );
    expect(options.rotate).toBe(1.2);
    expect(options.multi_hash_attack).toBeUndefined();
    expect(options.phash_epsilon).toBe(0.05);
    expect(options.phash_attack).toBe(true);
    expect(options.requant).toBe(64);
    expect(options.noise).toBe(0.004);
    expect(options.dct_step).toBe(12);
    expect(options.anti_reembed).toBeUndefined();
    expect(options.audio_strong).toBe(true);
  });

  it("DCT 量化步长使用面板值而不是写死 12", () => {
    const options = makeCleanOptions(panelState({ dctOn: true, dctStep: 20 }));
    expect(options.dct_step).toBe(20);
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
    expect(options.hsv_jitter).toBe(6);
  });

  it("默认跳过 VMAF 质量评估，保留 PSNR/SSIM", () => {
    expect(makeCleanOptions(panelState()).skip_vmaf).toBe(true);
  });

  it("再生重写武器按开关与强度映射为后端参数", () => {
    const options = makeCleanOptions(
      panelState({
        temporalSubOn: true,
        temporalSub: 1.2,
        nativeTemporalOn: true,
        fftOn: true,
        fftPhase: 0.4,
        fftMag: 0.7,
        dwtDetailOn: false,
        nonintOn: true,
        nonintRatio: 0.02,
        flowOn: true,
        flow: 1.5,
        textureOn: true,
        texture: 0.03,
        multiscaleOn: true,
        multiscale: 0.03,
        faceOn: true,
        face: 0.05,
        temporalBlurOn: true,
        temporalBlur: 0.3,
        lpcOn: true,
        lpc: 0.9,
        neuralOn: true,
        neural: 0.06,
      }),
    );
    expect(options.temporal_sub).toBe(1.2);
    expect(options.native_temporal).toBe(true);
    expect(options.fft_phase).toBe(0.4);
    expect(options.fft_mag).toBe(0.7);
    expect(options.dwt_detail).toBeUndefined();
    expect(options.nonint_ratio).toBe(0.02);
    expect(options.flow_disturb).toBe(1.5);
    expect(options.texture_inject).toBe(0.03);
    expect(options.complexity_trap).toBeCloseTo(0.075);
    expect(options.multiscale).toBe(0.03);
    expect(options.face_perturb).toBe(0.05);
    expect(options.temporal_blur).toBe(0.3);
    expect(options.lpc_attack).toBe(0.9);
    expect(options.copy_attack).toBe(0.06);
  });

  it("再生重写默认关闭，不携带任何再生参数", () => {
    const options = makeCleanOptions(panelState());
    expect(options.temporal_sub).toBeUndefined();
    expect(options.fft_phase).toBeUndefined();
    expect(options.fft_mag).toBeUndefined();
    expect(options.dwt_detail).toBeUndefined();
    expect(options.nonint_ratio).toBeUndefined();
    expect(options.flow_disturb).toBeUndefined();
    expect(options.texture_inject).toBeUndefined();
    expect(options.multiscale).toBeUndefined();
    expect(options.complexity_trap).toBeUndefined();
    expect(options.face_perturb).toBeUndefined();
    expect(options.temporal_blur).toBeUndefined();
    expect(options.lpc_attack).toBeUndefined();
    expect(options.copy_attack).toBeUndefined();
  });

  it("神经对抗统一映射到 DINOv2 判重描述子攻击", () => {
    const dino = makeCleanOptions(panelState({ neuralOn: true, neural: 0.06 }));
    expect(dino.copy_attack).toBe(0.06);
  });

  it("画质保护开启时携带 PSNR/SSIM 目标，关闭时不携带", () => {
    const options = makeCleanOptions(
      panelState({ qualityProtectOn: true, psnrTarget: 40, ssimTarget: 0.96 }),
    );
    expect(options.quality_protect).toBe(true);
    expect(options.psnr_target).toBe(40);
    expect(options.ssim_target).toBe(0.96);
    expect(makeCleanOptions(panelState()).quality_protect).toBeUndefined();
  });

  it("潜空间净化与嵌入域定向按开关映射为后端参数", () => {
    const options = makeCleanOptions(
      panelState({
        purifyOn: true,
        purifyStrength: 0.3,
        purifyDetail: 0.65,
        purifyDetailSigma: 1.8,
        purifyTemporal: 0.35,
        purifyMaxEdge: 320,
        purifyBatch: 8,
        embeddingOn: true,
        embeddingAttack: "auto",
        embeddingStrength: 0.5,
        embeddingVariant: "v2",
        embeddingAggressive: true,
        autoProfile: true,
      }),
    );
    expect(options.purify_strength).toBe(0.3);
    expect(options.purify_detail).toBe(0.65);
    expect(options.purify_detail_sigma).toBe(1.8);
    expect(options.purify_temporal).toBe(0.35);
    expect(options.purify_max_edge).toBe(320);
    expect(options.purify_batch).toBe(8);
    expect(options.embedding_attack).toBe("auto");
    expect(options.embedding_strength).toBe(0.5);
    expect(options.embedding_variant).toBe("v2");
    expect(options.embedding_aggressive).toBe(true);
    expect(options.auto_profile).toBe(true);
    const off = makeCleanOptions(panelState());
    expect(off.purify_strength).toBeUndefined();
    expect(off.purify_detail).toBeUndefined();
    expect(off.purify_detail_sigma).toBeUndefined();
    expect(off.purify_temporal).toBeUndefined();
    expect(off.purify_max_edge).toBeUndefined();
    expect(off.purify_batch).toBeUndefined();
    expect(off.embedding_attack).toBeUndefined();
    expect(off.embedding_strength).toBeUndefined();
    expect(off.embedding_variant).toBeUndefined();
    expect(off.embedding_aggressive).toBeUndefined();
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

describe("collusionOutputPath", () => {
  it("以第一个副本命名并标注共谋平均", () => {
    expect(collusionOutputPath("/素材/视频.mp4", "/导出", "20240101")).toBe(
      "/导出/视频_共谋平均_20240101.mp4",
    );
    expect(collusionOutputPath("/素材/视频.mov", "", "20240101")).toBe(
      "~/Documents/Cthulhu/视频_共谋平均_20240101.mp4",
    );
  });
});
