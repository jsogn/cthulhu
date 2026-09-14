"""潜空间净化原语的契约：回退、批处理、确定性与打包口径（不依赖真实权重）。"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cthulhu_backend.transform import purify


def _write_taesd_dir(path: Path) -> None:
    """写入最小可用的 TAESD 目录（config + 非空权重）。"""
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}")
    (path / "diffusion_pytorch_model.safetensors").write_bytes(b"x")


def test_device_preference_env_overrides_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    """设备偏好优先级：环境变量 > 设置项 > auto。"""
    from cthulhu_backend import db

    db.save_settings({"purify_device": "mps"})
    try:
        monkeypatch.delenv("CTHULHU_PURIFY_DEVICE", raising=False)
        assert purify.device_preference() == "mps"
        monkeypatch.setenv("CTHULHU_PURIFY_DEVICE", "cpu")
        assert purify.device_preference() == "cpu"
        monkeypatch.setenv("CTHULHU_PURIFY_DEVICE", "没这个设备")
        assert purify.device_preference() == "mps", "非法环境变量值应被忽略"
    finally:
        db.save_settings({"purify_device": "auto"})


def test_device_selection_honours_cpu_and_degrades(monkeypatch: pytest.MonkeyPatch) -> None:
    """显式要求 CPU 时不再走 GPU；指定了不可用的设备则退回 CPU。"""
    monkeypatch.setenv("CTHULHU_PURIFY_DEVICE", "cpu")
    assert purify._device_and_dtype()[0] == "cpu"

    import torch

    monkeypatch.setenv("CTHULHU_PURIFY_DEVICE", "cuda")
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert purify._device_and_dtype()[0] == expected


class FakeTaesd:
    """假 TAESD：encode/decode 是确定性的张量变换，并记录输入形状。"""

    device = "cpu"

    def __init__(self, mode: str = "shift") -> None:
        self.mode = mode
        self.encoded: list[tuple[int, ...]] = []
        self.decoded: list[np.ndarray] = []

    def parameters(self):
        import torch

        yield torch.zeros(1, dtype=torch.float32)

    def encode(self, tensor):
        self.encoded.append(tuple(tensor.shape))
        return SimpleNamespace(latents=tensor)

    def decode(self, latents):
        import torch

        self.decoded.append(latents.detach().cpu().numpy().copy())
        if self.mode == "flat":
            out = torch.full_like(latents, 0.5)
        else:
            out = torch.clamp(latents + 0.02, 0.0, 1.0)
        return SimpleNamespace(sample=out)


def _patch_engine(monkeypatch: pytest.MonkeyPatch, model: FakeTaesd) -> FakeTaesd:
    # 引擎路径需要 torch：轻量环境（CI 的 `uv sync` 不带 --extra purify）显式跳过，
    # 而不是让 purify 静默降级后报一堆看不懂的断言失败。
    pytest.importorskip("torch", reason="净化引擎测试需要 torch（uv sync --extra purify）")
    monkeypatch.setattr(purify, "check_available", lambda: (True, ""))
    monkeypatch.setattr(purify, "_load_taesd", lambda: model)
    return model


def test_strength_zero_is_identity() -> None:
    frames = np.random.default_rng(0).uniform(0, 1, (2, 16, 16, 3)).astype(np.float32)
    assert purify.purify_frames(frames, strength=0.0) is frames


def test_check_available_returns_reason() -> None:
    ok, reason = purify.check_available()
    assert isinstance(ok, bool)
    assert isinstance(reason, str)


def test_preflight_follows_taesd_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """预检只看 10MB 的 TAESD：内置/缓存即 ready，缺失且禁下载才不可用。"""
    monkeypatch.setattr(purify, "check_available", lambda: (True, ""))
    monkeypatch.setattr(purify, "taesd_cached", lambda: True)
    assert purify.preflight() == (True, "ready")
    monkeypatch.setattr(purify, "taesd_cached", lambda: False)
    monkeypatch.setattr(purify, "allow_download", lambda: False)
    ok, reason = purify.preflight()
    assert not ok and "TAESD" in reason
    monkeypatch.setattr(purify, "allow_download", lambda: True)
    assert purify.preflight() == (True, "will_download")


def test_missing_weights_falls_back_to_input(monkeypatch: pytest.MonkeyPatch) -> None:
    """TAESD 不可用且禁止下载时回退原帧，不抛异常、不破坏任务。"""
    monkeypatch.setattr(purify, "check_available", lambda: (True, ""))
    monkeypatch.setattr(purify, "allow_download", lambda: False)
    monkeypatch.setattr(purify, "taesd_cached", lambda: False)
    monkeypatch.setattr(purify, "_TAESD", None)
    purify.reset_failure()
    frames = np.random.default_rng(3).integers(0, 255, (2, 32, 32, 3), dtype=np.uint8)
    out = purify.purify_frames(frames, strength=0.1)
    np.testing.assert_array_equal(out, frames)
    assert purify.last_failure() is not None


def test_allow_download_defaults_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 方案：不设置环境变量时默认允许首次下载。"""
    monkeypatch.delenv(purify.ALLOW_DOWNLOAD_ENV, raising=False)
    assert purify.allow_download() is True
    monkeypatch.setenv(purify.ALLOW_DOWNLOAD_ENV, "0")
    assert purify.allow_download() is False


def test_model_dir_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv(purify.MODEL_DIR_ENV, str(tmp_path))
    assert purify.model_dir() == tmp_path
    assert purify.local_taesd_path() == tmp_path / purify.taesd_dir_name()


def test_model_status_shape() -> None:
    status = purify.model_status()
    for key in (
        "available",
        "reason",
        "model_id",
        "model_cached",
        "bundled",
        "allow_download",
        "device",
        "state",
        "progress",
        "error",
        "model_dir",
        "latent",
    ):
        assert key in status
    assert status["model_id"] == purify.TAESD_MODEL_ID
    assert status["latent"]["model_id"] == purify.TAESD_MODEL_ID


def test_install_model_reports_unavailable_without_deps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(purify, "check_available", lambda: (False, "missing dependency: torch"))
    status = purify.install_model()
    assert status["state"] == "unavailable"
    assert "torch" in status["error"]


def test_taesd_ready_requires_config_and_weights(tmp_path) -> None:
    path = tmp_path / "taesd"
    assert not purify.taesd_ready(path)
    path.mkdir()
    (path / "config.json").write_text("{}")
    assert not purify.taesd_ready(path)
    (path / "diffusion_pytorch_model.safetensors").write_bytes(b"x")
    assert purify.taesd_ready(path)


def test_bundled_taesd_is_preferred(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    bundled = tmp_path / "models"
    _write_taesd_dir(bundled / purify.taesd_dir_name())
    monkeypatch.setattr(purify, "_repo_bundled_model_dir", lambda: bundled)
    monkeypatch.delenv(purify.MODEL_DIR_ENV, raising=False)
    assert purify.bundled_taesd_path() == bundled / purify.taesd_dir_name()
    assert purify.model_dir() == bundled
    assert purify.taesd_cached() is True


def test_frozen_bundled_dir_uses_meipass(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    models = tmp_path / "models"
    models.mkdir()
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    assert purify.frozen_bundled_model_dir() == models


def test_packaging_fetch_script_matches_backend_readiness(tmp_path) -> None:
    """打包脚本的权重校验必须与后端加载口径一致。"""
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "fetch_purify_model", root / "packaging" / "fetch_purify_model.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    model = tmp_path / "taesd"
    _write_taesd_dir(model)
    assert module.is_ready(model)
    assert purify.taesd_ready(model)
    empty = tmp_path / "empty"
    empty.mkdir()
    assert not module.is_ready(empty)
    assert not purify.taesd_ready(empty)


def test_latent_engine_is_deterministic_and_batched(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _patch_engine(monkeypatch, FakeTaesd())
    frames = np.random.default_rng(1).integers(0, 255, (3, 32, 32, 3), dtype=np.uint8)
    first = purify.purify_frames(frames, strength=0.2, max_edge=16, batch=2, detail=0.0)
    second = purify.purify_frames(frames, strength=0.2, max_edge=16, batch=2, detail=0.0)
    assert first.dtype == np.uint8
    assert first.shape == frames.shape
    assert not np.array_equal(first, frames)
    np.testing.assert_array_equal(first, second)
    # 长边被压到 16 的倍数后送进编码器：32×32 → 16×16，批大小 2。
    # 两次调用各跑一遍批处理，形状完全一致（确定性 + 批处理生效）。
    assert model.encoded == [(2, 3, 16, 16), (1, 3, 16, 16)] * 2


def test_stop_signal_interrupts_purify(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_engine(monkeypatch, FakeTaesd())
    frames = np.zeros((2, 16, 16, 3), dtype=np.uint8)
    with pytest.raises(InterruptedError):
        purify.purify_frames(frames, strength=0.2, should_stop=lambda: True)


def test_control_plane_provides_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_engine(monkeypatch, FakeTaesd())
    seen: list[tuple[float, str]] = []
    purify.set_control(progress=lambda fraction, note: seen.append((fraction, note)))
    try:
        purify.purify_frames(np.zeros((2, 16, 16, 3), dtype=np.uint8), strength=0.2)
    finally:
        purify.clear_control()
    assert seen[-1][0] == 1.0
    assert "潜空间净化" in seen[-1][1]


def test_detail_reinjection_preserves_high_frequency(monkeypatch: pytest.MonkeyPatch) -> None:
    """细节回注应把重建结果的纹理拉回原帧，而不是把结果留在平坦图上。"""
    _patch_engine(monkeypatch, FakeTaesd(mode="flat"))
    frames = np.random.default_rng(12).integers(0, 255, (3, 32, 32, 3), dtype=np.uint8)
    out = purify.purify_frames(frames, strength=0.2, detail=1.0)
    flat = np.full_like(frames, 128)
    from scipy.ndimage import gaussian_filter

    def high_frequency(values: np.ndarray) -> np.ndarray:
        work = values.astype(np.float32)
        return work - gaussian_filter(work, sigma=(0.0, 1.2, 1.2, 0.0))

    assert float(np.mean(np.abs(high_frequency(out)))) > float(
        np.mean(np.abs(high_frequency(flat)))
    )
    # 回归：purified 已是 [0,1] float，不能再按 uint8 除一次 255 变成黑图。
    assert 80.0 < float(out.mean()) < 180.0


def test_detail_sigma_controls_reinjected_band(monkeypatch: pytest.MonkeyPatch) -> None:
    """σ 越大回注的中频越多：同一帧下 σ1.5 应比 σ0.6 更接近原帧。"""
    frames = np.random.default_rng(7).integers(0, 255, (3, 64, 64, 3), dtype=np.uint8)
    _patch_engine(monkeypatch, FakeTaesd(mode="flat"))
    narrow = purify.purify_frames(frames, strength=0.2, detail=1.0, detail_sigma=0.6)
    _patch_engine(monkeypatch, FakeTaesd(mode="flat"))
    wide = purify.purify_frames(frames, strength=0.2, detail=1.0, detail_sigma=1.5)
    reference = frames.astype(np.float32)

    def deviation(candidate: np.ndarray) -> float:
        return float(np.mean(np.abs(candidate.astype(np.float32) - reference)))

    assert deviation(wide) < deviation(narrow)


def test_auto_detail_sigma_scales_with_resolution() -> None:
    """σ 是像素单位，必须随分辨率放大，否则 1080p 会被糊掉（research §19.5）。"""
    assert purify.auto_detail_sigma(512, 512, 0.0) == pytest.approx(1.23, abs=0.01)
    assert purify.auto_detail_sigma(1080, 1920, 0.0) == pytest.approx(2.6, abs=0.05)
    assert purify.auto_detail_sigma(2160, 3840, 0.0) == purify.DETAIL_SIGMA_MAX
    assert purify.auto_detail_sigma(1080, 1920, 1.8) == 1.8


def test_temporal_subtraction_runs_before_rebuild(monkeypatch: pytest.MonkeyPatch) -> None:
    """时序增强应在送入潜空间重建之前生效。"""
    model = _patch_engine(monkeypatch, FakeTaesd())
    monkeypatch.setattr(
        purify.temporal, "estimate", lambda *args, **kwargs: SimpleNamespace(coherence=1.0)
    )

    def fake_subtract(frames, estimate, strength, mode):
        del estimate, mode
        work = frames.astype(np.float32) + 10.0 * strength
        return np.clip(work, 0, 255).astype(frames.dtype)

    monkeypatch.setattr(purify.temporal, "subtract", fake_subtract)
    frames = np.full((4, 16, 16, 3), 100, dtype=np.uint8)
    purify.purify_frames(frames, strength=0.2, temporal_strength=1.0)
    assert model.decoded
    assert round(float(model.decoded[0].mean()) * 255) == 110


def test_latent_engine_rebuilds_frames(tmp_path) -> None:
    """端到端：潜空间引擎走完整清洗管线，成片帧数不变且画面确实被重建。"""
    from cthulhu_backend import samples, services
    from cthulhu_backend.media import ffmpeg

    if not purify.check_available()[0]:
        pytest.skip("diffusers 不可用")
    if not purify.taesd_cached():
        pytest.skip("TAESD 权重未内置（打包/开发环境未准备）")
    frames = samples.make_video_frames(6, 160, 120, seed=11)
    source = tmp_path / "in.mp4"
    output = tmp_path / "out.mp4"
    ffmpeg.encode_video(frames, str(source), fps=30)

    report = services.run_desensitize(
        str(source),
        str(output),
        purify_strength=0.10,
        purify_max_edge=128,
        purify_batch=4,
        purify_detail=1.0,
        purify_detail_sigma=1.5,
        audio_remix=False,
        auto_profile=False,
        regrade=False,
        sharpness=False,
        color_restore=False,
        denoise=False,
        compute_metrics=False,
    )

    assert report["purify_note"].startswith("applied")
    assert "潜空间重建" in report["purify_note"]
    assert report["out_frames"] == len(frames)
    decoded, _ = ffmpeg.decode_video(str(output))
    assert decoded.shape[0] == len(frames)
    rebuilt = decoded.astype(np.float32)
    original = frames.astype(np.float32)
    assert not np.allclose(rebuilt, original), "潜空间引擎未改动画面"
    psnr = 10 * np.log10(255.0**2 / float(np.mean((rebuilt - original) ** 2)))
    assert psnr > 20, f"重建画质过低：{psnr:.1f}dB"


def test_weapon_pool_pickles_without_purify_control(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """回归：stop/pause 不得进入 TransformContext，否则多进程武器池 pickle 失败。"""
    from cthulhu_backend import samples, services
    from cthulhu_backend.media import ffmpeg

    monkeypatch.setenv("CTHULHU_PROCESS_WORKERS", "2")
    frames = samples.make_cut_video(1, 8, 160, 120, seed=0)
    source = tmp_path / "in.mp4"
    output = tmp_path / "out.mp4"
    ffmpeg.encode_video(frames, str(source), fps=30)
    report = services.run_desensitize(
        str(source),
        str(output),
        compute_metrics=False,
        audio_remix=False,
    )
    assert report["output"] == str(output)


def test_pipeline_records_purify_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """依赖缺位时净化降级原因应写进清洗任务结果。"""
    from cthulhu_backend import samples, services
    from cthulhu_backend.media import ffmpeg

    monkeypatch.setattr(purify, "check_available", lambda: (False, "test: no model"))
    frames = samples.make_cut_video(1, 8, 320, 240, seed=0)

    with tempfile.TemporaryDirectory() as tmp:
        inp = os.path.join(tmp, "in.mp4")
        out = os.path.join(tmp, "out.mp4")
        ffmpeg.encode_video(frames, inp, fps=30)
        report = services.run_desensitize(
            inp,
            out,
            purify_strength=0.3,
            audio_remix=False,
            regrade=False,
            color_restore=False,
            sharpness=False,
            compute_metrics=False,
        )
    assert report["purify_note"] == "skipped: test: no model"


def test_pipeline_records_mid_task_purify_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """预检通过但推理中途失败时，结果必须回写 fallback 原因。"""
    from cthulhu_backend import samples, services
    from cthulhu_backend.media import ffmpeg

    monkeypatch.setattr(purify, "check_available", lambda: (True, ""))
    monkeypatch.setattr(purify, "preflight", lambda: (True, "ready"))

    def fail_load():
        raise RuntimeError("boom")

    monkeypatch.setattr(purify, "_load_taesd", fail_load)
    frames = samples.make_cut_video(1, 8, 160, 120, seed=0)
    source = tmp_path / "in.mp4"
    output = tmp_path / "out.mp4"
    ffmpeg.encode_video(frames, str(source), fps=30)
    report = services.run_desensitize(
        str(source),
        str(output),
        purify_strength=0.3,
        audio_remix=False,
        regrade=False,
        color_restore=False,
        sharpness=False,
        compute_metrics=False,
    )
    assert report["purify_note"] == "fallback: boom"
