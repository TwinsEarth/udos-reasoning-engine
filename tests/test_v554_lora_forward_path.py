"""v5.5.4 消除 LoRA 死路 (TDD): 被注入基座真正参与场景前向编码。

契约:
  1. GPM 生成 LoRA -> 注入基座 -> 基座前向必须真正被调用, 且 live 档
     (init_scaler_b_zero=False) 注入后输出相对注入前发生非零改变;
  2. reset 后基座前向逐位回到注入前基线 (reset_error≈0);
  3. internalize/reset 消息与状态可观测注入差/复位误差;
  4. scaler_B=0 训练档下前向闭环照样运行, 但注入差≈0 (LoRA 零扰动约定);
  5. reason 结果反映 LoRA 前向通道是否激活。
"""
from __future__ import annotations

import pytest
import torch

from udos.ctm_engine import CTMConfig
from udos.gpm_engine import GPMConfig, TinyBaseModel
from udos.reasoning import UDOSReasoningEngine


def _engine(live: bool = True):
    base = TinyBaseModel(hidden=32, n_layers=2)
    # live 档 scaler_B=1 (init_scaler_b_zero=False); 训练档 B=0
    gpm = GPMConfig(feature_dim=32, latent_size=32, n_latents=8, lora_rank=4,
                    layer_indices=(0, 1), num_pre_head_layers=1, heads=2,
                    init_scaler_b_zero=not live)
    ctm = CTMConfig(iterations=2, d_model=32, d_input=32, heads=2,
                    n_synch_out=8, n_synch_action=8, memory_length=4,
                    nlm_hidden=8, out_dims=32, certainty_threshold=0.0,
                    n_random_pairing_self=2)
    return UDOSReasoningEngine(ctm, gpm, base_model=base)


def test_self_test_live_injection_changes_output_and_reset_is_exact(tiny_scene):
    eng = _engine(live=True)
    rep = eng.lora_path_self_test(tiny_scene)
    assert rep["injection_delta"] > 1e-6          # live LoRA 真正改变基座前向
    assert rep["reset_error"] == pytest.approx(0.0, abs=1e-6)
    assert eng.injector.active is None            # 自测后已复位, 不留补丁
    assert rep["patched_modules"] > 0


def test_self_test_deterministic(tiny_scene):
    eng = _engine(live=True)
    r1 = eng.lora_path_self_test(tiny_scene)
    r2 = eng.lora_path_self_test(tiny_scene)
    assert r1["injection_delta"] == pytest.approx(r2["injection_delta"], rel=1e-6)


def test_base_scene_encoding_reflects_injection(tiny_scene):
    eng = _engine(live=True)
    enc0, active0 = eng.base_scene_encoding(tiny_scene)
    assert active0 is False
    eng.internalize_scene(tiny_scene)
    enc1, active1 = eng.base_scene_encoding(tiny_scene)
    assert active1 is True
    assert float((enc1 - enc0).norm()) > 1e-6     # 注入后场景编码确实改变


def test_internalize_and_reset_messages_and_exact_restore(tiny_scene):
    eng = _engine(live=True)
    msg = eng.internalize_scene(tiny_scene)
    assert "已内化" in msg
    assert "注入差" in msg
    assert eng.injector.active is not None
    feats = eng._scene_features(tiny_scene)
    baseline = eng._last_base_baseline.clone()
    rmsg = eng.reset_scene(tiny_scene.scene_id)
    assert "复位误差" in rmsg
    assert eng.injector.active is None
    # 复位后前向与注入前基线逐位一致
    with torch.no_grad():
        now = eng.base_model(feats)
    assert torch.allclose(now, baseline, atol=1e-6)


def test_zero_scaler_convention_path_runs_but_no_delta(tiny_scene):
    eng = _engine(live=False)
    rep = eng.lora_path_self_test(tiny_scene)
    assert rep["injection_delta"] == pytest.approx(0.0, abs=1e-6)
    assert rep["reset_error"] == pytest.approx(0.0, abs=1e-6)
    assert rep["patched_modules"] > 0             # 补丁照样打过 (前向闭环存在)


def test_reason_reports_lora_forward_state(tiny_scene):
    eng = _engine(live=True)
    r0 = eng.reason(tiny_scene, horizon=1)
    assert r0.lora_injection_active is False
    eng.internalize_scene(tiny_scene)
    r1 = eng.reason(tiny_scene, horizon=1)
    assert r1.lora_injection_active is True
    assert r1.lora_patched_modules > 0
    eng.reset_scene(tiny_scene.scene_id)
    r2 = eng.reason(tiny_scene, horizon=1)
    assert r2.lora_injection_active is False
    assert r2.lora_patched_modules == 0


def test_default_engine_uses_live_demo_lora(tiny_scene):
    """引擎自建默认 GPM 走 live 演示档, LoRA 前向通道非死路。"""
    eng = UDOSReasoningEngine()
    rep = eng.lora_path_self_test(tiny_scene)
    assert rep["injection_delta"] > 1e-6
