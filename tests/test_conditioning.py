"""v2.0.0 GPM->CTM 场景条件真耦合测试。"""

import pytest
import torch

from udos import (CTMConfig, CTMPhysicsEngine, GPMConfig,
                  PhysicsHypernetwork, UDOSReasoningEngine,
                  PhysicsPredictor, PhysicalToken, PhysicsScene)


def cond_cfg(**kw):
    base = dict(iterations=4, d_model=32, d_input=16, n_synch_out=8,
                n_synch_action=6, memory_length=4, out_dims=16,
                scene_dim=12, certainty_threshold=0.0)
    base.update(kw)
    return CTMConfig(**base)


def test_zero_gate_initially_equals_unconditioned():
    # 向后兼容契约: gate 零初始化, 开启条件化初始等价于一代无条件前向
    m = CTMPhysicsEngine(cond_cfg()); m.eval()
    x = torch.randn(3, 5, 16); ctx = torch.randn(3, 12)
    with torch.no_grad():
        p0, _, _, _ = m(x)
        p1, _, _, _ = m(x, scene_context=ctx)
    assert float(m.scene_gate.detach()) == 0.0
    assert torch.allclose(p0, p1, atol=1e-6)


def test_same_context_same_output():
    m = CTMPhysicsEngine(cond_cfg()); m.eval()
    with torch.no_grad():
        m.scene_gate.fill_(1.0)
        x = torch.randn(2, 5, 16); c = torch.randn(2, 12)
        a, _, _, _ = m(x, scene_context=c)
        b, _, _, _ = m(x, scene_context=c)
    assert torch.allclose(a, b, atol=1e-6)


def test_different_context_changes_output_after_gate():
    # 反例: 若 ctx 被忽略, 不同场景条件产出相同结果
    m = CTMPhysicsEngine(cond_cfg()); m.eval()
    x = torch.randn(2, 5, 16)
    with torch.no_grad():
        m.scene_gate.fill_(1.0)
        a, _, _, _ = m(x, scene_context=torch.randn(2, 12))
        b, _, _, _ = m(x, scene_context=torch.randn(2, 12))
    assert not torch.allclose(a, b, atol=1e-5)


def test_context_without_scene_dim_raises():
    m = CTMPhysicsEngine(cond_cfg(scene_dim=None))
    with pytest.raises(AssertionError):
        m(torch.randn(2, 5, 16), scene_context=torch.randn(2, 12))


def test_gpm_scene_embedding_shape_and_determinism(tiny_scene):
    gpm = PhysicsHypernetwork(GPMConfig(feature_dim=16, latent_size=16,
                                        n_latents=4,
                                        init_scaler_b_zero=False))
    e1 = gpm.scene_embedding(tiny_scene)
    e2 = gpm.scene_embedding(tiny_scene)  # 持久编码器, 必须可复现
    assert e1.shape == (16,)
    assert torch.allclose(e1, e2, atol=1e-6)


def _eng(scene_conditioning):
    return UDOSReasoningEngine(
        CTMConfig(iterations=3, d_model=32, d_input=16, n_synch_out=8,
                  n_synch_action=6, memory_length=4, out_dims=16,
                  certainty_threshold=0.0),
        GPMConfig(feature_dim=16, latent_size=16, n_latents=4,
                  init_scaler_b_zero=False),
        scene_conditioning=scene_conditioning)


def test_engine_conditioning_flag(tiny_scene):
    eng_on = _eng(True)
    assert eng_on.ctm.scene_proj is not None
    r = eng_on.reason(tiny_scene)
    assert r.scene_conditioned is True

    eng_off = _eng(False)
    assert eng_off.ctm.scene_proj is None
    r2 = eng_off.reason(tiny_scene)
    assert r2.scene_conditioned is False


def test_attached_predictor_gives_interpretable_state(tiny_scene):
    eng = _eng(True)
    pred = PhysicsPredictor(CTMConfig(
        iterations=3, d_model=32, d_input=16, n_synch_out=8,
        n_synch_action=6, memory_length=4, out_dims=16,
        certainty_threshold=0.0))
    eng.attach_predictor(pred)
    r = eng.reason(tiny_scene, query="下一时刻?")
    st = r.predicted_state
    assert set(st) == {"position", "velocity"}
    assert len(st["position"]) == 3 and len(st["velocity"]) == 3
    assert "predicted_next_state" in r.summary()
