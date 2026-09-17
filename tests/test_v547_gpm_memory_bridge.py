"""
v5.4.7 GPM 记忆桥 (scene_embedding -> 主预测员 CTM 的零初始化可学习加性桥)
=========================================================================
5.4.6 让主预测员吃上 4 维显式物理参数; GPM 的 scene_embedding 仍只喂内部
未训练的演示 CTM。本版新增一条独立、零初始化、可学习的加性桥, 把 GPM 场景
嵌入投影到主预测员 CTM 的 scene_dim, 与 scene_encoder(P) 相加。

硬契约:
  * 桥权重/偏置零初始化 => 接入瞬间对主预测零影响 (与 5.4.6 逐位兼容),
    包括"场景盲"路径 (零偏置必须短路, 不得把 ctx 从 None 变成零张量);
  * 桥权重一旦非零必须真实改变主预测 (反 LoRA 死路: 注入却从不 forward);
  * 桥是独立 nn.Module, 不进 PhysicsPredictor.state_dict, 主锚点 52191 不变;
  * 真实增益在 v5.5.0 联合训练后产生, 本版只交付机制与可观测性。
"""

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from udos.ctm_engine import CTMConfig
from udos.dynamics import traj_spring
from udos.gpm_engine import GPMConfig, TinyBaseModel
from udos.gpm_memory_bridge import GPMSceneBridge
from udos.pce_format import PhysicalToken, PhysicsScene
from udos.persistence import load_predictor
from udos.reasoning import UDOSReasoningEngine

CKPT = Path(__file__).resolve().parents[1] / "checkpoints" / "predictor_v4.3.9.pt"
W, H, DT, OMEGA, AMP = 6, 4, 0.5, 1.2, 1.5


@pytest.fixture(scope="module")
def predictor():
    p, _ = load_predictor(CKPT)
    p.eval()
    return p


def _engine(predictor, use_bridge=True):
    return UDOSReasoningEngine(
        CTMConfig(iterations=2, d_model=16, d_input=16, heads=2,
                  n_synch_out=8, n_synch_action=8, memory_length=4,
                  nlm_hidden=8, out_dims=16, certainty_threshold=0.0,
                  n_random_pairing_self=2),
        GPMConfig(feature_dim=16, latent_size=8, n_latents=4, lora_rank=4,
                  layer_indices=(0, 1), num_pre_head_layers=1, heads=2),
        base_model=TinyBaseModel(hidden=16, n_layers=2),
        predictor=predictor, use_gpm_bridge=use_bridge)


def _spring_scene(omega, conditioned=True):
    traj = traj_spring(W + H, DT, amp=AMP, omega=omega, phi=0.0)
    meta = {"scene_params": {"spring_omega": omega}} if conditioned else {}
    scene = PhysicsScene(scene_id="spring", duration=W + H, metadata=meta)
    for i, (pos, vel) in enumerate(traj[:W]):
        scene.add(PhysicalToken(object_id="obj-a", timestamp=i,
                                position=pos, velocity=vel))
    return scene


def _future_tensor(result):
    return torch.tensor([s["position"] + s["velocity"]
                         for s in result.future_states])


def test_bridge_zero_init_outputs_zero():
    bridge = GPMSceneBridge(latent_dim=8, scene_dim=32)
    assert torch.count_nonzero(bridge.proj.weight) == 0
    assert torch.count_nonzero(bridge.proj.bias) == 0
    out = bridge(torch.randn(3, 8))
    assert out.shape == (3, 32)
    assert torch.count_nonzero(out) == 0


def test_zero_bridge_conditioned_bitidentical_to_546(predictor):
    eng = _engine(predictor)
    scene = _spring_scene(OMEGA, conditioned=True)
    r = eng.reason(scene, horizon=H)  # 首次 reason 懒建桥
    assert eng.gpm_scene_bridge is not None
    assert eng.gpm_scene_bridge.proj.weight.shape == (32, 8)
    assert r.gpm_bridge_active is True
    assert r.gpm_bridge_norm == 0.0
    direct = predictor.rollout(eng._tokens_raw(scene), H,
                               scene_params=torch.tensor([[0., 0., OMEGA, 0.]]))[0]
    # reason 对外状态做 6 位小数 JSON 舍入, 故在 1e-6 容差内逐位一致
    assert torch.allclose(_future_tensor(r), direct, atol=1e-6)


def test_zero_bridge_blind_stays_blind(predictor):
    # 零偏置不得把场景盲 ctx 从 None 变成零张量 (否则经 scene_proj 偏置失真)
    eng = _engine(predictor)
    scene = _spring_scene(OMEGA, conditioned=False)
    r = eng.reason(scene, horizon=H)
    legacy = predictor.rollout(eng._tokens_raw(scene), H)[0]
    assert r.predictor_conditioned is False
    assert r.gpm_bridge_norm == 0.0
    assert torch.allclose(_future_tensor(r), legacy, atol=1e-6)


def test_nonzero_bridge_changes_prediction(predictor):
    # 反 LoRA 死路: 桥权重非零后必须真实改变主预测输出
    eng = _engine(predictor)
    scene = _spring_scene(OMEGA, conditioned=True)
    r0 = eng.reason(scene, horizon=H)
    base = _future_tensor(r0)
    nn.init.normal_(eng.gpm_scene_bridge.proj.weight, std=0.1)
    nn.init.normal_(eng.gpm_scene_bridge.proj.bias, std=0.1)
    r1 = eng.reason(scene, horizon=H)
    moved = _future_tensor(r1)
    assert r1.gpm_bridge_norm > 0.0
    assert not torch.allclose(base, moved, atol=1e-5)


def test_bridge_independent_of_predictor_anchor(predictor):
    # 主 predictor 可学习参数锚点不变; 桥独立成模块、可独立存取
    n_params = sum(p.numel() for p in predictor.parameters() if p.requires_grad)
    assert n_params == 52191
    bridge = GPMSceneBridge(8, 32)
    with torch.no_grad():
        bridge.proj.weight.fill_(0.25)
    sd = bridge.state_dict()
    bridge2 = GPMSceneBridge(8, 32)
    bridge2.load_state_dict(sd)
    emb = torch.randn(1, 8)
    assert torch.allclose(bridge(emb), bridge2(emb), atol=1e-7)
    assert "gpm_scene_bridge" not in dict(predictor.named_modules())


def test_disable_bridge_keeps_546_path(predictor):
    eng = _engine(predictor, use_bridge=False)
    scene = _spring_scene(OMEGA, conditioned=True)
    r = eng.reason(scene, horizon=H)
    assert eng.gpm_scene_bridge is None
    assert r.gpm_bridge_active is False
    direct = predictor.rollout(eng._tokens_raw(scene), H,
                               scene_params=torch.tensor([[0., 0., OMEGA, 0.]]))[0]
    assert torch.allclose(_future_tensor(r), direct, atol=1e-6)


def test_resolve_context_bias_rules():
    # 小型带场景 predictor 直接验证 _resolve_context 的加性/短路规则
    pred = PhysicsPredictorMini()
    P = torch.tensor([[0., 0., OMEGA, 0.]])
    base_ctx = pred.scene_encoder(P)
    bias = torch.full((1, 32), 0.1)
    # params + 非零 bias => 相加
    ctx = pred._resolve_context(None, P, scene_bias=bias)
    assert torch.allclose(ctx, base_ctx + bias, atol=1e-6)
    # 仅非零 bias => 成为 ctx
    ctx2 = pred._resolve_context(None, None, scene_bias=bias)
    assert torch.allclose(ctx2, bias, atol=1e-6)
    # 零 bias + 无 params => 短路为 None (场景盲兼容)
    ctx3 = pred._resolve_context(None, None, scene_bias=torch.zeros(1, 32))
    assert ctx3 is None
    # 零 bias + params => 等于 params ctx (浮点 +0 恒等)
    ctx4 = pred._resolve_context(None, P, scene_bias=torch.zeros(1, 32))
    assert torch.allclose(ctx4, base_ctx, atol=1e-7)


# 小型带场景编码器的 predictor 替身 (只测 _resolve_context, 不跑 CTM)
def PhysicsPredictorMini():
    from udos.training import PhysicsPredictor
    cfg = CTMConfig(iterations=2, d_model=32, d_input=16, heads=2,
                    n_synch_out=8, n_synch_action=8, memory_length=4,
                    nlm_hidden=16, out_dims=16, scene_dim=32,
                    certainty_threshold=0.0)
    return PhysicsPredictor(cfg, scene_param_dim=4)
