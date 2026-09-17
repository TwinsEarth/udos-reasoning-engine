"""
v5.5.0 学习型场景估计头 (冻结主模型, 端到端训练, 独立权重)
=========================================================
契约:
  1. 头末层零初始化 -> 初始对任意窗口输出全 0 (从近似场景盲平稳起步);
  2. differentiable_rollout 与 PhysicsPredictor.rollout 在 no_grad 下数值一致
     (训练优化的就是真实推理 rollout, 不是另一条路径);
  3. 短训练冒烟: 损失有限且下降, 头参数被真正更新;
  4. 已发布产物 checkpoints/scene_head_v5.5.0.pt 在 held-out(seed=2026) 上:
     弹簧 4 步 learned_recovery >= 0.8 (经典估计器仅 0.03), 整体 learned 不劣于
     显式真值条件, 弹簧 learned 显著优于 classical;
  5. reason 接线: 无显式场景参数但挂载头时 source=="learned_head" 且更准;
     未挂载头严格回退场景盲 (向后兼容)。
主预测员始终冻结, 52191 参数锚点不变。
"""

from pathlib import Path

import pytest
import torch

from udos.ctm_engine import CTMConfig
from udos.dynamics import build_parametric_dataset, traj_spring
from udos.gpm_engine import GPMConfig, TinyBaseModel
from udos.pce_format import PhysicsScene, PhysicalToken
from udos.persistence import load_predictor
from udos.reasoning import UDOSReasoningEngine
from udos.scene_head import (SceneEstimationHead, differentiable_rollout,
                             load_scene_head, train_scene_head)

CKPT = Path(__file__).resolve().parents[1] / "checkpoints" / "predictor_v4.3.9.pt"
HEAD_CKPT = Path(__file__).resolve().parents[1] / "checkpoints" / "scene_head_v5.5.0.pt"
W, H, DT, OMEGA, AMP = 6, 4, 0.5, 1.2, 1.5


@pytest.fixture(scope="module")
def predictor():
    p, _ = load_predictor(CKPT)
    p.eval()
    return p


def test_head_zero_initialized():
    head = SceneEstimationHead(window=6)
    x = torch.randn(3, 6, 6)
    out = head(x)
    assert out.shape == (3, 4)
    assert torch.count_nonzero(out) == 0          # 末层零初始化


def test_rejects_bad_shape():
    head = SceneEstimationHead(window=6)
    with pytest.raises(ValueError):
        head(torch.zeros(2, 5, 6))                # 窗口长度不符
    with pytest.raises(ValueError):
        head(torch.zeros(2, 6, 5))                # 末维不符


def test_differentiable_rollout_matches_inference(predictor):
    ds = build_parametric_dataset(n_per_kind=8, seed=1)
    P = ds.P[:16]
    with torch.no_grad():
        ref = predictor.rollout(ds.X[:16], H, scene_params=P)
        got = differentiable_rollout(predictor, ds.X[:16], P, H)
    assert got.shape == ref.shape
    assert torch.allclose(got, ref, atol=1e-6)


def test_differentiable_rollout_builds_grad_to_params(predictor):
    # 梯度穿过冻结主预测员回到场景参数 (头训练可行性的最小证明)
    ds = build_parametric_dataset(n_per_kind=8, seed=1)
    P = torch.zeros(16, 4, requires_grad=True)
    out = differentiable_rollout(predictor, ds.X[:16], P, H)
    out.sum().backward()
    assert P.grad is not None and torch.isfinite(P.grad).all()
    assert P.grad.abs().sum() > 0


def test_short_training_decreases_loss(predictor):
    ds = build_parametric_dataset(n_per_kind=16, seed=3)
    res = train_scene_head(predictor, ds, epochs=3, batch_size=64, seed=3)
    hist = res.loss_history
    assert len(hist) == 3
    assert all(torch.isfinite(torch.tensor(v)) for v in hist)
    assert hist[-1] < hist[0]
    out = res.head(ds.X[:4])
    assert torch.count_nonzero(out) > 0           # 参数已更新


@pytest.mark.skipif(not HEAD_CKPT.exists(),
                    reason="需先运行 scripts/train_scene_head.py 生成产物")
def test_shipped_head_artifact_quality(predictor):
    head, meta = load_scene_head(HEAD_CKPT)
    held = build_parametric_dataset(n_per_kind=128, seed=2026)

    def roll4(mask, params):
        with torch.no_grad():
            pred = predictor.rollout(held.X[mask], H, scene_params=params)
        return float(((pred - held.Y[mask]) ** 2).mean())

    with torch.no_grad():
        P_learn = head(held.X)
    # 整体
    blind = roll4(slice(None), None)
    explicit = roll4(slice(None), held.P)
    learned = roll4(slice(None), P_learn)
    assert learned < blind
    assert learned <= explicit * 1.10             # 达到/接近显式上界
    # 弹簧: 学习头根治经典估计器短板
    sm = held.kind_mask("spring")
    sb, se = roll4(sm, None), roll4(sm, held.P[sm])
    sl = roll4(sm, P_learn[sm])
    spring_rec = (sb - sl) / (sb - se)
    assert spring_rec >= 0.80
    assert sl < 0.5 * sb


def _engine(predictor, head=None):
    return UDOSReasoningEngine(
        CTMConfig(iterations=2, d_model=16, d_input=16, heads=2,
                  n_synch_out=8, n_synch_action=8, memory_length=4,
                  nlm_hidden=8, out_dims=16, certainty_threshold=0.0,
                  n_random_pairing_self=2),
        GPMConfig(feature_dim=16, latent_size=8, n_latents=4, lora_rank=4,
                  layer_indices=(0, 1), num_pre_head_layers=1, heads=2),
        base_model=TinyBaseModel(hidden=16, n_layers=2),
        predictor=predictor, scene_head=head)


def _spring_scene():
    traj = traj_spring(W + H, DT, amp=AMP, omega=OMEGA, phi=0.0)
    scene = PhysicsScene(scene_id="spring", duration=W + H)
    for i, (pos, vel) in enumerate(traj[:W]):
        scene.add(PhysicalToken(object_id="obj-a", timestamp=i,
                                position=pos, velocity=vel))
    gt_pos = torch.tensor([p for p, _ in traj[W:W + H]])
    return scene, gt_pos


def _pos_mse(result, gt_pos):
    pred = torch.tensor([s["position"] for s in result.future_states])
    return ((pred - gt_pos) ** 2).mean().item()


@pytest.mark.skipif(not HEAD_CKPT.exists(),
                    reason="需先运行 scripts/train_scene_head.py 生成产物")
def test_reason_uses_attached_head_without_explicit_params(predictor):
    head, _ = load_scene_head(HEAD_CKPT)
    scene, gt = _spring_scene()

    eng_blind = _engine(predictor, head=None)
    eng_head = _engine(predictor, head=head)
    r_blind = eng_blind.reason(scene, horizon=H)
    r_head = eng_head.reason(scene, horizon=H)

    assert r_blind.predictor_conditioned is False      # 未挂载: 场景盲
    assert r_blind.scene_params_source is None
    assert r_head.predictor_conditioned is True
    assert r_head.scene_params_source == "learned_head"
    assert len(r_head.scene_params) == 4
    assert _pos_mse(r_head, gt) < 0.6 * _pos_mse(r_blind, gt)


def test_attach_and_detach_head(predictor):
    eng = _engine(predictor)
    assert eng.scene_head is None
    head = SceneEstimationHead(window=6)
    eng.attach_scene_head(head)
    assert eng.scene_head is head
    eng.attach_scene_head(None)
    assert eng.scene_head is None
