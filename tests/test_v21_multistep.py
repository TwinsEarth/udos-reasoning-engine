"""v2.1.0 多步滚动推演 / 场景条件联合训练 / 物理一致性 回归测试。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402
import torch  # noqa: E402

from udos.dynamics import (build_parametric_dataset, kinematic_residual,
                           SCENE_PARAM_DIM, SCENE_PARAM_NAMES)  # noqa: E402
from udos.ctm_engine import CTMConfig  # noqa: E402
from udos.training import PhysicsPredictor, CTMTrainer, TrainConfig  # noqa: E402
from udos.evaluation import evaluate_predictor  # noqa: E402
from udos.persistence import save_predictor, load_predictor  # noqa: E402
from udos.reasoning import UDOSReasoningEngine  # noqa: E402
from udos.ctm_engine import CTMConfig as Cfg  # noqa: E402
from udos.gpm_engine import GPMConfig  # noqa: E402


def _small_cfg():
    return CTMConfig(iterations=6, d_model=40, d_input=24, heads=2,
                     n_synch_out=12, n_synch_action=8, memory_length=6,
                     nlm_hidden=12, out_dims=24, certainty_threshold=0.0)


# ---------- M1 数据: 参数化多步数据集 ----------
def test_parametric_dataset_shapes():
    # 每轨迹起点数 = n_steps-window-horizon+1 = 5; 4 类 * 8 轨迹 * 5 = 160
    ds = build_parametric_dataset(n_per_kind=8, n_steps=14, window=6,
                                  horizon=4, dt=0.5)
    assert ds.X.shape == (160, 6, 6)
    assert ds.Y.shape == (160, 4, 6)
    assert ds.P.shape == (160, SCENE_PARAM_DIM)
    assert SCENE_PARAM_DIM == 4 and len(SCENE_PARAM_NAMES) == 4
    assert ds.horizon == 4
    tr, te = ds.split(0.75)
    assert len(tr) + len(te) == len(ds)
    # batches 产出 (X, P, Y) 三元组
    xb, pb, yb = next(iter(ds.batches(16)))
    assert xb.ndim == 3 and pb.ndim == 2 and yb.ndim == 3
    assert set(ds.class_names) == {"uniform", "accel", "spring", "collision"}
    mk = ds.kind_mask("spring")
    assert mk.dtype == torch.bool and int(mk.sum()) == 40


def test_kinematic_residual_formula():
    """匀速段一阶欧拉严格成立; 人为偏离残差>0; 用前一帧速度。"""
    dt = 0.5
    prev = torch.tensor([[0.0, 0, 0, 1.0, 0, 0]])
    exact = torch.tensor([[0.5, 0, 0, 1.0, 0, 0]])   # x'=x+v*dt
    bad = torch.tensor([[2.0, 0, 0, 1.0, 0, 0]])
    assert kinematic_residual(exact, prev, dt).item() == pytest.approx(0, abs=1e-6)
    assert kinematic_residual(bad, prev, dt).item() > 0.5


# ---------- M1 模型: scene_encoder 与自由 rollout ----------
def test_scene_encoder_and_rollout_shapes():
    m = PhysicsPredictor(_small_cfg(), scene_param_dim=4)
    assert m.scene_encoder is not None
    assert m.ctm.cfg.scene_dim == 32
    X = torch.randn(5, 6, 6)
    P = torch.randn(5, 4)
    one = m.predict_next(X, scene_params=P)
    assert one.shape == (5, 6)
    roll = m.rollout(X, 4, scene_params=P)
    assert roll.shape == (5, 4, 6)
    # rollout 首步等于单步预测
    assert torch.allclose(roll[:, 0, :], one, atol=1e-5)
    # 无 scene_param_dim 的旧模型 rollout 仍可用 (向后兼容)
    legacy = PhysicsPredictor(_small_cfg())
    assert legacy.scene_encoder is None
    assert legacy.rollout(X, 3).shape == (5, 3, 6)


def test_multistep_train_step_runs():
    """teacher-forcing 多步 + 物理正则路径不报错, phys_violation 被记录。"""
    ds = build_parametric_dataset(n_per_kind=6, n_steps=14, window=6, horizon=4)
    tr, te = ds.split(0.8)
    m = PhysicsPredictor(_small_cfg(), scene_param_dim=4)
    hist = CTMTrainer(m, TrainConfig(epochs=2, phys_weight=0.05)).train(tr, te)
    assert len(hist.phys_violation) == 2
    assert len(hist.eval_mse) == 2
    # 兼容 v2.0 单步数据集
    from udos.dynamics import build_dynamics_dataset
    old = build_dynamics_dataset(n_per_kind=6, n_steps=12, window=6)
    otr, ote = old.split(0.8)
    m2 = PhysicsPredictor(_small_cfg())
    CTMTrainer(m2, TrainConfig(epochs=1)).train(otr, ote)


# ---------- 评估体系 ----------
def test_evaluation_report_structure():
    ds = build_parametric_dataset(n_per_kind=6, n_steps=14, window=6, horizon=4)
    tr, te = ds.split(0.8)
    m = PhysicsPredictor(_small_cfg(), scene_param_dim=4)
    rep = evaluate_predictor(m, te)
    assert rep["horizon"] == 4
    assert len(rep["rollout_mse_curve"]) == 4
    assert set(rep["per_kind_mse"]) == set(te.class_names)
    assert {"conditioned_mse", "unconditioned_mse",
            "condition_gain_x"} <= set(rep["ablation"])
    assert "overall" in rep["kinematic_residual"]


# ---------- M2 核心: 场景条件联合训练后, 条件优于无条件 (消融) ----------
@pytest.mark.slow
def test_scene_conditioning_ablation_gain():
    torch.manual_seed(0)
    ds = build_parametric_dataset(n_per_kind=20, n_steps=14, window=6, horizon=4)
    tr, te = ds.split(0.8)
    cfg = CTMConfig(iterations=8, d_model=64, d_input=32, heads=4,
                    n_synch_out=16, n_synch_action=8, memory_length=8,
                    nlm_hidden=16, out_dims=32, certainty_threshold=0.0)
    m = PhysicsPredictor(cfg, scene_param_dim=4)
    assert float(m.ctm.scene_gate.detach()) == 0.0   # 零门控初始等价无条件
    CTMTrainer(m, TrainConfig(epochs=30, lr=3e-3)).train(tr)
    rep = evaluate_predictor(m, te)
    gain = rep["ablation"]["condition_gain_x"]
    assert rep["ablation"]["conditioned_mse"] < rep["ablation"]["unconditioned_mse"]
    assert gain > 1.3, f"场景条件增益不足: {gain}"
    assert abs(float(m.ctm.scene_gate.detach())) > 1e-3  # 门控确实被训练激活


# ---------- 持久化: 带 scene_encoder 的存载逐元素一致 ----------
def test_save_load_scene_predictor(tmp_path):
    m = PhysicsPredictor(_small_cfg(), scene_param_dim=4)
    X = torch.randn(3, 6, 6)
    P = torch.randn(3, 4)
    with torch.no_grad():
        before = m.predict_next(X, scene_params=P)
    p = save_predictor(m, tmp_path / "p21.pt")
    m2, meta = load_predictor(p)
    assert m2.scene_param_dim == 4 and m2.scene_encoder is not None
    with torch.no_grad():
        after = m2.predict_next(X, scene_params=P)
    assert torch.allclose(before, after, atol=1e-6)
    assert meta["scene_param_dim"] == 4


# ---------- M1 协同引擎多步 reason ----------
def test_reason_multistep_future_states(tiny_scene):
    eng = UDOSReasoningEngine(
        Cfg(iterations=4, d_model=32, d_input=16, n_synch_out=8,
            n_synch_action=8, memory_length=4, nlm_hidden=8, out_dims=16,
            certainty_threshold=0.0),
        GPMConfig(feature_dim=16, latent_size=16, n_latents=4, lora_rank=2,
                  target_modules=("down_proj",), layer_indices=(0,)),
        scene_conditioning=False)
    eng.attach_predictor(PhysicsPredictor(_small_cfg()))
    r1 = eng.reason(tiny_scene, horizon=1)
    assert r1.future_states is None and r1.predicted_state is not None
    r4 = eng.reason(tiny_scene, horizon=4)
    assert len(r4.future_states) == 4
    assert "position" in r4.future_states[0] and "velocity" in r4.future_states[0]
    assert r4.summary()["future_states"] == r4.future_states


# ---------- 服务: --checkpoint 预加载 ----------
def test_service_preload_checkpoint(tmp_path):
    from udos.server import UDOSService
    m = PhysicsPredictor(_small_cfg(), scene_param_dim=4)
    ckpt = save_predictor(m, tmp_path / "pre.pt", metrics={"trained_mse": 0.1})
    svc = UDOSService(checkpoint=str(ckpt))
    h = svc.health()
    assert h["predictor_trained"] is True
    # 默认无 checkpoint 仍为未训练 (向后兼容)
    assert UDOSService().health()["predictor_trained"] is False
