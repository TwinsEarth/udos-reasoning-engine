"""
v2.6.0+dev1 因果链 / 反事实推演单元测试
==========================================
锚点纪律: intervention=None 时 counterfactual 与 baseline 逐位一致 (且等于
predictor.rollout); ATE 非负; 场景参数/速度干预后轨迹按物理合理方向改变。
"""
import pytest
import torch

from udos import __version__
from udos.persistence import load_predictor
from udos.counterfactual import CounterfactualEngine
from udos.dynamics import build_parametric_dataset, traj_spring, traj_uniform, _raw

CKPT = "checkpoints/predictor_v2.6.0.pt"


@pytest.fixture(scope="module")
def engine():
    model, _ = load_predictor(CKPT)
    return CounterfactualEngine(model)


@pytest.fixture(scope="module")
def spring_batch():
    ds = build_parametric_dataset(n_per_kind=8, n_steps=14, window=6,
                                  horizon=4, dt=0.5, seed=101)
    mask = ds.kind_mask("spring")
    idx = torch.nonzero(mask, as_tuple=False).flatten()[:2]
    return ds.X[idx], ds.P[idx]


def test_version():
    assert __version__ == "5.5.5"


def test_zero_intervention_equals_baseline(engine, spring_batch):
    """intervention=None / 空 dict => cf==baseline 逐位, 且 == predictor.rollout。"""
    xb, pb = spring_batch
    out = engine.counterfactual(xb, horizon=4, scene_params=pb, intervention=None)
    assert torch.allclose(out["counterfactual"], out["baseline"], atol=1e-7)
    # 与 predictor.rollout 逐位
    raw_roll = engine.predictor.rollout(xb, 4, scene_params=pb)
    assert torch.allclose(out["baseline"], raw_roll, atol=1e-7)
    assert out["ate_mean"] == pytest.approx(0.0, abs=1e-12)
    assert torch.allclose(out["final_state_diff"], torch.zeros_like(
        out["final_state_diff"]), atol=1e-7)


def test_scene_param_intervention(engine, spring_batch):
    """覆盖 spring_omega 后反事实轨迹与基线不同 (频率改变 => 轨迹发散)。"""
    xb, pb = spring_batch
    omega0 = float(pb[0, 2])
    out = engine.counterfactual(xb, horizon=4, scene_params=pb,
                                 intervention={"scene_params": {2: omega0 * 1.6}})
    assert not torch.allclose(out["counterfactual"], out["baseline"], atol=1e-5)
    # 干预越往后累积越大 (rollout 自回归误差累积)
    diff = ((out["counterfactual"] - out["baseline"]) ** 2).mean(dim=(0, 2))
    assert float(diff[-1]) >= float(diff[0]) - 1e-9


def test_ate_nonnegative(engine, spring_batch):
    xb, pb = spring_batch
    out = engine.counterfactual(xb, horizon=4, scene_params=pb,
                                 intervention={"scene_params": {2: 1.2}})
    assert torch.all(out["ate_by_step"] >= -1e-12)
    assert out["ate_mean"] > 0.0


def test_velocity_override(engine):
    """覆盖初始速度后, 位置轨迹斜率改变 (匀速段位置位移量显著不同)。"""
    traj = traj_uniform(10, 0.5, v0=1.0, x0=0.0)
    raw = torch.tensor([_raw(p, v) for p, v in traj], dtype=torch.float32)
    window = raw[:6].unsqueeze(0)
    pb = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    base = engine.rollout_intervention(window, horizon=5, scene_params=pb)
    cf = engine.rollout_intervention(
        window, horizon=5, scene_params=pb,
        intervention={"velocity_override": torch.tensor([3.0, 0.0, 0.0])})
    # x 轴位置末值应明显不同 (速度从 1.0 -> 3.0; 学习模型非解析, 阈值放宽)
    assert abs(float(cf[0, -1, 0] - base[0, -1, 0])) > 0.2
