"""v3.7.3 Patch 精修边界测试。

覆盖 (按 VERSION_PLAN_3.7 第 10 节点):
    * 零/负频率与负延迟预算显式 ValueError;
    * 反射冲突 (碰撞+越界同步触发, 多事件记录, 输出有限);
    * PID 数值稳定 (大误差不产生 inf/nan);
    * 大脑无候选: None 退化 vs [] 报错;
    * cortex_every=1 与 reset 清空反射日志。
"""
from pathlib import Path

import pytest
import torch

from udos import __version__
from udos.persistence import load_predictor
from udos.dynamics import build_parametric_dataset
from udos.neural_control import (
    HierarchicalController, CerebellumTracker, SpinalReflex, ControlLayer)

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "checkpoints" / "predictor_v3.7.0.pt"


def test_version():
    assert __version__ == "5.5.5"


@pytest.fixture(scope="module")
def predictor():
    m, _ = load_predictor(str(CKPT))
    m.eval()
    return m


def test_zero_negative_frequency_guard():
    with pytest.raises(ValueError):
        ControlLayer("x", frequency_hz=0.0, latency_budget_ms=1.0, priority_rank=0)
    with pytest.raises(ValueError):
        ControlLayer("x", frequency_hz=-1.0, latency_budget_ms=1.0, priority_rank=0)
    with pytest.raises(ValueError):
        ControlLayer("x", frequency_hz=1.0, latency_budget_ms=0.0, priority_rank=0)


def test_reflex_conflict_same_step(predictor):
    """碰撞+越界同步触发: 两类事件都记录, 输出有限且安全。"""
    ctrl = HierarchicalController(
        predictor, cortex_every=1,
        collision_obstacles=[[0.0, 0.0, 0.0]], collision_radius=1.0,
        position_bounds=0.05)
    w = torch.zeros(1, 6, 6)
    w[0, -1, :3] = torch.tensor([0.1, 0.0, 0.0])   # 近障碍 + 越界
    w[0, -1, 3:] = torch.tensor([2.0, 0.0, 0.0])
    out = ctrl.step(w)
    assert out["spinal"]["reflex_triggered"] is True
    kinds = {e["kind"] for e in out["spinal"]["events"]}
    assert "collision_brake" in kinds
    assert "bound_clamp" in kinds
    assert bool(torch.isfinite(out["command"]).all())
    # 碰撞制动: 速度归零
    assert float(out["command"][3:].abs().sum()) == 0.0


def test_pid_numerically_stable():
    """大目标误差 + 大 kp 不产生 inf/nan。"""
    cb = CerebellumTracker(kp=10.0, ki=5.0, kd=1.0, alpha=0.9)
    s = torch.zeros(1, 6)
    target = torch.tensor([1e3, -1e3, 0.0, 0.0, 0.0, 0.0])
    out = cb.act(s, {"target": target})
    assert bool(torch.isfinite(out["command"]).all())


def test_cortex_no_candidate_none_vs_empty(predictor):
    ctrl = HierarchicalController(predictor, cortex_every=1)
    ds = build_parametric_dataset(n_per_kind=4, n_steps=14, window=6,
                                  horizon=4, dt=0.5, seed=99)
    w, sp = ds.X[:1], ds.P[:1]
    out = ctrl.step(w, scene_params=sp)   # candidate_actions=None
    assert out["cortex"]["planned"] is False
    with pytest.raises(ValueError):
        ctrl.step(w, scene_params=sp, candidate_actions=[])


def test_cortex_every_one_and_reset_log(predictor):
    ctrl = HierarchicalController(
        predictor, cortex_every=1,
        collision_obstacles=[[0.0, 0.0, 0.0]], collision_radius=1.0)
    w = torch.zeros(1, 6, 6)
    w[0, -1, 0] = 0.1
    ctrl.step(w)
    ctrl.step(w)
    assert len(ctrl.reflex_log) == 2
    ctrl.reset()
    assert len(ctrl.reflex_log) == 0
    assert ctrl.run_counts["cortex"] == 0
