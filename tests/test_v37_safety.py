"""v3.7.0.dev5 安全边界与反射优先级测试。

覆盖:
    * 优先级矩阵: 脊髓(0)<小脑(1)<大脑(2), 在 step 输出中显式暴露;
    * 安全违例制动: 碰撞 -> safety_state=reflex, winner=spinal;
    * 状态机: 反射后下一良性步回到 nominal;
    * 正常场景不触发: safety_state=nominal;
    * 大脑目标经 decision.safety_boundary 校验 (target_safe 字段)。
"""
from pathlib import Path

import pytest
import torch

from udos import __version__
from udos.persistence import load_predictor
from udos.dynamics import build_parametric_dataset
from udos.neural_control import HierarchicalController, ControlLayer

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "checkpoints" / "predictor_v3.7.0.pt"


def test_version():
    assert __version__ == "5.5.5"


@pytest.fixture(scope="module")
def predictor():
    m, _ = load_predictor(str(CKPT))
    m.eval()
    return m


def _window_with_state(pos, vel):
    w = torch.zeros(1, 6, 6)
    w[0, -1, :3] = torch.as_tensor(pos, dtype=torch.float32)
    w[0, -1, 3:] = torch.as_tensor(vel, dtype=torch.float32)
    return w


def test_priority_matrix_exposed(predictor):
    ctrl = HierarchicalController(predictor, cortex_every=1)
    ds = build_parametric_dataset(n_per_kind=4, n_steps=14, window=6,
                                  horizon=4, dt=0.5, seed=55)
    out = ctrl.step(ds.X[:1], scene_params=ds.P[:1])
    pm = out["priority_matrix"]
    assert pm["spinal"] < pm["cerebellum"] < pm["cortex"]
    assert pm == ControlLayer.PRIORITY


def test_safety_violation_reflex(predictor):
    ctrl = HierarchicalController(
        predictor, cortex_every=1,
        collision_obstacles=[[0.0, 0.0, 0.0]], collision_radius=0.5)
    w = _window_with_state([0.1, 0.0, 0.0], [1.0, 0.0, 0.0])
    out = ctrl.step(w)
    assert out["safety_state"] == "reflex"
    assert out["priority_winner"] == "spinal"


def test_state_machine_recovers_to_nominal(predictor):
    ctrl = HierarchicalController(
        predictor, cortex_every=1,
        collision_obstacles=[[0.0, 0.0, 0.0]], collision_radius=0.5)
    bad = _window_with_state([0.1, 0.0, 0.0], [1.0, 0.0, 0.0])
    ctrl.step(bad)
    assert ctrl.safety_state == "reflex"
    # 下一良性步回到 nominal
    ds = build_parametric_dataset(n_per_kind=4, n_steps=14, window=6,
                                  horizon=4, dt=0.5, seed=56)
    out = ctrl.step(ds.X[:1], scene_params=ds.P[:1])
    assert out["safety_state"] == "nominal"


def test_normal_nominal(predictor):
    ctrl = HierarchicalController(predictor, cortex_every=1)
    ds = build_parametric_dataset(n_per_kind=4, n_steps=14, window=6,
                                  horizon=4, dt=0.5, seed=57)
    out = ctrl.step(ds.X[:1], scene_params=ds.P[:1])
    assert out["safety_state"] == "nominal"
    assert out["priority_winner"] == "cerebellum"


def test_cortex_target_safe_field(predictor):
    ctrl = HierarchicalController(predictor, cortex_every=1)
    ds = build_parametric_dataset(n_per_kind=4, n_steps=14, window=6,
                                  horizon=4, dt=0.5, seed=58)
    w, sp = ds.X[:1], ds.P[:1]
    cands = [{"candidate_state": (w[0, -1, :] + 0.05)}]
    out = ctrl.step(w, scene_params=sp, candidate_actions=cands)
    # 大脑规划步应带 target_safe 字段 (bool)
    assert "target_safe" in out["cortex"]
    assert isinstance(out["cortex"]["target_safe"], bool)
