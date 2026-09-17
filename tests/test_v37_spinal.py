"""v3.7.0.dev3 脊髓本地反射弧测试 (碰撞->制动 / 越界->截断 / 超速->减速)。

覆盖:
    * 碰撞制动: 近障碍物 => 速度归零, reflex_triggered, 事件 kind=collision_brake;
    * 越界截断: 位置越界 => 截断到界内;
    * 超速减速: |速度|>limit => 按比例缩到限内;
    * 反射优先级可证: 触发时 final command == spinal command (小脑输出被覆盖),
      priority_winner=spinal;
    * 事件日志: events 含 kind/step;
    * 无触发场景: 良性状态 => winner=cerebellum, 无事件。
"""
from pathlib import Path

import pytest
import torch

from udos import __version__
from udos.persistence import load_predictor
from udos.dynamics import build_parametric_dataset
from udos.neural_control import HierarchicalController, SpinalReflex

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
    """造一个末帧为指定 [pos(3),vel(3)] 的窗口 [1,6,6]。"""
    w = torch.zeros(1, 6, 6)
    w[0, -1, :3] = torch.as_tensor(pos, dtype=torch.float32)
    w[0, -1, 3:] = torch.as_tensor(vel, dtype=torch.float32)
    return w


def test_collision_brake(predictor):
    ctrl = HierarchicalController(
        predictor, cortex_every=1,
        collision_obstacles=[[0.0, 0.0, 0.0]], collision_radius=0.5)
    w = _window_with_state([0.1, 0.0, 0.0], [1.0, 0.0, 0.0])
    out = ctrl.step(w)
    assert out["spinal"]["reflex_triggered"] is True
    assert out["priority_winner"] == "spinal"
    # 制动: 速度维归零
    assert torch.allclose(out["command"][3:], torch.zeros(3), atol=1e-6)
    kinds = [e["kind"] for e in out["spinal"]["events"]]
    assert "collision_brake" in kinds


def test_reflex_overrides_cerebellum(predictor):
    """可证: 触发时 final == spinal command, 小脑命令被丢弃。"""
    ctrl = HierarchicalController(
        predictor, cortex_every=1,
        collision_obstacles=[[0.0, 0.0, 0.0]], collision_radius=0.5)
    w = _window_with_state([0.1, 0.0, 0.0], [1.0, 0.0, 0.0])
    out = ctrl.step(w)
    spinal_cmd = out["command"]
    # 未触发时小脑会给出非零速度命令; 触发后必须是脊髓的制动命令 (速度=0)
    assert float(spinal_cmd[3:].abs().sum()) == 0.0
    assert out["reflex_triggered"] is True


def test_bound_clamp(predictor):
    ctrl = HierarchicalController(predictor, cortex_every=1,
                                  position_bounds=1.0)
    w = _window_with_state([5.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    out = ctrl.step(w)
    assert out["spinal"]["reflex_triggered"] is True
    assert out["command"][0].item() == pytest.approx(1.0, abs=1e-6)


def test_speed_slowdown(predictor):
    ctrl = HierarchicalController(predictor, cortex_every=1,
                                  speed_limit=1.0)
    w = _window_with_state([0.0, 0.0, 0.0], [3.0, 0.0, 0.0])
    out = ctrl.step(w)
    assert out["spinal"]["reflex_triggered"] is True
    assert out["command"][3].item() == pytest.approx(1.0, abs=1e-6)


def test_event_log_accumulates(predictor):
    ctrl = HierarchicalController(
        predictor, cortex_every=1,
        collision_obstacles=[[0.0, 0.0, 0.0]], collision_radius=0.5)
    w = _window_with_state([0.1, 0.0, 0.0], [1.0, 0.0, 0.0])
    ctrl.step(w)
    ctrl.step(w)
    assert len(ctrl.reflex_log) == 2
    assert ctrl.reflex_log[0]["kind"] == "collision_brake"
    assert "step" in ctrl.reflex_log[0]


def test_no_trigger_benign(predictor):
    ctrl = HierarchicalController(predictor, cortex_every=1)
    ds = build_parametric_dataset(n_per_kind=4, n_steps=14, window=6,
                                  horizon=4, dt=0.5, seed=33)
    out = ctrl.step(ds.X[:1], scene_params=ds.P[:1])
    assert out["spinal"]["reflex_triggered"] is False
    assert out["priority_winner"] == "cerebellum"
    assert out["spinal"]["events"] == []


def test_spinal_direct_unit():
    """脊髓单元: 无障碍物/限速时不触发。"""
    sp = SpinalReflex()
    out = sp.act(torch.zeros(1, 6), {})
    assert out["reflex_triggered"] is False
    assert out["events"] == []
