"""v3.8.0.dev3 合成数字孪生场景: DigitalTwinScene 生成 / 快照 / 回放。

覆盖:
    * 参数化生成 (n_agents/n_obstacles/seed/bounds);
    * 同 seed 确定性逐位可复现;
    * 快照 -> 回放状态逐位一致;
    * 一步 step 推进 (冲突消解+积分);
    * 非法配置守卫 (n_agents<1 / n_obstacles<0 / bounds<=0);
    * 与 multi_agent 接口一致; 空快照/错类型守卫。
"""
import torch

from udos import __version__
from udos.digital_twin import DigitalTwinScene


def test_version():
    assert __version__ == "5.5.5"


def test_parametric_generation():
    t = DigitalTwinScene(n_agents=4, n_obstacles=6, seed=0)
    assert t.agents.n_agents == 4
    assert len(t.obstacles) == 6
    assert t.agents.ids == [f"ag{k}" for k in range(4)]
    assert t.summary()["n_obstacles"] == 6


def test_seed_determinism():
    a = DigitalTwinScene(n_agents=3, n_obstacles=2, seed=123)
    b = DigitalTwinScene(n_agents=3, n_obstacles=2, seed=123)
    assert torch.allclose(a.agents.states(), b.agents.states())
    for x, y in zip(a.obstacles, b.obstacles):
        assert torch.allclose(x, y)


def test_different_seed_differs():
    a = DigitalTwinScene(n_agents=3, seed=1)
    b = DigitalTwinScene(n_agents=3, seed=2)
    assert not torch.allclose(a.agents.states(), b.agents.states())


def test_snapshot_replay_roundtrip():
    t = DigitalTwinScene(n_agents=4, n_obstacles=3, seed=7)
    t.step()
    t.step()
    snap = t.snapshot()
    r = DigitalTwinScene.from_snapshot(snap)
    assert r.step_index == t.step_index
    assert torch.allclose(r.agents.states(), t.agents.states())
    for x, y in zip(r.obstacles, t.obstacles):
        assert torch.allclose(x, y)


def test_step_advances():
    t = DigitalTwinScene(n_agents=5, seed=3)
    before = t.agents.positions().clone()
    t.step()
    assert t.step_index == 1
    # 位置必然推进 (至少有速度)
    assert not torch.allclose(before, t.agents.positions())


def test_invalid_config_guard():
    try:
        DigitalTwinScene(n_agents=0)
        assert False
    except ValueError:
        pass
    try:
        DigitalTwinScene(n_obstacles=-1)
        assert False
    except ValueError:
        pass
    try:
        DigitalTwinScene(bounds=0)
        assert False
    except ValueError:
        pass


def test_from_snapshot_bad_type():
    try:
        DigitalTwinScene.from_snapshot({"kind": "other"})
        assert False
    except ValueError:
        pass


def test_zero_obstacles_ok():
    t = DigitalTwinScene(n_agents=2, n_obstacles=0, seed=0)
    assert len(t.obstacles) == 0
    t.step()
    assert t.step_index == 1
