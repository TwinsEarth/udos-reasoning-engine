"""v3.8.5 边界测试: 多体零智能体 / 孪生极端配置 / 闭环模块 / 调度预算为零。

覆盖:
    * 多体零智能体守卫;
    * 孪生极端配置 (单 agent / 大 bounds / 零障碍);
    * 闭环空 window / 非有限守卫;
    * 调度预算为零 / 负预算守卫;
    * 协调器负 buffer 守卫; 单 agent 边界。
"""
import pytest
import torch

from udos import __version__, load_predictor
from udos.multi_agent import AgentCoordinator, MultiAgentScene
from udos.wm_scheduler import WMScheduler
from udos.closed_loop import ClosedLoopOrchestrator
from udos.digital_twin import DigitalTwinScene

CKPT = "checkpoints/predictor_v3.8.0.pt"


def test_version():
    assert __version__ == "5.5.5"


def test_zero_agents_guard():
    sc = MultiAgentScene()
    for op in (sc.states, sc.positions, sc.require_agents):
        if callable(op) and op.__name__ == "require_agents":
            with pytest.raises(ValueError):
                op()
        else:
            with pytest.raises(ValueError):
                op()
    with pytest.raises(ValueError):
        AgentCoordinator().resolve(sc)


def test_twin_extreme_config():
    # 单智能体
    t = DigitalTwinScene(n_agents=1, n_obstacles=0, seed=0, bounds=0.5)
    assert t.agents.n_agents == 1
    t.step()
    # 大量障碍
    t2 = DigitalTwinScene(n_agents=2, n_obstacles=50, seed=1, bounds=100.0)
    assert len(t2.obstacles) == 50


def test_twin_bad_config():
    with pytest.raises(ValueError):
        DigitalTwinScene(n_agents=-1)
    with pytest.raises(ValueError):
        DigitalTwinScene(n_obstacles=-5)


def test_scheduler_zero_budget_guard():
    sc = MultiAgentScene()
    sc.add_agent("a", [0, 0, 0, 0, 0, 0])
    with pytest.raises(ValueError):
        WMScheduler(0)
    with pytest.raises(ValueError):
        WMScheduler(-3)
    # 预算不足: 2 体需 >= 2*base=2, 给 1 应报错
    sc2 = MultiAgentScene()
    sc2.add_agent("a", [0, 0, 0, 0, 0, 0])
    sc2.add_agent("b", [1, 0, 0, 0, 0, 0])
    with pytest.raises(ValueError):
        WMScheduler(1).allocate(sc2)   # < 2*base=2


def test_coordinator_negative_buffer():
    with pytest.raises(ValueError):
        AgentCoordinator(buffer=-1.0)


def test_closed_loop_bad_window():
    model, _ = load_predictor(CKPT)
    orch = ClosedLoopOrchestrator(model)
    with pytest.raises(ValueError):
        orch.step(torch.randn(1, 0, 6))     # 空窗
    with pytest.raises(ValueError):
        orch.step(torch.randn(1, 6, 6) * float("inf"))


def test_scheduler_max_below_base_guard():
    with pytest.raises(ValueError):
        WMScheduler(5, base_horizon=2, max_horizon=1)
