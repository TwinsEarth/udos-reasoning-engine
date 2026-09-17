"""v3.8.0.dev1 世界模型调度器: WMScheduler 预算分配 / 回退 / 复用 LatentWorldModel。

覆盖:
    * 按优先级贪婪分配想象预算 (高优先获更多步);
    * 预算上限严格 (used <= cap); 不足预算守卫;
    * horizon==1 即回退真实单步预测;
    * 同级按 agent_id 字典序 (确定性);
    * 复用 LatentWorldModel.imagine 逐体调度, 轨迹形状正确;
    * 单体/空场景守卫; 零外挂 (不改 predictor 权重)。
"""
from pathlib import Path

import torch

from udos import __version__, load_predictor
from udos.multi_agent import MultiAgentScene
from udos.wm_scheduler import WMScheduler
from udos.world_model import LatentWorldModel

ROOT = Path(__file__).resolve().parents[1]
CKPT = str(ROOT / "checkpoints" / "predictor_v3.8.0.pt")


def test_version():
    assert __version__ == "5.5.5"


def _scene():
    sc = MultiAgentScene()
    sc.add_agent("a_lo", [0, 0, 0, 0, 0, 0], priority=9)
    sc.add_agent("b_hi", [1, 0, 0, 0, 0, 0], priority=1)
    sc.add_agent("c_mid", [2, 0, 0, 0, 0, 0], priority=5)
    return sc


def test_priority_allocation():
    sc = _scene()
    sched = WMScheduler(total_imagination_budget=6, base_horizon=1, max_horizon=4)
    alloc = sched.allocate(sc)
    # 3 体 base 各 1 = 3; 剩 3 步集中给最高优先 b_hi 至 max_horizon=4
    assert alloc["b_hi"] == 4      # +3 占满
    assert alloc["c_mid"] == 1     # 未获额外
    assert alloc["a_lo"] == 1
    assert sum(alloc.values()) == 6


def test_budget_cap_respected():
    sc = _scene()
    sched = WMScheduler(total_imagination_budget=100, base_horizon=1, max_horizon=2)
    alloc = sched.allocate(sc)
    # max_horizon=2 => 全部触顶, used = 3*2 = 6
    assert all(v == 2 for v in alloc.values())


def test_insufficient_budget_guard():
    sc = _scene()
    try:
        WMScheduler(total_imagination_budget=2).allocate(sc)  # < 3 体各 1
        assert False
    except ValueError:
        pass


def test_fallback_when_budget_tight():
    """预算恰为 n*base => 全部 horizon=1 => 全部回退真实预测。"""
    sc = _scene()
    sched = WMScheduler(total_imagination_budget=3, base_horizon=1)
    alloc = sched.allocate(sc)
    assert all(v == 1 for v in alloc.values())


def test_empty_scene_guard():
    sc = MultiAgentScene()
    try:
        WMScheduler(5).allocate(sc)
        assert False
    except ValueError:
        pass


def test_bad_budget_guard():
    try:
        WMScheduler(0)
        assert False
    except ValueError:
        pass


def test_tiebreak_deterministic():
    sc = MultiAgentScene()
    sc.add_agent("b", [0, 0, 0, 0, 0, 0], priority=5)
    sc.add_agent("a", [1, 0, 0, 0, 0, 0], priority=5)
    s = WMScheduler(total_imagination_budget=3, base_horizon=1)
    a1 = s.allocate(sc)
    a2 = s.allocate(sc)
    assert a1 == a2
    # 同级: 字典序 a 先得额外步
    assert a1["a"] == 2 and a1["b"] == 1


def test_imagine_reuses_wm_and_fallback_flag():
    model, _ = load_predictor(CKPT)
    wm = LatentWorldModel(model)
    sc = MultiAgentScene()
    sc.add_agent("hi", [0, 0, 0, 0, 0, 0], priority=1)
    sc.add_agent("lo", [1, 0, 0, 0, 0, 0], priority=9)
    sched = WMScheduler(total_imagination_budget=3, base_horizon=1, max_horizon=3)
    alloc = sched.allocate(sc)
    windows = {i: torch.randn(1, 6, 6) for i in sc.ids}
    rep = sched.imagine(wm, sc, windows)
    assert rep["used_budget"] == sum(alloc.values())
    # hi 得到 >1 步想象; lo 回退真实
    assert rep["agents"]["hi"]["horizon"] == alloc["hi"]
    assert rep["agents"]["lo"]["fell_back_real"] is True
    assert rep["agents"]["lo"]["traj_shape"] == [1, 6]


def test_imagine_bitwise_anchor_on_fallback():
    """horizon=1 回退真实: 与 predictor.predict_next 逐位一致 (PWM 锚点)。"""
    model, _ = load_predictor(CKPT)
    wm = LatentWorldModel(model)
    sc = MultiAgentScene()
    sc.add_agent("only", [0, 0, 0, 0, 0, 0])
    sched = WMScheduler(total_imagination_budget=1, base_horizon=1)
    sched.allocate(sc)
    w = torch.randn(1, 6, 6)
    rep = sched.imagine(wm, sc, {"only": w})
    with torch.no_grad():
        ref = model.predict_next(w)[0]
    got = torch.tensor(rep["agents"]["only"]["trajectory"]).reshape(6)
    assert torch.allclose(got, ref, atol=1e-5)


def test_imagine_missing_window_guard():
    model, _ = load_predictor(CKPT)
    wm = LatentWorldModel(model)
    sc = MultiAgentScene()
    sc.add_agent("a", [0, 0, 0, 0, 0, 0])
    sched = WMScheduler(1).allocate(sc)
    try:
        WMScheduler(1).imagine(wm, sc, {})   # 无 window
        assert False
    except ValueError:
        pass
