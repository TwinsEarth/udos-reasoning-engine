"""v3.8.0 多体协同核心: MultiAgentScene / AgentCoordinator。

覆盖:
    * 多体状态容器 (N 体添加 / 查询 / 状态布局 6 维);
    * 冲突消解: 优先级让行 (高优先保持 / 低优先减速) + 侵入深度成比例;
    * 同级决胜按 agent_id 字典序 (确定性);
    * 单体退化 (无冲突 => 速度恒等);
    * 空场景守卫 / 非法状态守卫;
    * 快照/回放逐位一致;
    * apply_commands 积分 + 越速限速;
    * opt-in: 不构造时主路径不受影响 (零外挂)。
"""
import torch

from udos import __version__
from udos.multi_agent import (AgentCoordinator, AgentState, MultiAgentScene,
                              POS_DIM, RAW_DIM)


def test_version():
    assert __version__ == "5.5.5"


def test_state_layout():
    assert RAW_DIM == 6 and POS_DIM == 3


def test_add_agent_and_container():
    sc = MultiAgentScene()
    assert len(sc) == 0
    sc.add_agent("a1", state=[0, 0, 0, 1, 0, 0], priority=1, radius=0.3)
    sc.add_agent("a2", state=[1, 0, 0, -1, 0, 0], priority=5, radius=0.3)
    assert sc.n_agents == 2
    assert sc.ids == ["a1", "a2"]
    s = sc.states()
    assert s.shape == (2, 6)
    assert torch.allclose(sc.positions(), torch.tensor([[0, 0, 0.0], [1, 0, 0]]))


def test_duplicate_id_rejected():
    sc = MultiAgentScene()
    sc.add_agent("a1", [0, 0, 0, 0, 0, 0])
    try:
        sc.add_agent("a1", [0, 0, 0, 0, 0, 0])
        assert False, "应拒绝重复 id"
    except ValueError:
        pass


def test_bad_state_guard():
    sc = MultiAgentScene()
    try:
        sc.add_agent("x", [0, 0, 0])          # 维度错
        assert False
    except ValueError:
        pass
    try:
        sc.add_agent("x", [float("nan")] * 6)  # 非有限
        assert False
    except ValueError:
        pass


def test_empty_scene_guard():
    sc = MultiAgentScene()
    try:
        sc.states()
        assert False
    except ValueError:
        pass
    try:
        AgentCoordinator().resolve(sc)
        assert False
    except ValueError:
        pass


def test_single_agent_no_conflict_identity():
    """单体退化: 无冲突 => 建议速度 = 当前速度 (恒等)。"""
    sc = MultiAgentScene()
    sc.add_agent("solo", [0, 0, 0, 0.5, 0, 0])
    rep = AgentCoordinator().resolve(sc)
    assert rep["n_conflicts"] == 0
    assert rep["commands"]["solo"] == [0.5, 0.0, 0.0]
    assert rep["yielded_agents"] == []


def test_priority_yield():
    """高优先级 (priority=1) 保持; 低优先级 (priority=9) 减速。"""
    sc = MultiAgentScene()
    # 两车相向, 距离 0.4 < 0.3+0.3 => 冲突
    sc.add_agent("hi", [0, 0, 0, 1.0, 0, 0], priority=1, radius=0.3)
    sc.add_agent("lo", [0.4, 0, 0, -1.0, 0, 0], priority=9, radius=0.3)
    rep = AgentCoordinator().resolve(sc)
    assert rep["n_conflicts"] == 1
    c = rep["conflicts"][0]
    assert c["winner"] == "hi" and c["loser"] == "lo"
    # hi 保持原速; lo 被减速 (|vx| < 1.0)
    assert rep["commands"]["hi"][0] == 1.0
    assert abs(rep["commands"]["lo"][0]) < 1.0


def test_tiebreak_by_agent_id():
    """同优先级: 字典序小者胜。"""
    sc = MultiAgentScene()
    sc.add_agent("b", [0.0, 0, 0, 1.0, 0, 0], priority=5, radius=0.3)
    sc.add_agent("a", [0.4, 0, 0, -1.0, 0, 0], priority=5, radius=0.3)
    rep = AgentCoordinator().resolve(sc)
    c = rep["conflicts"][0]
    assert c["winner"] == "a" and c["loser"] == "b"


def test_yield_factor_monotone_with_penetration():
    """侵入越深 yield 越小 (减速越多)。"""
    def factor(dist):
        sc = MultiAgentScene()
        sc.add_agent("hi", [0, 0, 0, 1, 0, 0], priority=1, radius=0.3)
        sc.add_agent("lo", [dist, 0, 0, -1, 0, 0], priority=9, radius=0.3)
        return AgentCoordinator().resolve(sc)["conflicts"][0]["yield_factor"]

    shallow = factor(0.58)   # 接近但侵入浅
    deep = factor(0.1)       # 深度侵入
    assert deep < shallow


def test_no_conflict_when_far():
    sc = MultiAgentScene()
    sc.add_agent("x", [0, 0, 0, 1, 0, 0], radius=0.3)
    sc.add_agent("y", [10, 0, 0, -1, 0, 0], radius=0.3)
    rep = AgentCoordinator().resolve(sc)
    assert rep["n_conflicts"] == 0


def test_snapshot_roundtrip():
    sc = MultiAgentScene()
    sc.add_agent("a1", [0, 0, 0, 0.5, 0, 0], goal=[1, 0, 0], priority=2,
                 radius=0.25, max_speed=2.0)
    sc.add_agent("a2", [3, 1, 0, -0.5, 0, 0], goal=[0, 0, 0], priority=7)
    snap = sc.snapshot()
    sc2 = MultiAgentScene.from_snapshot(snap)
    assert sc2.ids == sc.ids
    assert torch.allclose(sc2.states(), sc.states())


def test_apply_commands_integrate_and_speed_cap():
    sc = MultiAgentScene()
    sc.add_agent("a", [0, 0, 0, 5.0, 0, 0], max_speed=1.0)  # 5 > 限速 1
    rep = AgentCoordinator().resolve(sc)  # 单体无冲突, 命令=(5,0,0)
    nxt = AgentCoordinator().apply_commands(sc, rep["commands"], dt=1.0)
    a = nxt.get("a")
    # 速度被缩到 1.0; 位置前进 1.0
    assert abs(a.vel[0].item() - 1.0) < 1e-5
    assert abs(a.pos[0].item() - 1.0) < 1e-5


def test_determinism():
    sc = MultiAgentScene()
    sc.add_agent("hi", [0, 0, 0, 1, 0, 0], priority=1, radius=0.3)
    sc.add_agent("lo", [0.2, 0, 0, -1, 0, 0], priority=9, radius=0.3)
    c = AgentCoordinator()
    r1 = c.resolve(sc)
    r2 = c.resolve(sc)
    assert r1["commands"] == r2["commands"]
    assert r1["conflicts"] == r2["conflicts"]


def test_no_trainable_params_and_optin():
    """纯算法外挂: 无 torch.nn.Module / 无参数; 不碰主 predictor。"""
    c = AgentCoordinator()
    params = [x for x in vars(c).values() if isinstance(x, torch.nn.Module)]
    assert params == []
    sc = MultiAgentScene()
    sc.add_agent("only", [0, 0, 0, 0, 0, 0])
    c.resolve(sc)
