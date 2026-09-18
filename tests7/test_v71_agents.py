"""v7.1 多 Agent 协同内核契约测试（契约 → 反例 → 证据）。

覆盖：共享记忆 CAS/血缘、资源碰撞、分类拆解、协调器异步 DAG、预测客观选优、
加权集成、动作排序、代码补丁仅胜者合并、分支隔离、跨协调器共享、AL 门禁。
所有预测用真实 checkpoint 在 held-out(seed=2026) 数据上验证。
"""
import asyncio
import os

import numpy as np
import pytest

from udos7.agents import (AL, CapabilityGate, CloudTransport, CoordinatorAgent,
                          GateError, LocalTransport, SharedMemory)
from udos7.agents.memory import CollisionError
from udos7.agents.types import Claim, Task, TaskKind, TaskStatus
from udos7.agents.workers import ClassifierExpert, DecomposerExpert, AgentContext
from udos7.dynamics import build_split

CKPT = os.environ.get("UDOS7_CKPT", "checkpoints7/worldmodel_v7.0.3.pt")


@pytest.fixture(scope="module")
def model():
    from udos7.persistence import load_worldmodel
    m, _ = load_worldmodel(CKPT)
    return m


@pytest.fixture(scope="module")
def batch():
    ds = build_split(2026, n_traj_per_kind=2)
    return ds.X[0].tolist(), ds.Y[0].tolist()


# --------------------------------------------------------------------------
# 共享记忆：CAS 与事件血缘
# --------------------------------------------------------------------------
def test_cas_versions_and_conflict():
    mem = SharedMemory()
    v1 = mem.put("ns", "k", 1, owner="a")
    assert v1 == 1
    ok, v2 = mem.cas("ns", "k", 1, 2, owner="a")
    assert ok and v2 == 2
    # 反例：基于过期版本的 CAS 必须失败（否则会覆盖他人写入）
    ok, cur = mem.cas("ns", "k", 1, 99, owner="b")
    assert not ok and cur == 2
    assert mem.get("ns", "k") == 2


def test_event_log_is_append_only_and_lineaged():
    from udos7.agents.types import Event
    mem = SharedMemory()
    a = mem.append(Event(kind="x", author="a"))
    b = mem.append(Event(kind="y", author="b", parent_ids=[a.id]))
    assert len(mem.events()) == 2
    assert mem.event_by_id(b.id).parent_ids == [a.id]


# --------------------------------------------------------------------------
# 资源碰撞
# --------------------------------------------------------------------------
def test_claim_write_collision_and_readers_ok():
    mem = SharedMemory()
    mem.declare(Claim("file:m.py", "read", "a1"))
    mem.declare(Claim("file:m.py", "read", "a2"))  # 两读者不冲突
    mem.declare(Claim("file:m.py", "write", "a1"))
    # 反例：a2 再写同一文件必须报碰撞
    with pytest.raises(CollisionError):
        mem.declare(Claim("file:m.py", "write", "a2"))


# --------------------------------------------------------------------------
# 分类与拆解
# --------------------------------------------------------------------------
def test_classify_and_decompose_modes(model):
    ctx = AgentContext(model=model)
    cls = ClassifierExpert("c", ctx)
    assert cls.classify("预测下一阶段", {"window": []}) == TaskKind.PREDICT
    assert cls.classify("比较候选动作", {"actions": []}) == TaskKind.ACTION_COMPARE
    dec = DecomposerExpert("d", ctx)
    no_exp = dec.decompose("预测", {"window": [0]})
    branches = {t.branch for t in no_exp}
    assert "pred-blind" in branches and "pred-analytic" in branches
    assert "pred-explicit" not in branches          # 未给显式参数则不拆 explicit
    with_exp = dec.decompose("预测", {"window": [0], "explicit": [0, 0, 0, 0]})
    assert "pred-explicit" in {t.branch for t in with_exp}


# --------------------------------------------------------------------------
# 协调器：预测客观选优 + 集成
# --------------------------------------------------------------------------
def test_coordinator_predict_objective_winner(model, batch):
    window, truth = batch

    async def go():
        c = CoordinatorAgent(model=model, concurrency=4)
        rep = await c.run("预测下一阶段轨迹",
                          {"window": window, "truth": truth,
                           "expert_modes": ["blind", "analytic"]})
        await c.aclose()
        return rep
    rep = asyncio.run(go())
    assert rep.kind == "predict"
    scores = sorted((p["score"] for p in rep.products), reverse=True)
    assert rep.winner is not None
    # 选优必须选中客观分最高者（反例：选中位即错误）
    assert rep.winner["score"] == scores[0]
    assert rep.decision["method"] == "best_score"
    assert np.asarray(rep.ensemble).shape[-2:] == (4, 6)


def test_action_compare_ranking(model, batch):
    window, _ = batch

    async def go():
        c = CoordinatorAgent(model=model)
        rep = await c.run("比较候选动作", {
            "window": window, "target": [2.0, 0, 0],
            "actions": [{"dv": [0, 0, 0]}, {"dv": [1.5, 0, 0]},
                        {"dv": [-1.0, 0, 0]}]})
        await c.aclose()
        return rep
    rep = asyncio.run(go())
    ranking = rep.products[0]
    assert ranking["rationale"] == "action ranking by goal+risk"


# --------------------------------------------------------------------------
# 代码补丁：仅胜者分支合并，败者不污染主线
# --------------------------------------------------------------------------
def test_code_patch_only_winner_merged(model):
    def harness(out):
        val = {"fast": 0.02, "slow": 0.5, "buggy": 0.9}[out["candidate"]]
        return {"score": 1 / (1 + val), "metrics": {"loss": val},
                "evidence": "verified"}

    async def go():
        c = CoordinatorAgent(model=model)
        rep = await c.run("选择最优实现并合并", {
            "target_file": "core.py", "base_files": {"core.py": "slow"},
            "candidates": {"fast": None, "slow": None, "buggy": None},
            "harness": harness})
        await c.aclose()
        return rep, c
    rep, c = asyncio.run(go())
    assert rep.winner["branch"] == "patch-fast"
    assert rep.main_files["core.py"] == "fast"     # 仅胜者合并
    # 败者分支留在各自私有草稿，主线不含 buggy/slow
    assert rep.main_files["core.py"] not in ("buggy", "slow")


# --------------------------------------------------------------------------
# 异步 DAG：依赖顺序与并发完成
# --------------------------------------------------------------------------
def test_async_dag_dependency_and_completion(model, batch):
    window, truth = batch

    async def go():
        c = CoordinatorAgent(model=model, concurrency=8)
        a = Task(goal="预测A", kind=TaskKind.PREDICT,
                 payload={"window": window, "truth": truth,
                          "expert_modes": ["blind"]}, branch="job-a")
        b = Task(goal="预测B", kind=TaskKind.PREDICT,
                 payload={"window": window, "truth": truth,
                          "expert_modes": ["blind"]},
                 deps=[a.id], branch="job-b")
        cc_ = Task(goal="预测C", kind=TaskKind.PREDICT,
                   payload={"window": window, "truth": truth,
                            "expert_modes": ["analytic"]},
                   deps=[a.id], branch="job-c")
        prods = await c._execute_dag([b, a, cc_])
        await c.aclose()
        return prods, a, b, cc_, c
    prods, a, b, cc_, c = asyncio.run(go())
    assert {a.id, b.id, cc_.id} <= set(prods.keys())
    assert all(t.status == TaskStatus.DONE for t in (a, b, cc_))
    # 子任务事件必须挂在父任务之后（血缘）
    child_evt = next(e for e in c.mem.events()
                     if e.payload.get("task") == b.id and e.kind == "work_product")
    assert a.id in child_evt.parent_ids


# --------------------------------------------------------------------------
# 跨协调器共享上下文（LocalTransport 真实同步）
# --------------------------------------------------------------------------
def test_local_transport_shares_context(model, batch):
    window, truth = batch

    async def go():
        t = LocalTransport()
        c1 = CoordinatorAgent(project="P", model=model, transport=t)
        await c1.run("预测", {"window": window, "truth": truth,
                              "expert_modes": ["blind", "analytic"]})
        c2 = CoordinatorAgent(project="P", model=model, transport=t)
        c2._sync_pull()
        kinds = {e.kind for e in c2.mem.events()}
        await c1.aclose(); await c2.aclose()
        return kinds
    kinds = asyncio.run(go())
    assert {"goal", "work_product", "decision", "goal_complete"} <= kinds


# --------------------------------------------------------------------------
# AL 门禁
# --------------------------------------------------------------------------
def test_al_gate():
    with pytest.raises(GateError):
        CloudTransport().pull("P")          # 未配置云端 → AL4 门禁
    g = CapabilityGate()
    with pytest.raises(GateError):
        g.require(AL.AL4)                   # 无 LLM key/云
    g2 = CapabilityGate(llm_keys_configured=True,
                        cloud_context_configured=True)
    g2.require(AL.AL4)                      # 配齐后 AL4 放行
    with pytest.raises(GateError):
        g2.require(AL.AL5)                  # 仍无 RSI 闭环 → AL5 拦截
