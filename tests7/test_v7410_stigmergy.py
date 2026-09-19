"""v7.4.10 Stigmergy 环境媒介协作测试。"""
import random

import pytest

from udos7.topology.stigmergy import (Blackboard, run_stigmergy,
                                      contract_net_message_count)


def test_atomic_claim_no_duplicate_work():
    b = Blackboard()
    b.post("T1")
    assert b.claim("a", "T1") is True
    assert b.claim("b", "T1") is False
    assert b.items["T1"].claimed_by == "a"


def test_claim_after_completion_rejected():
    b = Blackboard()
    b.post("T1")
    b.claim("a", "T1")
    b.complete("T1")
    assert b.claim("b", "T1") is False


def test_all_tasks_eventually_done():
    keys = [f"T{i}" for i in range(20)]
    r = run_stigmergy(keys, ["a", "b", "c", "d"], seed=1)
    assert r["done"] == 20
    assert r["duplicate_claims"] == 0


def test_stigmergy_uses_fewer_messages_than_contract_net():
    keys = [f"T{i}" for i in range(30)]
    r = run_stigmergy(keys, ["a", "b", "c", "d", "e"], seed=2)
    cn = contract_net_message_count(30, n_bidders=5)
    # stigmergy 每任务 2 条痕迹；contract-net 每任务 8 条
    assert r["messages"] == 60
    assert r["messages"] < cn
    assert cn == 30 * 8


def test_load_balanced_across_agents():
    keys = [f"T{i}" for i in range(40)]
    r = run_stigmergy(keys, ["a", "b", "c", "d"], seed=3)
    loads = list(r["loads"].values())
    assert sum(loads) == 40
    # 均匀轮询认领，最大-最小 ≤ 1
    assert max(loads) - min(loads) <= 1


def test_pheromone_evaporation():
    b = Blackboard(evaporation=0.5)
    b.post("T1")
    b.deposit("T1", 3.0)
    before = b.items["T1"].pheromone
    b.evaporate()
    assert b.items["T1"].pheromone == before * 0.5


def test_pheromone_biases_selection():
    """高信息素任务更可能被优先选中（统计性，固定种子验证）。"""
    rng = random.Random(7)
    first_picks = []
    for _ in range(40):
        b = Blackboard()
        b.post("lo", value=1.0)
        b.post("hi", value=1.0)
        b.deposit("hi", 20.0)
        k = b.pick("a", rng)
        first_picks.append(k)
    assert first_picks.count("hi") > first_picks.count("lo")


def test_no_direct_messaging_required():
    """agent 之间零直接通信：黑板是唯一协调面。"""
    b = Blackboard()
    for k in ("T1", "T2"):
        b.post(k)
    assert b.claim("a", "T1")
    assert b.claim("b", "T2")
    b.complete("T1"); b.complete("T2")
    # 全部消息都落在环境（黑板），无 agent→agent 计数
    assert b.messages == 4
    assert all(it.done for it in b.items.values())
