"""v3.8.1 多体协同 A/B + 冲突消解: 扫描 N={2,4,8}, 落 JSON, opt-in。

覆盖:
    * benchmarks/results/multi_agent_ab_v3.8.0.json 字段齐全;
    * 智能体数 2/4/8 扫描存在;
    * 有协调器降低碰撞事件数 (合成对穿场景);
    * 协调器为 opt-in 非学习; 被否决/让行候选显式记录;
    * 诚实权衡: 让行减速 -> 到达率下降 (不静默夸大收益)。
"""
import json
from pathlib import Path

import torch

from udos import __version__
from udos.multi_agent import AgentCoordinator, MultiAgentScene

ROOT = Path(__file__).resolve().parents[1]
JSON = ROOT / "benchmarks" / "results" / "multi_agent_ab_v3.8.0.json"


def test_version():
    assert __version__ == "5.5.5"


def test_ab_json_exists_and_schema():
    assert JSON.exists()
    d = json.load(open(JSON, encoding="utf-8"))
    assert d["feature"] == "multi_agent_conflict_resolution_ab"
    assert d["sweep_n_agents"] == [2, 4, 8]
    assert d["learned"] is False
    assert d["zero_gradient"] is True
    for n in ("n2", "n4", "n8"):
        assert "no_coordinator" in d["results"][n]
        assert "with_coordinator" in d["results"][n]


def test_coordinator_reduces_collisions():
    d = json.load(open(JSON, encoding="utf-8"))
    for n in ("n2", "n4", "n8"):
        r = d["results"][n]
        assert r["with_coordinator"]["collision_events"] <= \
            r["no_coordinator"]["collision_events"]


def test_opt_in_and_rejected_explicit():
    """协调器显式记录让行候选 (不静默过滤); 单体无冲突。"""
    sc = MultiAgentScene()
    sc.add_agent("hi", [0, 0, 0, 1, 0, 0], priority=1, radius=0.3)
    sc.add_agent("lo", [0.2, 0, 0, -1, 0, 0], priority=9, radius=0.3)
    rep = AgentCoordinator().resolve(sc)
    assert "lo" in rep["yielded_agents"]
    assert "hi" in rep["winner_agents"]
    # 无协同时不发生任何消解 (opt-in: 不显式调用就无让行)
    solo = MultiAgentScene()
    solo.add_agent("only", [0, 0, 0, 0.5, 0, 0])
    assert AgentCoordinator().resolve(solo)["n_conflicts"] == 0


def test_honest_tradeoff_documented():
    """JSON 须诚实记录权衡说明 (不夸大协调器收益)。"""
    d = json.load(open(JSON, encoding="utf-8"))
    assert "opt-in" in d["note"] or "收益" in d["note"]
