"""v7.5.0 百万级 Agent 矩阵集成测试。"""
import pytest

from udos7.topology.matrix import run_mission, MatrixConfig, build_matrix
from udos7.topology.base import standard_workload


@pytest.fixture
def orders():
    return standard_workload(12, seed=5)


def test_clean_mission_full_acceptance(orders):
    r = run_mission(orders, MatrixConfig(), seed=1)
    assert r["accepted"] == 12
    assert r["retries"] == 0
    assert r["governance"]["ok"] is True
    assert r["trace_verified"] is True
    assert r["market"]["conserved"] is True
    assert r["isolated_agents"] == []


def test_faulty_workers_isolated_and_work_reassigned(orders):
    faults = {"spec-kinematics-0": "crash",
              "spec-spatial-1": "drop_context",
              "spec-dataflywheel-2": "byzantine"}
    r = run_mission(orders, MatrixConfig(), faulty_workers=faults, seed=1)
    # 熔断隔离故障节点，任务改派健康节点，全部收口
    assert r["accepted"] == 12
    assert "spec-kinematics-0" in r["isolated_agents"]
    assert "spec-spatial-1" in r["isolated_agents"]
    assert r["retries"] > 0
    assert r["governance"]["ok"] is True
    assert r["market"]["conserved"] is True


def test_byzantine_qa_minority_cannot_block_acceptance(orders):
    # QA 委员会默认 7 人含 2 个拜占庭（f=2），诚实多数仍可验收
    r = run_mission(orders, MatrixConfig(qa_f=2), seed=2)
    assert r["accepted"] == 12


def test_market_pays_only_accepted(orders):
    r = run_mission(orders, MatrixConfig(reward_per_order=10), seed=3)
    assert r["market"]["total_paid"] == 120.0
    assert r["market"]["holds"] == 0.0


def test_trace_chain_intact_after_rollbacks(orders):
    faults = {"spec-kinematics-0": "crash", "spec-spatial-0": "crash"}
    r = run_mission(orders, MatrixConfig(), faulty_workers=faults, seed=4)
    assert r["trace_verified"] is True
    # 失败执行的半截 trace 被回滚，最终每单 2 条（specialist+qa）
    assert r["trace_len"] == 2 * 12


def test_scale_benchmark_included_to_million():
    r = run_mission(standard_workload(6, seed=0), MatrixConfig(), seed=0)
    scales = {row["target_agents"]: row for row in r["scale"]}
    assert set(scales) == {10, 100, 1000, 10000, 100000, 1000000}
    million = scales[1000000]
    assert million["layered_max_fanin"] == 9
    assert million["layered_rounds"] == 14
    assert million["star_center_fanin"] > 1_000_000
    assert million["grade"] == "analytical"


def test_kill_switch_on_retry_budget(orders):
    # 全部某领域 worker 故障且预算为 0：无法改派，全局熔断如实报告降级
    faults = {f"spec-kinematics-{k}": "crash" for k in range(3)}
    cfg = MatrixConfig(global_retry_budget=0, max_rounds=10)
    r = run_mission(orders, cfg, faulty_workers=faults, seed=5)
    assert r["kill_switch"] is True
    assert r["accepted"] < 12
    # 不冒充成功：未收口工单进入治理报告
    assert r["governance"]["ok"] is False


def test_evidence_grade_labeled_cpu_proto(orders):
    r = run_mission(orders, MatrixConfig(), seed=0)
    assert r["evidence_grade"] == "cpu-proto"


def test_committee_sizing_validated(orders):
    with pytest.raises(ValueError):
        run_mission(orders, MatrixConfig(qa_voters=4, qa_f=2), seed=0)
