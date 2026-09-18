"""v7.4.0 第一人称经验数据飞轮契约测试。"""
from udos7.contracts import EvidenceGrade
from udos7.egodata.episodes import collect_episode, generate_pool, TASK_NAMES
from udos7.egodata.coverage import CoverageMap, state_cells
from udos7.egodata.processing import (quality_control, experience_density,
                                      process_batch, signature)
from udos7.egodata.collection import (passive_select, active_select,
                                      run_strategy, PASSIVE_SKEW)
from udos7.egodata.benchmark import run_flywheel_benchmark, EXTERNAL_REFERENCE


def test_episode_deterministic():
    a = collect_episode("contact_gate", 2, 42, K=6)
    b = collect_episode("contact_gate", 2, 42, K=6)
    assert a.states == b.states and a.contacts == b.contacts
    assert len(a.hand_track) == len(a.states)


def test_qc_rejects_static():
    ep = collect_episode("reach_free", 0, 1, K=6, defect="static")
    qc = quality_control(ep, set())
    assert not qc["accepted"] and "static_or_empty" in qc["reasons"]


def test_qc_rejects_drift():
    ep = collect_episode("contact_gate", 0, 5, K=6, defect="drift")
    qc = quality_control(ep, set())
    assert "camera_drift" in qc["reasons"]
    clean = collect_episode("contact_gate", 0, 5, K=6)
    assert quality_control(clean, set())["accepted"]


def test_qc_rejects_duplicate_pair():
    a = collect_episode("ordered_sort", 3, 9, K=6)
    b = collect_episode("ordered_sort", 3, 9, K=6)
    seen = {signature(a)}
    qc_b = quality_control(b, seen)
    assert "duplicate" in qc_b["reasons"]


def test_state_vs_task_coverage():
    m = CoverageMap()
    e1 = collect_episode("reach_free", 0, 2, K=6)
    e2 = collect_episode("reach_free", 2, 2, K=6)
    m.add(e1); n2 = m.add(e2)
    assert m.task_coverage() == 1 / len(TASK_NAMES)
    assert n2 > 0                                  # 同任务不同变体仍补新状态
    assert len(state_cells(e1) | state_cells(e2)) > len(state_cells(e1))


def test_density_contact_rich_above_static():
    rich = collect_episode("ordered_sort", 1, 3, K=6)
    static = collect_episode("ordered_sort", 1, 3, K=6, defect="static")
    assert experience_density(rich)["density_score"] > \
        experience_density(static)["density_score"]


def test_active_selection_beats_passive_on_same_pool():
    pool = generate_pool(60, seed=7, K=6, task_skew=PASSIVE_SKEW)
    p = run_strategy(pool, 18, active=False)
    a = run_strategy(pool, 18, active=True)
    assert a["final_state_cells"] >= p["final_state_cells"]


def test_process_batch_yield_and_reject_counts():
    pool = generate_pool(40, seed=3, K=6, defect_rate=0.4)
    accepted, ledger, rejects = process_batch(list(pool))
    assert ledger.raw_episodes == 40
    assert 0.0 < ledger.yield_rate < 1.0
    assert sum(rejects.values()) == 40 - ledger.accepted_episodes


def test_benchmark_active_covers_more_and_graded():
    r1 = run_flywheel_benchmark(pool_size=50, budget=16, seed=11, K=6)
    r2 = run_flywheel_benchmark(pool_size=50, budget=16, seed=11, K=6)
    assert r1["evidence_grade"] == EvidenceGrade.CPU_PROTO.value
    assert r1["coverage_aware_collection"]["oracle_fraction"] > \
        r1["passive_collection"]["oracle_fraction"]
    assert r1["coverage_aware_collection"]["mean_density"] > \
        r1["passive_collection"]["mean_density"]
    assert r1["coverage_aware_collection"]["state_cells"] == \
        r2["coverage_aware_collection"]["state_cells"]
    assert "unverified" in str(EXTERNAL_REFERENCE).lower() or \
        "unverified" in str(r1["external_reference"]).lower()
    assert len(r1["passive_collection"]["coverage_curve"]) == 16


def test_sparsest_task_directs_collection():
    m = CoverageMap()
    m.add(collect_episode("reach_free", 0, 1, K=6))
    m.add(collect_episode("reach_free", 1, 1, K=6))
    assert m.sparsest_tasks()[0] != "reach_free"
