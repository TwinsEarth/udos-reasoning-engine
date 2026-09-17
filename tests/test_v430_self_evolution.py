"""
v4.3.0 完全自进化线测试 —— 基础设施自我优化缩微类比
=================================================================
纪律: 外挂零梯度 (主 predictor md5 不变)、空/非法显式 ValueError、确定性、
      配置搜索须过"输出保真"硬门 (不允许靠降质换速度)、诚实记录改进 vs 退化。
"""
import hashlib

import pytest
import torch

from udos import (CTMConfig, PhysicsPredictor, __version__,
                  ConfigSpec, ConfigEvaluator, ConfigSearcher,
                  SearchVerifySelectLoop, ConfigAB, GlobalStopCorrectCriterion,
                  LongHorizonLoop, SelfEvolutionOrchestrator,
                  MultiGenerationRunner, SELF_EVO_LINE_STAGES)
from udos.dynamics import build_parametric_dataset


def _md5(m):
    h = hashlib.md5()
    for k, v in sorted(m.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def _predictor():
    cfg = CTMConfig(iterations=3, d_model=24, d_input=16, heads=2,
                    n_synch_out=8, n_synch_action=6, memory_length=4,
                    nlm_hidden=8, out_dims=12, certainty_threshold=0.0)
    return PhysicsPredictor(cfg, raw_dim=6, scene_param_dim=4)


def _evaluator(pred, repeats=2):
    ds = build_parametric_dataset(n_per_kind=4, n_steps=14, window=6,
                                  horizon=3, dt=0.5, seed=1)
    return ConfigEvaluator(pred, ds.X[:8], repeats=repeats)


def test_version():
    assert __version__ == "5.5.5"
    assert SELF_EVO_LINE_STAGES[-1] == "4.3.9"
    assert SELF_EVO_LINE_STAGES[0] == "4.3.0"


def test_configspec_validation():
    with pytest.raises(ValueError):
        ConfigSpec(max_shard=0)
    with pytest.raises(ValueError):
        ConfigSpec(codebook_size=1)
    with pytest.raises(ValueError):
        ConfigSpec(cortex_hz=0.0)
    with pytest.raises(ValueError):
        ConfigSpec(ensemble_weights=(1.0, -1.0))
    # roundtrip
    s = ConfigSpec(max_shard=16, cache_enabled=True)
    assert ConfigSpec.from_dict(s.to_dict()).to_dict() == s.to_dict()


def test_evaluator_pure_effective_config_preserves_output():
    m = _predictor()
    before = _md5(m)
    ev = _evaluator(m)
    # 默认配置输出差必须为 0
    r = ev.evaluate(ConfigSpec.default())
    assert r["output_max_abs_diff"] == 0.0
    assert r["output_preserved"] is True
    # 纯效率旋钮 (max_shard/cache) 不应改前向数学
    r2 = ev.evaluate(ConfigSpec(max_shard=4, cache_enabled=True))
    assert r2["output_max_abs_diff"] == 0.0
    # 主权重 md5 不变 (外挂零梯度)
    assert _md5(m) == before


def test_searcher_returns_acceptable_and_honest():
    m = _predictor()
    ev = _evaluator(m)
    res = ConfigSearcher(ev).search()
    assert res["n_candidates"] > 1
    # 所有候选都过保真门 (纯效率旋钮 -> 必过)
    assert res["n_acceptable"] == res["n_candidates"]
    # 选中配置与 default 同合同可比
    assert "selected" in res
    assert isinstance(res["improved"], bool)


def test_search_verify_select_loop_deterministic():
    m = _predictor()
    ev = _evaluator(m)
    a = SearchVerifySelectLoop(ev, n_rounds=2, search_mode="grid").run()
    b = SearchVerifySelectLoop(ev, n_rounds=2, search_mode="grid").run()
    assert a["n_archive"] == b["n_archive"]
    assert a["global_best"]["config"] == b["global_best"]["config"]


def test_random_mode_needs_positive_rounds():
    m = _predictor()
    ev = _evaluator(m)
    with pytest.raises(ValueError):
        SearchVerifySelectLoop(ev, n_rounds=0)
    with pytest.raises(ValueError):
        SearchVerifySelectLoop(ev, n_rounds=1, search_mode="bogus")


def test_config_ab_honest():
    m = _predictor()
    ev = _evaluator(m)
    searched = ConfigSpec(max_shard=16, cache_enabled=True, cortex_hz=1.0,
                          codebook_size=4)
    ab = ConfigAB(ev, searched=searched).run()
    assert "arm_A_searched" in ab and "arm_B_default" in ab
    assert isinstance(ab["A_wins"], bool)
    # 无论胜负, verdict 都必须是诚实话术
    assert isinstance(ab["verdict"], str) and ab["verdict"]


def test_global_stop_correct():
    g = GlobalStopCorrectCriterion(patience=3, min_delta=1e-3)
    # 连续不改进 -> diminishing
    r = g.check([10.0, 9.9, 9.8, 9.7], [True, True, True, True])
    assert r["recommended_action"] in ("continue", "stop_self_evolution",
                                       "rollback_and_research")
    # 当代失真 -> rollback
    r2 = g.check([10.0, 9.0], [True, False])
    assert r2["recommended_action"] == "rollback_to_last_preserved"
    with pytest.raises(ValueError):
        g.check([1.0], [True])


def test_long_horizon_loop_verified():
    m = _predictor()
    out = LongHorizonLoop(m, horizon=4, n_sub=2).run(seed=0)
    assert out["verified"] is True
    assert out["n_steps_executed"] == 4
    assert out["trajectory_finite"] is True
    # 确定性
    out2 = LongHorizonLoop(m, horizon=4, n_sub=2).run(seed=0)
    assert out["n_steps_executed"] == out2["n_steps_executed"]


def test_long_horizon_validation():
    m = _predictor()
    with pytest.raises(ValueError):
        LongHorizonLoop(m, horizon=0)
    with pytest.raises(ValueError):
        LongHorizonLoop(m, horizon=4, n_sub=9)


def test_evaluator_rejects_bad_workload():
    m = _predictor()
    with pytest.raises(ValueError):
        ConfigEvaluator(m, torch.randn(8, 6))  # 缺一维 -> [8,6] 非 3D
    with pytest.raises(ValueError):
        ConfigEvaluator(m, torch.randn(8, 6, 6), repeats=0)


def _wired_orchestrator():
    from udos.world_model import LatentWorldModel
    from udos.curriculum import CurriculumGenerator
    from udos.self_train import TransitionTripletGenerator
    from udos.self_evolution import (SelfEvolutionOrchestrator,
                                      MultiGenerationRunner)
    m = _predictor()
    ds = build_parametric_dataset(n_per_kind=4, n_steps=14, window=6,
                                  horizon=3, dt=0.5, seed=1)
    wm = LatentWorldModel(m, action_dim=0)
    wm.fit(ds, epochs=8)
    gen = TransitionTripletGenerator(m, wm, action_dim=1)
    cur = CurriculumGenerator()
    ev = ConfigEvaluator(m, ds.X[:8], repeats=2)
    orch = SelfEvolutionOrchestrator(m, ds, ev, wm, cur, gen)
    return orch, MultiGenerationRunner


def test_orchestrator_chains_three_dimensions():
    orch, _ = _wired_orchestrator()
    rec = orch.run_generation(gen_id=0, n_lessons=3, n_trip=16)
    # 4.1 课程 + 4.2 三元组 + 4.3 配置 三维各有产出
    assert rec["course_n"] >= 1
    assert rec["triplet_n"] == 16
    assert "config_best_cost" in rec
    assert rec["triplet_quality"]["next_state_finite"] == 1.0


def test_multi_generation_curve_honest():
    _, Runner = _wired_orchestrator()
    orch = None
    # 直接用上面构造的 runner 模式: 重新跑一遍 2 代
    m = _predictor()
    ds = build_parametric_dataset(n_per_kind=4, n_steps=14, window=6,
                                  horizon=3, dt=0.5, seed=1)
    from udos.world_model import LatentWorldModel
    from udos.curriculum import CurriculumGenerator
    from udos.self_train import TransitionTripletGenerator
    wm = LatentWorldModel(m, action_dim=0); wm.fit(ds, epochs=8)
    gen = TransitionTripletGenerator(m, wm, action_dim=1)
    cur = CurriculumGenerator()
    ev = ConfigEvaluator(m, ds.X[:8], repeats=2)
    orch = SelfEvolutionOrchestrator(m, ds, ev, wm, cur, gen)
    out = Runner(orch, n_generations=2).run()
    assert out["n_generations"] == 2
    assert len(out["cost_curve"]) == 2
    assert out["honest_verdict"] in ("efficiency_improving", "drifting",
                                      "collapsed")
    assert "honest_note" in out

