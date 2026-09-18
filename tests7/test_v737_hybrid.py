"""v7.3.7 具身混合控制（语义层 × 动作先验）契约与互补性测试。"""
import torch

from udos7.contracts import STATE_DIM, EvidenceGrade
from udos7.embodied import (EnvTask, MotionPrior, PointMassEnv,
                            SemanticCritic, run_episode, run_suite)
from udos7.embodied.env import standard_suite
from udos7.embodied.hybrid import SEG_LEN
from udos7.observability import Tracer


def _gate():
    return next(t for t in standard_suite() if t.name == "contact_gate")


def _ordered():
    return next(t for t in standard_suite() if t.name == "ordered_sort")


def test_env_state_contract_and_ordered_waypoints():
    t = _ordered()
    env = PointMassEnv(t)
    s = env.reset()
    assert s.shape == (STATE_DIM,) and torch.isfinite(s).all()
    # 直接把末端放到“第二个”目标上：严格有序，不应计入进度
    env.pos = torch.tensor(t.waypoints[1], dtype=torch.float32)
    env._update_waypoints()
    assert env.active == 0
    # 放到第一个目标：推进一格
    env.pos = torch.tensor(t.waypoints[0], dtype=torch.float32)
    env._update_waypoints()
    assert env.active == 1


def test_contact_gate_collision_terminal():
    env = PointMassEnv(_gate())
    env.reset()
    for _ in range(_gate().max_steps):
        # 直奔原点障碍
        a = 2.4 * (torch.tensor([0.0, 0.0, 0.0]) - env.pos) - 1.1 * env.vel
        _, info = env.step(a)
        if env.done:
            break
    assert env.collisions >= 1 and env.success is False


def test_prior_proposes_K_segments():
    env = PointMassEnv(_gate()); env.reset()
    prior = MotionPrior(K=32, seed=7)
    cands = prior.propose(env, torch.tensor([2.0, 0.0, 0.0]))
    assert len(cands) == 32
    assert all(len(c.accs) == SEG_LEN and c.accs[0].shape == (3,) for c in cands)


def test_critic_picks_collision_free_candidate():
    env = PointMassEnv(_gate()); env.reset()
    critic = SemanticCritic()
    tgt = critic._active_target(env)
    cands = MotionPrior(K=32, seed=7).propose(env, tgt)
    cands.append(__import__("udos7.embodied.hybrid", fromlist=["Candidate"]).Candidate(
        "override", critic.override_program(env, tgt)))
    chosen = critic.decide(env, cands)
    assert chosen.rollout["collide"] is False
    assert chosen.kind == "prior"   # 接触任务应由动作先验绕行，而非直线接管


def test_episode_metrics_and_intervention_bounds():
    t = _ordered()
    d = run_episode(t, "direct", seed=1000)
    m = run_episode(t, "motion", seed=1000)
    h = run_episode(t, "hybrid", seed=1000)
    assert d["intervention_rate"] == 1.0 and m["intervention_rate"] == 0.0
    assert 0.0 <= h["intervention_rate"] <= 1.0
    for r in (d, m, h):
        assert 0.0 <= r["score"] <= 100.0 and r["decisions"] >= 1


def test_determinism_same_seed():
    t = _gate()
    a = run_episode(t, "hybrid", seed=123)
    b = run_episode(t, "hybrid", seed=123)
    assert a["score"] == b["score"] and a["steps"] == b["steps"]


def test_complementarity_on_suite():
    """固定 seed 实测：语义强于顺序任务、动作先验强于接触任务，hybrid 全绿。"""
    rep = run_suite(episodes_per_task=10, K=32)
    assert rep["evidence_grade"] == EvidenceGrade.CPU_PROTO.value
    by = rep["by_task"]
    assert by["contact_gate"]["direct"]["success_rate"] == 0.0
    assert by["contact_gate"]["motion"]["success_rate"] == 1.0
    assert by["contact_gate"]["hybrid"]["success_rate"] == 1.0
    assert by["ordered_sort"]["motion"]["success_rate"] == 0.0
    assert by["ordered_sort"]["hybrid"]["success_rate"] == 1.0
    agg = rep["aggregate"]
    assert agg["hybrid"]["success_rate"] >= agg["direct"]["success_rate"]
    assert agg["hybrid"]["success_rate"] >= agg["motion"]["success_rate"]
    assert agg["hybrid"]["mean_collisions"] == 0.0
    # 外部参照必须标 unverified，且不得混入本仓指标
    ext = rep["external_reference"]
    assert ext["grade"] == EvidenceGrade.UNVERIFIED.value


def test_hybrid_costs_less_token_proxy_than_direct():
    rep = run_suite(episodes_per_task=10, K=32)
    direct_in = rep["aggregate"]["direct"]["tokens_in_proxy_total"]
    hybrid_in = rep["aggregate"]["hybrid"]["tokens_in_proxy_total"]
    assert hybrid_in < direct_in


def test_tracing_spans_and_tokens():
    tr = Tracer()
    run_episode(_ordered(), "hybrid", seed=1000, tracer=tr)
    names = {s.name for s in tr.spans}
    assert "embodied.episode" in names
    assert "motion.propose" in names and "semantic.critic" in names
    root = next(s for s in tr.spans if s.name == "embodied.episode")
    from udos7.observability.semconv import GEN_AI_USAGE_INPUT_TOKENS
    assert root.attrs.get(GEN_AI_USAGE_INPUT_TOKENS, 0) > 0
