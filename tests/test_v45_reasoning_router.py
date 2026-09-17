"""v4.5.0.dev1 reasoning_router 测试。"""
import pytest

from udos.reasoning_router import (
    ReasoningRouter, DifficultySignals, quick_signals,
)
from udos.ctm_engine import CTMConfig
from udos.gpm_engine import GPMConfig, TinyBaseModel
from udos.reasoning import UDOSReasoningEngine


def _engine():
    base = TinyBaseModel(hidden=32, n_layers=2)
    gpm = GPMConfig(feature_dim=32, latent_size=32, n_latents=8, lora_rank=4,
                    layer_indices=(0, 1), num_pre_head_layers=1, heads=2)
    ctm = CTMConfig(iterations=8, d_model=64, d_input=32, heads=2,
                    n_synch_out=16, n_synch_action=16, memory_length=8,
                    nlm_hidden=16, out_dims=32, certainty_threshold=0.0,
                    n_random_pairing_self=2)
    return UDOSReasoningEngine(ctm, gpm, base_model=base)


def test_easy_routes_to_none():
    r = ReasoningRouter()
    s = DifficultySignals(ctm_convergence=0.99)  # 高收敛 = 易
    d = r.route(s)
    assert d.effort == "none"
    assert d.externalize is False
    assert 0.0 <= d.difficulty <= 1.0


def test_hard_routes_to_max():
    r = ReasoningRouter()
    # 低收敛 + 高分歧 + OOD + 复杂任务
    s = DifficultySignals(ctm_convergence=0.1, ensemble_disagreement=0.9,
                          ood_score=0.9, task_complexity=0.9, needs_deeper=0.9)
    d = r.route(s)
    assert d.effort == "max"
    assert d.externalize is True


def test_mid_routes_to_low_or_high():
    r = ReasoningRouter()
    # 仅 CTM 信号: difficulty = 0.4*(1-conv); conv=0.2 -> 0.32 -> low
    s_lo = DifficultySignals(ctm_convergence=0.2)
    d_lo = r.route(s_lo)
    assert d_lo.effort == "low"
    assert d_lo.difficulty >= ReasoningRouter.THR_NONE - 1e-6


def test_rationale_nonempty_and_explains():
    r = ReasoningRouter()
    s = DifficultySignals(ctm_convergence=0.3, ensemble_disagreement=0.7,
                          ood_score=0.6)
    d = r.route(s)
    assert len(d.rationale) >= 2
    assert any("effort=" in line for line in d.rationale)
    assert "集成分歧" in "".join(d.rationale)


def test_difficulty_monotonic_with_uncertainty():
    r = ReasoningRouter()
    easy = r.synthesize(DifficultySignals(ctm_convergence=0.99))
    hard = r.synthesize(DifficultySignals(ctm_convergence=0.01))
    assert hard > easy


def test_clamp_out_of_range():
    r = ReasoningRouter()
    s = DifficultySignals(ctm_convergence=5.0, ood_score=-2.0)
    d = r.route(s)
    assert 0.0 <= d.difficulty <= 1.0


def test_quick_signals_reads_convergence(tiny_scene, torch_seed):
    eng = _engine()
    s = quick_signals(tiny_scene, eng)
    assert 0.0 <= s.ctm_convergence <= 1.0
    r = ReasoningRouter().route(s)
    assert r.effort in ("none", "low", "high", "max")
