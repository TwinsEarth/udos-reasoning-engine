"""
v4.1.0 自规划自监督线起点测试
==============================
覆盖: LessonSpec / CurriculumGenerator / SolvabilityVerifier。
纪律: 外挂零梯度 (主 predictor md5 不变)、空/非法显式 ValueError、版本断言、确定性。
"""
import hashlib

import pytest
import torch

from udos import (__version__, CTMConfig, PhysicsPredictor,
                  LessonSpec, CurriculumGenerator, SolvabilityVerifier)
from udos.dynamics import build_parametric_dataset


def _md5(m):
    h = hashlib.md5()
    for k, v in sorted(m.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def _predictor():
    cfg = CTMConfig(iterations=8, d_model=64, d_input=32, heads=4,
                    n_synch_out=16, n_synch_action=8, memory_length=8,
                    nlm_hidden=16, out_dims=32, certainty_threshold=0.0)
    return PhysicsPredictor(cfg, scene_param_dim=4)


def test_version():
    assert __version__ == "5.5.5"


def test_lesson_spec_immutable():
    s = LessonSpec(stage=1, n_per_kind=8, horizon=3, spread=1.2, seed=7)
    d = s.describe()
    assert d["stage"] == 1 and d["spread"] == 1.2
    with pytest.raises(Exception):
        s.stage = 99  # frozen dataclass


def test_stage_profile_grows_with_stage():
    gen = CurriculumGenerator(base_seed=0)
    p0 = gen.stage_profile(0)
    p3 = gen.stage_profile(3)
    assert p3.n_per_kind > p0.n_per_kind
    assert p3.horizon >= p0.horizon
    assert p3.spread >= p0.spread
    # 子种子派生确定
    assert gen.stage_profile(0).seed == gen.base_seed


def test_stage_profile_rejects_bad():
    gen = CurriculumGenerator()
    with pytest.raises(ValueError):
        gen.stage_profile(-1)
    with pytest.raises(ValueError):
        gen.stage_profile(1.5)


def test_make_lesson_shape_and_kinds():
    gen = CurriculumGenerator(base_seed=11)
    lesson = gen.make_lesson(0)
    ds = lesson["dataset"]
    assert lesson["n_samples"] == len(ds)
    assert set(lesson["kinds"].keys()) == {"uniform", "accel", "spring", "collision"}
    assert ds.X.dim() == 3 and ds.Y.dim() == 3
    assert ds.X.size(-1) == 6


def test_generate_empty_rejected():
    gen = CurriculumGenerator()
    with pytest.raises(ValueError):
        gen.generate([])
    lessons = gen.generate([0, 1])
    assert len(lessons) == 2


def test_generator_deterministic():
    a = CurriculumGenerator(base_seed=123).make_lesson(2)
    b = CurriculumGenerator(base_seed=123).make_lesson(2)
    assert torch.allclose(a["dataset"].X, b["dataset"].X)
    assert torch.allclose(a["dataset"].P, b["dataset"].P)


def test_verify_solvable_task():
    m = _predictor()
    v = SolvabilityVerifier(m)
    ds = build_parametric_dataset(n_per_kind=4, n_steps=14, window=6,
                                  horizon=2, dt=0.5, seed=5)
    r = v.verify(ds.X[0:1], scene_params=ds.P[0:1], horizon=2)
    assert r["finite"] is True
    assert r["within_bounds"] in (True, False)
    assert "solvable" in r


def test_verify_rejects_bad_input():
    m = _predictor()
    v = SolvabilityVerifier(m)
    with pytest.raises(ValueError):
        v.verify(torch.zeros(2, 6, 6))          # 批维>1
    with pytest.raises(ValueError):
        v.verify(torch.zeros(1, 6, 6) * float("nan"))
    with pytest.raises(ValueError):
        v.verify(torch.zeros(1, 6, 6), horizon=0)


def test_verify_dataset_aggregate():
    m = _predictor()
    v = SolvabilityVerifier(m)
    ds = build_parametric_dataset(n_per_kind=4, n_steps=14, window=6,
                                  horizon=1, dt=0.5, seed=9)
    out = v.verify_dataset(ds, max_samples=12, horizon=1)
    assert out["n_checked"] == 12
    assert 0.0 <= out["solvable_ratio"] <= 1.0
    assert out["n_solvable"] == len(out["solvable_indices"])
    with pytest.raises(ValueError):
        v.verify_dataset(ds, max_samples=0)


def test_verifier_zero_gradient_main_untouched():
    m = _predictor()
    before = _md5(m)
    v = SolvabilityVerifier(m)
    ds = build_parametric_dataset(n_per_kind=4, n_steps=14, window=6,
                                  horizon=2, dt=0.5, seed=21)
    v.verify_dataset(ds, max_samples=8, horizon=2)
    after = _md5(m)
    assert before == after


def test_verifier_ctor_bad_bounds():
    m = _predictor()
    with pytest.raises(ValueError):
        SolvabilityVerifier(m, pos_bound=-1.0)
    with pytest.raises(ValueError):
        SolvabilityVerifier(m, divergence_growth=1.0)


def test_curriculum_ctor_bad_window():
    with pytest.raises(ValueError):
        CurriculumGenerator(window=0)


# ---------------- v4.1.0.dev1 课程难度递进 ---------------- #

def test_auto_progression_records_difficulty():
    m = _predictor()
    v = SolvabilityVerifier(m)
    gen = CurriculumGenerator(base_seed=5)
    table = gen.auto_progression(v, [0, 1, 2, 3], n_probe=8)
    assert len(table) == 4
    horizons = [t["horizon"] for t in table]
    assert horizons == sorted(horizons)          # 环境复杂度随 stage 不降
    spreads = [t["spread"] for t in table]
    assert spreads == sorted(spreads)
    for t in table:
        assert 0.0 <= t["solvable_ratio"] <= 1.0


def test_auto_progression_empty_rejected():
    m = _predictor()
    v = SolvabilityVerifier(m)
    with pytest.raises(ValueError):
        v  # noqa
        CurriculumGenerator().auto_progression(SolvabilityVerifier(m), [])


def test_auto_expand_horizon_finds_frontier():
    m = _predictor()
    v = SolvabilityVerifier(m)
    gen = CurriculumGenerator(base_seed=9)
    out = gen.auto_expand_horizon(v, start=2, step=1, max_h=5,
                                  solvable_floor=0.5, n_probe=8)
    assert out["frontier_horizon"] >= 2
    assert out["stopped_by"] in ("floor", "max_h")
    assert out["schedule"][0]["horizon"] == 2
    # 难度单调上升
    hs = [s["horizon"] for s in out["schedule"]]
    assert hs == sorted(hs)


def test_auto_expand_rejects_bad_args():
    m = _predictor()
    v = SolvabilityVerifier(m)
    gen = CurriculumGenerator()
    with pytest.raises(ValueError):
        gen.auto_expand_horizon(v, start=0)
    with pytest.raises(ValueError):
        gen.auto_expand_horizon(v, solvable_floor=0.0)
    with pytest.raises(ValueError):
        gen.auto_expand_horizon(v, start=5, max_h=2)
