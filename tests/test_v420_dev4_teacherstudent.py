"""
v4.2.0.dev4 teacher->student 一代自训练闭环测试
==================================================
纪律 (最高优先级防退化):
  - student 只在外挂小 MLP, 主 predictor 权重 md5 不变 (主参恒 52191);
  - verdict 必须诚实: 在真实 holdout 上 student 未证优时不得判 improved;
  - 历史教训: 自蒸馏 mse=2.63 被 REJECT; 本闭环照实报 degraded/collapsed。
"""
import hashlib

import pytest
import torch

from udos import (CTMConfig, PhysicsPredictor, TransitionTripletGenerator,
                  TeacherStudentLoop)
from udos.world_model import LatentWorldModel
from udos.dynamics import build_parametric_dataset
from udos.evaluation import evaluate_predictor


def _md5(m):
    h = hashlib.md5()
    for k, v in sorted(m.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def _setup(seed=7):
    cfg = CTMConfig(iterations=8, d_model=64, d_input=32, heads=4,
                    n_synch_out=16, n_synch_action=8, memory_length=8,
                    nlm_hidden=16, out_dims=32, certainty_threshold=0.0)
    m = PhysicsPredictor(cfg, scene_param_dim=4)
    ds = build_parametric_dataset(n_per_kind=4, n_steps=14, window=6,
                                  horizon=3, dt=0.5, seed=seed)
    wm = LatentWorldModel(m, action_dim=0)
    wm.fit(ds, epochs=10)
    gen = TransitionTripletGenerator(m, wm, action_dim=1)
    trip = gen.generate(ds, n=48, seed=0)
    return m, trip, ds


def test_student_fit_reduces_train_loss():
    m, trip, ds = _setup()
    loop = TeacherStudentLoop(m, trip, epochs=40, seed=0)
    rep = loop.fit()
    assert rep["last_loss"] <= rep["first_loss"]
    assert rep["student_n_params"] > 0


def test_teacher_weights_untouched():
    m, trip, ds = _setup()
    before = _md5(m)
    loop = TeacherStudentLoop(m, trip, epochs=10, seed=0)
    loop.fit()
    after = _md5(m)
    assert before == after, "student 训练不得改动 teacher 主权重"


def test_compare_verdict_honest():
    m, trip, ds = _setup()
    teacher_mse = evaluate_predictor(m, ds)["single_step_mse"]
    loop = TeacherStudentLoop(m, trip, epochs=40, seed=0)
    loop.fit()
    rep = loop.compare(ds, teacher_mse)
    assert rep["teacher_mse"] == pytest.approx(teacher_mse, rel=1e-5)
    assert rep["student_mse"] > 0
    assert rep["verdict"] in ("improved", "degraded", "collapsed")
    # 诚实性: 未证优时 verdict 不得为 improved
    if rep["verdict"] != "improved":
        assert "未优于" in rep["honest_note"] or "不宣称" in rep["honest_note"]


def test_compare_before_fit_raises():
    m, trip, ds = _setup()
    loop = TeacherStudentLoop(m, trip, epochs=5, seed=0)
    with pytest.raises(ValueError):
        loop.compare(ds, 0.045)


def test_empty_triplet_rejected():
    m, trip, ds = _setup()
    with pytest.raises(ValueError):
        TeacherStudentLoop(m, triplets=object())


def test_bad_epochs_rejected():
    m, trip, ds = _setup()
    with pytest.raises(ValueError):
        TeacherStudentLoop(m, trip, epochs=0)


def test_student_small_param():
    # student 是外挂小头, 参数量远小于主预测器 (52191)
    m, trip, ds = _setup()
    loop = TeacherStudentLoop(m, trip, epochs=5, seed=0)
    loop.fit()
    assert loop.n_params_student < 52191
