"""
v4.1.1 停止/自我纠正判据测试 (置信门控 + 收敛停止 + OOD 主动验证)
====================================================================
纪律: 外挂零梯度、决策优先级、非法参数 ValueError、确定性。
"""
import hashlib

import pytest
import torch

from udos import (__version__, CTMConfig, PhysicsPredictor,
                  GoalDecomposer, StopCorrectController)
from udos.ood import DistributionDriftDetector
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


def _plan_input(seed=3):
    m = _predictor()
    de = GoalDecomposer(m, horizon=2)
    ds = build_parametric_dataset(n_per_kind=2, n_steps=14, window=6,
                                  horizon=2, dt=0.5, seed=seed)
    return m, de, ds


def test_decision_accept_when_no_signals():
    m, de, ds = _plan_input()
    scc = StopCorrectController(de)
    out = scc.decide(ds.X[0:1], ds.Y[0, -1, :], n_subgoals=3,
                     scene_params=ds.P[0:1])
    assert out["decision"] in ("accept", "stop")
    assert out["confidence"] == 1.0


def test_low_confidence_triggers_self_correct():
    m, de, ds = _plan_input()
    scc = StopCorrectController(de, confidence_fn=lambda w: 0.1,
                                 conf_threshold=0.5)
    out = scc.decide(ds.X[0:1], ds.Y[0, -1, :], n_subgoals=3,
                     scene_params=ds.P[0:1])
    assert out["decision"] == "self_correct"
    assert out["low_confidence"] is True


def test_ood_triggers_verify():
    m, de, ds = _plan_input()
    # 拟合一个 OOD 检测器在训练分布上, 再注入极端窗使其 OOD
    det = DistributionDriftDetector()
    train_feats = ds.X.reshape(len(ds), -1)
    det.fit(train_feats)
    scc = StopCorrectController(de, ood_detector=det, confidence_fn=lambda w: 0.9)
    extreme = torch.rand(1, 6, 6) * 100.0   # 远离训练分布
    out = scc.decide(extreme, ds.Y[0, -1, :], n_subgoals=3)
    assert out["ood"] is True
    assert out["decision"] == "verify"


def test_convergence_stop_when_plateau():
    m, de, ds = _plan_input()
    # 收敛 tol 很大 => 一步即平台
    scc = StopCorrectController(de, confidence_fn=lambda w: 0.9,
                                 convergence_tol=1e9)
    out = scc.decide(ds.X[0:1], ds.Y[0, -1, :], n_subgoals=3,
                     scene_params=ds.P[0:1])
    assert out["converged"] is True
    assert out["decision"] == "stop"


def test_decision_priority_lowconf_beats_ood():
    m, de, ds = _plan_input()
    det = DistributionDriftDetector()
    det.fit(ds.X.reshape(len(ds), -1))
    scc = StopCorrectController(de, confidence_fn=lambda w: 0.05,
                                 ood_detector=det)
    extreme = torch.rand(1, 6, 6) * 100.0
    out = scc.decide(extreme, ds.Y[0, -1, :], n_subgoals=2)
    # 低置信优先级最高
    assert out["decision"] == "self_correct"


def test_ctor_bad_args():
    m, de, ds = _plan_input()
    with pytest.raises(ValueError):
        StopCorrectController(de, conf_threshold=1.5)
    with pytest.raises(ValueError):
        StopCorrectController(de, convergence_tol=-1)
    with pytest.raises(ValueError):
        StopCorrectController(de, max_iter=0)


def test_zero_gradient_main_untouched():
    m, de, ds = _plan_input()
    before = _md5(m)
    scc = StopCorrectController(de, confidence_fn=lambda w: 0.9)
    scc.decide(ds.X[0:1], ds.Y[0, -1, :], n_subgoals=2,
               scene_params=ds.P[0:1])
    assert before == _md5(m)
