"""
v2.6.0+dev7 跨特性集成加固测试
==================================
不重训、纯前向、确定性。把 v2.6 线各新模块串起来跑全链路, 确保它们互不破坏:
    hybrid <-> counterfactual; adaptive <-> risk; identify -> counterfactual;
    load -> predict -> risk -> counterfactual -> identify 全管线。
"""
import pytest
import torch

from udos import __version__
from udos.persistence import load_predictor
from udos.hybrid import HybridPhysicsCorrector
from udos.counterfactual import CounterfactualEngine
from udos.identification import SceneParameterIdentifier
from udos.adaptive import adaptive_rollout
from udos.decision import RiskGrader
from udos.dynamics import build_parametric_dataset

CKPT = "checkpoints/predictor_v2.6.0.pt"


@pytest.fixture(scope="module")
def predictor():
    model, _ = load_predictor(CKPT)
    return model


@pytest.fixture(scope="module")
def batch():
    ds = build_parametric_dataset(n_per_kind=8, n_steps=14, window=6,
                                  horizon=4, dt=0.5, seed=501)
    mask = ds.kind_mask("spring")
    idx = torch.nonzero(mask, as_tuple=False).flatten()[:1]
    return ds.X[idx], ds.P[idx]


def test_version():
    assert __version__ == "5.5.5"


def test_hybrid_plus_counterfactual(predictor, batch):
    """attach hybrid 后 CounterfactualEngine 仍正常; 零干预=基线逐位。"""
    w, p = batch
    saved = predictor.hybrid
    predictor.attach_hybrid(HybridPhysicsCorrector(raw_dim=predictor.raw_dim))
    try:
        out = CounterfactualEngine(predictor).counterfactual(
            w, horizon=3, scene_params=p, intervention=None)
        assert torch.allclose(out["counterfactual"], out["baseline"], atol=1e-7)
        assert out["ate_mean"] == pytest.approx(0.0, abs=1e-12)
    finally:
        predictor.hybrid = saved


def test_adaptive_plus_risk(predictor, batch):
    """adaptive_rollout 推进轨迹后, 用扩展窗口喂 RiskGrader, 全链路无异常。"""
    w, p = batch
    res = adaptive_rollout(predictor, w, max_horizon=3, scene_params=p)
    assert res["horizon_used"] >= 1
    # 把推进轨迹接回窗口末, 保留最近 window 帧 => 推进一步后重评风险
    extended = torch.cat([w, res["trajectory"]], dim=1)[:, -w.size(1):, :]
    risk = RiskGrader().grade(predictor, extended, scene_params=p, horizon=1)
    assert risk["risk_level"] in ("low", "medium", "high")
    assert 0.0 <= risk["risk_score"] <= 1.0


def test_identification_plus_counterfactual(predictor, batch):
    """用 identify 反演的参数作 scene_params, 再做反事实干预。"""
    w, p = batch
    ident = SceneParameterIdentifier(predictor, grid_size=4)
    idp = ident.identify(w, horizon=2)["identified_params"].reshape(1, -1)
    out = CounterfactualEngine(predictor).counterfactual(
        w, horizon=2, scene_params=idp,
        intervention={"scene_params": {"2": 1.2}})
    assert out["ate_mean"] >= 0.0
    assert out["counterfactual"].shape == out["baseline"].shape


def test_full_pipeline(predictor, batch):
    """load -> predict -> risk -> counterfactual -> identify 全链路无异常。"""
    w, p = batch
    pred = predictor.predict_next(w, scene_params=p)
    assert pred.shape[-1] == predictor.raw_dim
    risk = RiskGrader().grade(predictor, w, scene_params=p, horizon=1)
    assert risk["risk_level"] in ("low", "medium", "high")
    cf = CounterfactualEngine(predictor).counterfactual(
        w, horizon=2, scene_params=p)
    assert cf["baseline"].shape[0] == w.size(0)
    ident = SceneParameterIdentifier(predictor, grid_size=3).identify(w, horizon=2)
    assert len(ident["identified_params"]) == 4
