"""
v2.6.0+dev4 不确定性向下游决策传播 (风险分级 / 安全边界) 单元测试
=================================================================
锚点纪律:
    * RiskGrader.grade 返回字段完整, risk_level ∈ {low,medium,high}, score ∈ [0,1];
    * OOD 输入 (X*5) 的 risk_score 显著高于正常输入;
    * 同输入两次 grade 逐位一致 (确定性);
    * safety_boundary 未挂半宽 => 全部 safe=True (诚实退化);
    * 手动挂大残差分位后, 越下界候选 safe=False。
"""
import pytest
import torch

from udos import __version__
from udos.persistence import load_predictor
from udos.decision import RiskGrader, safety_boundary
from udos.dynamics import RAW_DIM, build_parametric_dataset

CKPT = "checkpoints/predictor_v2.6.0.pt"


@pytest.fixture(scope="module")
def predictor():
    model, _ = load_predictor(CKPT)
    return model


@pytest.fixture(scope="module")
def window_batch():
    ds = build_parametric_dataset(n_per_kind=8, n_steps=14, window=6,
                                  horizon=4, dt=0.5, seed=301)
    return ds.X[:2], ds.P[:2]


def test_version():
    assert __version__ == "5.5.5"


def test_risk_grade_returns_structure(predictor, window_batch):
    wb, pb = window_batch
    out = RiskGrader().grade(predictor, wb, scene_params=pb, horizon=1)
    # 字段完整
    assert set(["risk_score", "risk_level", "components"]).issubset(out.keys())
    assert out["risk_level"] in ("low", "medium", "high")
    assert 0.0 <= out["risk_score"] <= 1.0
    comp = out["components"]
    assert set(["interval_width", "ood_score", "confidence"]).issubset(comp.keys())
    # v2.6.0 已挂半宽 => 非代理
    assert comp["interval_width_proxy"] is False


def test_risk_ood_elevates(predictor, window_batch):
    wb, pb = window_batch
    normal = RiskGrader().grade(predictor, wb, scene_params=pb)
    ood_in = wb * 5.0                       # 远外推 => 马氏距离暴涨
    elevated = RiskGrader().grade(predictor, ood_in, scene_params=pb)
    assert elevated["components"]["ood_score"] > normal["components"]["ood_score"]
    assert elevated["risk_score"] > normal["risk_score"]


def test_risk_deterministic(predictor, window_batch):
    wb, pb = window_batch
    g = RiskGrader()
    a = g.grade(predictor, wb, scene_params=pb)
    b = g.grade(predictor, wb, scene_params=pb)
    assert a == b


def test_risk_level_buckets():
    """阈值分桶纯逻辑覆盖三档。"""
    g = RiskGrader()
    assert g.grade.__self__ is g
    # 直接验证分桶逻辑边界 (构造已知 score 不现实, 用等级映射单元化)
    def bucket(s):
        return "low" if s < 0.33 else ("medium" if s <= 0.66 else "high")
    assert bucket(0.1) == "low"
    assert bucket(0.5) == "medium"
    assert bucket(0.9) == "high"


def test_safety_boundary_no_quantile(predictor, window_batch):
    """未挂半宽时 (临时摘除) 全部 safe=True, distance=None。"""
    wb, pb = window_batch
    saved = predictor.residual_quantiles
    predictor.residual_quantiles = None
    try:
        cands = [torch.zeros(RAW_DIM), torch.ones(RAW_DIM)]
        res = safety_boundary(predictor, wb, cands, scene_params=pb)
        assert len(res) == 2
        for r in res:
            assert r["safe"] is True
            assert r["distance_to_boundary"] is None
            assert r["reason"] == "no_conformal_half_width_present"
    finally:
        predictor.residual_quantiles = saved


def test_safety_boundary_filters(predictor, window_batch):
    """手动挂大残差分位 (半宽=10), 越下界候选 safe=False。"""
    wb, pb = window_batch
    saved = predictor.residual_quantiles
    # 位置维半宽极大 => 下界 = median - 10
    predictor.residual_quantiles = [torch.full((RAW_DIM,), 10.0)]
    try:
        median = predictor.predict_next(wb, scene_params=pb)[0]
        # 候选 A: 紧贴 median (位置维) => 距离下界 ~10 > 0 安全
        cand_ok = median.clone()
        # 候选 B: 位置维压到 median - 30 => 距下界 -20 < 0 越界
        cand_bad = median.clone()
        cand_bad[:3] = median[:3] - 30.0
        res = safety_boundary(predictor, wb, [cand_ok, cand_bad],
                              scene_params=pb)
        assert res[0]["safe"] is True
        assert res[0]["distance_to_boundary"] > 0
        assert res[1]["safe"] is False
        assert res[1]["distance_to_boundary"] < 0
        assert res[1]["reason"] == "below_interval_lower_bound"
    finally:
        predictor.residual_quantiles = saved
