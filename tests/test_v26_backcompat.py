"""v2.6.0+dev7 后向兼容: 覆盖全部 7 个 checkpoint。

每个 checkpoint:
    * load_predictor 成功、版本元数据正确;
    * predict_next 正常输出形状 [B, RAW_DIM=6]、有限;
    * is_calibrated 属性存在 (旧件为 False);
    * hybrid 属性存在 (旧件为 None);
    * decision/persistence 新能力对旧件不报错。
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import torch  # noqa: E402

from udos import __version__  # noqa: E402
from udos.persistence import load_predictor, export_snapshot  # noqa: E402
from udos.decision import RiskGrader, safety_boundary  # noqa: E402


ALL_CKPTS = [
    ("predictor_v2.1.0.pt", "2.1.0"),
    ("predictor_v2.2.1.pt", "2.2.1"),
    ("predictor_v2.3.1.pt", "2.3.1"),
    ("predictor_v2.4.0.pt", "2.4.0"),
    ("predictor_v2.5.0.pt", "2.5.0"),
    ("predictor_v2.5.2.pt", "2.5.2"),
    ("predictor_v2.6.0.pt", "2.6.0"),
    ("predictor_v2.6.2.pt", "2.6.2"),
]


def test_version():
    assert __version__ == "5.5.5"


def test_all_eight_checkpoints_exist():
    for name, _ in ALL_CKPTS:
        assert (ROOT / "checkpoints" / name).exists(), f"缺 {name}"


def test_all_load_predict_and_attributes():
    """8 件全部: 加载成功 / predict 形状 [B,6] / is_calibrated 与 hybrid 属性存在。"""
    torch.manual_seed(0)
    for name, expected_ver in ALL_CKPTS:
        path = ROOT / "checkpoints" / name
        predictor, meta = load_predictor(str(path))
        assert meta["udos_version"] == expected_ver, \
            f"{name}: {meta['udos_version']} != {expected_ver}"
        # predict_next 正常
        x = torch.randn(2, 6, predictor.raw_dim)
        if predictor.scene_encoder is not None:
            p = torch.randn(2, predictor.scene_param_dim)
            out = predictor.predict_next(x, scene_params=p)
        else:
            out = predictor.predict_next(x)
        assert out.shape == (2, 6), f"{name}: 输出形状 {out.shape}"
        assert torch.isfinite(out).all(), f"{name}: 输出非有限"
        # v2.6 后处理属性在旧件上必须存在且诚实退化
        assert hasattr(predictor, "is_calibrated")
        assert hasattr(predictor, "hybrid")      # 旧件为 None
        if name < "predictor_v2.6.0.pt":
            assert predictor.hybrid is None, f"{name}: 不应有 hybrid"


def test_risk_grade_old_checkpoints_does_not_crash():
    """RiskGrader 对全部 8 件 (含无校准/无OOD的旧件) 诚实退化、不报错。"""
    torch.manual_seed(1)
    for name, _ in ALL_CKPTS:
        predictor, _ = load_predictor(str(ROOT / "checkpoints" / name))
        x = torch.randn(1, 6, predictor.raw_dim)
        p = (torch.randn(1, predictor.scene_param_dim)
             if predictor.scene_encoder is not None else None)
        r = RiskGrader().grade(predictor, x, scene_params=p, horizon=1)
        assert r["risk_level"] in ("low", "medium", "high")
        assert 0.0 <= r["risk_score"] <= 1.0


def test_safety_boundary_old_checkpoints_degrades():
    """无半宽的旧 checkpoint (v2.1.0/v2.2.1): 全部 safe=True 诚实退化;
    已挂半宽的件 (v2.3.1+) 正常计算边界, 只要求结构合法、不报错。"""
    torch.manual_seed(2)
    for name, _ in ALL_CKPTS:
        predictor, _ = load_predictor(str(ROOT / "checkpoints" / name))
        x = torch.randn(1, 6, predictor.raw_dim)
        cands = [torch.randn(predictor.raw_dim)]
        res = safety_boundary(predictor, x, cands)
        assert "safe" in res[0] and "reason" in res[0]
        if getattr(predictor, "residual_quantiles", None) is None:
            # 无区间信息 => 诚实退化全 safe
            assert res[0]["safe"] is True, f"{name} 应诚实退化全 safe"
            assert res[0]["distance_to_boundary"] is None
        else:
            # 有半宽 => 距离为数值
            assert isinstance(res[0]["distance_to_boundary"], (int, float))


def test_export_snapshot_all_checkpoints():
    """8 件均可 export_snapshot (v2.6.0+dev5 无状态快照)。"""
    for name, _ in ALL_CKPTS:
        predictor, _ = load_predictor(str(ROOT / "checkpoints" / name))
        snap = export_snapshot(predictor)
        assert snap["kind"] == "InferenceSnapshot"
        assert snap["raw_dim"] == predictor.raw_dim
