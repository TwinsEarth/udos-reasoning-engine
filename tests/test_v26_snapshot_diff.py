"""
v2.6.0+dev5 推理快照差分 / checkpoint 数值对比单元测试
========================================================
锚点纪律:
    * 同件 export 两次 => diff identical=True 且各段为空;
    * 受控扰动残差分位后 => calibration_diff 非空、identical=False;
    * compare_checkpoints(v2.5.2, v2.6.0) 返回字段完整、pred_diff 可计算;
    * 同件对比 pred_diff_max≈0。

诚实记录: v2.6.0 相对 v2.5.2 仅新增 hybrid 能力, 而 export_snapshot **不含权重也
不含 hybrid 状态**, 故二者快照在结构化层面完全一致 (identical=True) —— 这正是快照
"无状态后处理"语义的预期, 不是 bug。权重差异由 compare_checkpoints 在张量层面给出。
"""
import pytest
import torch

from udos import __version__
from udos.persistence import (load_predictor, export_snapshot,
                              diff_snapshots, compare_checkpoints)

CKPT_A = "checkpoints/predictor_v2.5.2.pt"
CKPT_B = "checkpoints/predictor_v2.6.0.pt"


def test_version():
    assert __version__ == "5.5.5"


def test_diff_same_snapshot():
    """同件 export 两次 => identical=True 且各 diff 段为空。"""
    model, _ = load_predictor(CKPT_B)
    s1 = export_snapshot(model)
    s2 = export_snapshot(model)
    d = diff_snapshots(s1, s2)
    assert d["identical"] is True
    assert d["config_diff"] == {}
    assert d["calibration_diff"] == {}
    assert d["ood_diff"] == {}


def test_diff_different_snapshot_controlled():
    """受控放大残差分位 1.5x => calibration_diff 非空, identical=False。"""
    model, _ = load_predictor(CKPT_B)
    pristine = export_snapshot(model)
    # 扰动后处理状态 (不改权重, 仅改快照里的校准半宽)
    model.residual_quantiles = [q * 1.5 for q in model.residual_quantiles]
    perturbed = export_snapshot(model)
    d = diff_snapshots(pristine, perturbed)
    assert d["identical"] is False
    # config 段不变 (架构/权重后处理未动)
    assert d["config_diff"] == {}
    # calibration 段非空, 且落到 residual_quantiles 逐维差
    assert d["calibration_diff"], "校准段应检测到差异"
    rq = d["calibration_diff"].get("residual_quantiles", {})
    assert rq, f"应定位到 residual_quantiles 差异: {d['calibration_diff']}"


def test_diff_v252_vs_v260_snapshot_equal():
    """
    v2.5.2 vs v2.6.0 快照: ctm_config 相同 (config_diff 空), 且因快照不含权重/hybrid,
    二者在后处理层面 identical=True (诚实记录, 非 bug)。
    """
    pa, _ = load_predictor(CKPT_A)
    pb, _ = load_predictor(CKPT_B)
    da = diff_snapshots(export_snapshot(pa), export_snapshot(pb))
    assert da["config_diff"] == {}   # 架构配置一致
    assert da["identical"] is True   # 无状态后处理一致


def test_compare_checkpoints_fields():
    """compare v2.5.2 vs v2.6.0: 返回字段完整, pred_diff 可计算。"""
    r = compare_checkpoints(CKPT_A, CKPT_B, n_per_kind=16, seed=999)
    for k in ("a_metrics", "b_metrics", "pred_diff_max", "pred_diff_mean",
              "mse_diff", "ece_diff", "coverage_diff",
              "params_a", "params_b"):
        assert k in r, f"缺字段 {k}"
    assert r["params_a"] == r["params_b"] == 52191
    # 两 checkpoint 权重同源 (v2.6.0=v2.5.2+hybrid能力), 单步预测逐位一致
    assert r["pred_diff_max"] == pytest.approx(0.0, abs=1e-9)
    assert r["mse_diff"] == pytest.approx(0.0, abs=1e-9)
    assert r["coverage_diff"] is not None


def test_compare_same_checkpoint():
    """同件对比: pred_diff_max≈0, ece_diff/coverage_diff≈0。"""
    r = compare_checkpoints(CKPT_B, CKPT_B, n_per_kind=16, seed=999)
    assert r["pred_diff_max"] == pytest.approx(0.0, abs=1e-12)
    assert r["pred_diff_mean"] == pytest.approx(0.0, abs=1e-12)
    assert r["ece_diff"] == pytest.approx(0.0, abs=1e-9)
