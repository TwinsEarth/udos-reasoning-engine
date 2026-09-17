"""v2.3.x 回归: 保序置信校准、多时域损失权重、split-conformal 区间、服务/持久化。

原则同 v22: 单测锁机制正确性与向后兼容锚点, 不锁随种子波动的增益阈值
(增益由 scripts/ablation_horizon_weight.py 与 VERIFICATION 的多种子证据裁决)。
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from udos.ctm_engine import CTMConfig  # noqa: E402
from udos.dynamics import build_parametric_dataset  # noqa: E402
from udos.training import PhysicsPredictor, CTMTrainer, TrainConfig  # noqa: E402
from udos.evaluation import evaluate_predictor  # noqa: E402
from udos.calibration import (  # noqa: E402
    ConfidenceCalibrator, fit_predictor_calibration, reliability, spearman,
    _pava_nondec,
)


def _cfg():
    return CTMConfig(iterations=4, d_model=32, d_input=20, heads=2,
                     n_synch_out=10, n_synch_action=8, memory_length=6,
                     nlm_hidden=12, out_dims=20, certainty_threshold=0.0)


def _pred():
    return PhysicsPredictor(_cfg(), scene_param_dim=4)


def _ds(horizon=4, n=8, seed=1):
    return build_parametric_dataset(n_per_kind=n, n_steps=6 + horizon + 4,
                                    window=6, horizon=horizon, dt=0.5,
                                    seed=seed)


# ---------- F1 保序校准基础机制 ----------
def test_pava_merges_violation_and_is_nondec():
    # 前两点 (0.9,0.1) 违规被合并为一块, 后两点保持 => 共 3 块且非降
    levels = _pava_nondec([0, 1, 2, 3], [0.9, 0.1, 0.95, 0.99])
    ys = [y for _, y in levels]
    assert len(levels) == 3
    assert all(ys[i] <= ys[i + 1] + 1e-12 for i in range(len(ys) - 1))


def test_spearman_monotone_and_constant():
    x = torch.tensor([1.0, 2, 3, 4])
    assert abs(spearman(x, x) - 1.0) < 1e-6
    assert abs(spearman(x, -x) + 1.0) < 1e-6
    assert spearman(x, torch.ones(4)) == 0.0  # 常量列相关无定义 -> 0


def test_calibrator_transform_monotone_clip_and_roundtrip():
    torch.manual_seed(0)
    # 置信越高、误差越低 (良好排序): 校准映射应单调非降
    conf = torch.linspace(0.05, 0.95, 60)
    err = (1.0 - conf) * 0.2 + 0.01
    cal = ConfidenceCalibrator().fit(conf, err)
    assert cal.fitted and cal.num_segments >= 2
    probe = torch.linspace(-0.2, 1.2, 40)
    out = cal.transform(probe)
    so = out[torch.argsort(probe)]
    assert bool((so.diff() >= -1e-6).all())          # 单调非降
    # 区间外 clip 为同一端点值 (容差, 不做浮点严格相等)
    left, right = out[probe <= 0], out[probe >= 1]
    assert float(left.max() - left.min()) < 1e-6
    assert float(right.max() - right.min()) < 1e-6
    # 序列化往返逐位一致
    cal2 = ConfidenceCalibrator().load_state_dict(cal.state_dict())
    assert torch.allclose(cal2.transform(probe), out)


def test_calibrator_edge_cases():
    # 少于 2 样本报错
    try:
        ConfidenceCalibrator().fit(torch.tensor([0.5]), torch.tensor([0.1]))
        assert False
    except ValueError:
        pass
    # 全零误差不崩 (scale 退化) 且仍可用
    cal = ConfidenceCalibrator().fit(torch.tensor([0.2, 0.5, 0.8]),
                                     torch.zeros(3))
    assert cal.fitted and cal.transform(torch.tensor([0.5])).numel() == 1
    # 置信/误差形状不一致报错
    try:
        ConfidenceCalibrator().fit(torch.tensor([0.2, 0.5]),
                                   torch.tensor([0.1, 0.2, 0.3])); assert False
    except ValueError:
        pass
    # 未拟合 transform 报错
    try:
        ConfidenceCalibrator().transform(torch.tensor([0.5])); assert False
    except RuntimeError:
        pass


def test_calibrator_degenerate_when_ranking_inverted():
    # 置信越高误差越大 (2.2.1 的坏排序): 保序被压平, 必须如实标注退化
    conf = torch.linspace(0.1, 0.9, 40)
    err = conf * 0.3 + 0.01
    cal = ConfidenceCalibrator().fit(conf, err)
    assert cal.is_degenerate and cal.num_segments == 1


def test_reliability_perfect_calibration_zero_ece():
    # 置信 == 经验精度时 ECE≈0, 桶结构完整
    conf = torch.linspace(0.1, 0.9, 50)
    err = -torch.log(conf.clamp_min(1e-6))  # 使 exp(-err)=conf
    rep = reliability(conf, err, scale=1.0, n_bins=5)
    assert rep["ece"] < 1e-6 and len(rep["bins"]) == 5
    try:
        reliability(torch.zeros(0), torch.zeros(0), 1.0); assert False
    except ValueError:
        pass


# ---------- F2 多时域损失权重 (含兼容锚点) ----------
def test_step_weight_schemes_and_legacy_anchor():
    H = 4
    legacy = torch.linspace(1.0, 0.5, H); legacy = legacy / legacy.sum()
    # 默认 front 逐位复现 2.2.1 (向后兼容锚点); uniform/back 为显式 opt-in
    assert TrainConfig().step_weight_scheme == "front"
    front = CTMTrainer(_pred(), TrainConfig())._step_weights(H, torch.device("cpu"))
    assert torch.allclose(front, legacy, atol=1e-7)
    uni = CTMTrainer(_pred(), TrainConfig(
        step_weight_scheme="uniform"))._step_weights(H, torch.device("cpu"))
    assert torch.allclose(uni, torch.full((H,), 0.25), atol=1e-7)
    back = CTMTrainer(_pred(), TrainConfig(
        step_weight_scheme="back"))._step_weights(H, torch.device("cpu"))
    assert torch.allclose(back, front.flip(0), atol=1e-7)  # back 为 front 反转
    # H=1 三方案都归一为 [1]
    for sch in ("front", "uniform", "back"):
        w = CTMTrainer(_pred(), TrainConfig(
            step_weight_scheme=sch))._step_weights(1, torch.device("cpu"))
        assert abs(float(w) - 1.0) < 1e-7
    try:
        CTMTrainer(_pred(), TrainConfig(
            step_weight_scheme="nope"))._step_weights(H, torch.device("cpu"))
        assert False
    except ValueError:
        pass


# ---------- F3 预测区间 ----------
def test_predict_interval_requires_then_orders_bounds():
    model = _pred(); model.eval()
    ds = _ds()
    try:
        model.predict_interval(ds.X[:4], 4, scene_params=ds.P[:4]); assert False
    except RuntimeError:  # 未挂载分位 -> 明确报错, 不静默退化
        pass
    cal, _, rq = fit_predictor_calibration(_pred(), ds)
    model.attach_calibration(cal, rq)
    iv = model.predict_interval(ds.X[:4], 4, scene_params=ds.P[:4])
    B, H, R = iv["median"].shape
    assert iv["lower"].shape == iv["upper"].shape == (4, H, R)
    # 绝对残差对称区间: 下界恒<=中值<=上界 (半宽非负)
    assert bool((iv["lower"] <= iv["median"] + 1e-6).all())
    assert bool((iv["upper"] >= iv["median"] - 1e-6).all())
    assert all(w >= 0 for w in iv["half_width_by_step"])
    assert len(iv["half_width_by_step"]) == H
    # 逐维半宽: 每步形状为 [RAW]
    assert rq[0].shape == (R,)


def test_conformal_quantiles_are_conservative_and_bounded():
    from udos.calibration import _conservative_upper_quantile
    v = torch.arange(100, dtype=torch.float32)  # 0..99, 线性0.9=89.1
    q = _conservative_upper_quantile(v, 0.9)
    assert q.item() >= torch.quantile(v, 0.9).item()  # 向上取整, 不偏窄
    assert q.item() == 90
    # 极端概率/极小样本不越界
    assert _conservative_upper_quantile(torch.tensor([1.0, 2.0]), 1.0).item() == 2.0
    # 绝对残差半宽必非负
    assert _conservative_upper_quantile(torch.tensor([0.0, -3.0, 2.0]).abs(), 0.9).item() >= 0


def test_calibrator_is_external_not_state_dict_param():
    model = _pred()
    cal, _, rq = fit_predictor_calibration(_pred(), _ds())
    model.attach_calibration(cal, rq)
    keys = " ".join(model.state_dict().keys())
    assert "calibrator" not in keys and "residual_quantiles" not in keys


# ---------- 评估段: 旧契约不破坏, 新段按需出现 ----------
def test_evaluate_sections_gated_by_calibration():
    ds1, ds2 = _ds(seed=1), _ds(seed=2)
    model = _pred()
    raw_rep = evaluate_predictor(model, ds2)
    assert "calibration" not in raw_rep and "interval" not in raw_rep  # 未校准不新增
    cal, _, rq = fit_predictor_calibration(_pred(), ds1)
    model.attach_calibration(cal, rq)
    rep = evaluate_predictor(model, ds2)
    c = rep["calibration"]
    for k in ("raw", "calibrated", "ece_reduction_x", "ranking_informative"):
        assert k in c
    assert isinstance(c["ranking_informative"], bool)
    iv = rep["interval"]
    assert len(iv["coverage_by_step"]) == ds2.Y.size(1)
    assert 0.0 <= iv["coverage_overall"] <= 1.0


# ---------- 持久化: 校准往返 + 旧档兼容 ----------
def test_persistence_calibration_roundtrip_and_legacy(tmp_path):
    ds = _ds()
    model = _pred()
    cal, _, rq = fit_predictor_calibration(_pred(), ds)
    model.attach_calibration(cal, rq)
    p = tmp_path / "p23.pt"
    from udos import save_predictor, load_predictor
    save_predictor(model, p)
    loaded, meta = load_predictor(p)
    assert loaded.is_calibrated and len(loaded.residual_quantiles) == ds.Y.size(1)
    probe = torch.linspace(0.1, 0.9, 12)
    assert torch.allclose(loaded.calibrator.transform(probe),
                          cal.transform(probe))
    assert "calibration" in meta
    # 向后兼容: v2.2.1 旧档无 calibration 键 -> 未校准、可正常载入
    legacy = ROOT / "checkpoints" / "predictor_v2.2.1.pt"
    if legacy.exists():
        old, om = load_predictor(legacy)
        assert not old.is_calibrated and om["udos_version"] == "2.2.1"


# ---------- F4 服务: /calibrate /checkpoints /load ----------
def test_service_calibrate_flow_and_409(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from udos.server import UDOSService, ServiceNotReady
    s = UDOSService(preset="small")
    try:
        s.calibrate({}); assert False
    except ServiceNotReady:
        pass
    s.train({"epochs": 2, "n_per_kind": 6, "horizon": 4})
    out = s.calibrate({"n_per_kind": 8, "horizon": 4})
    assert out["status"] == "calibrated"
    assert "fitted_on_calibration_set" in out
    assert out["independent_test"]["calibration"] is not None
    assert s.engine.predictor.is_calibrated
    # 非法入参 -> ValueError(400 语义)
    for bad in ({"horizon": 99}, {"n_per_kind": 0}):
        try:
            s.calibrate(bad); assert False, bad
        except ValueError:
            pass


def test_service_checkpoints_list_and_load(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from udos.server import UDOSService
    s = UDOSService(preset="small",
                    checkpoints_dir=str(tmp_path / "checkpoints"))
    empty = s.list_checkpoints()
    assert empty["count"] == 0 and empty["checkpoints"] == []
    s.train({"epochs": 1, "n_per_kind": 6, "horizon": 4})
    s.calibrate({"n_per_kind": 6, "horizon": 4})
    sv = s.save({"name": "mine"})
    listed = s.list_checkpoints()
    assert listed["count"] == 1 and listed["checkpoints"][0]["name"] == "mine.pt"
    loaded = s.load({"name": "mine.pt"})
    assert loaded["status"] == "loaded" and loaded["calibrated"]
    for bad in ("../../x.pt", "missing.pt", "   "):
        try:
            s.load({"name": bad}); assert False, bad
        except ValueError:
            pass


def test_service_train_rejects_bad_step_scheme_and_echoes():
    from udos.server import UDOSService
    s = UDOSService(preset="small")
    try:
        s.train({"epochs": 1, "step_weight_scheme": "weird"}); assert False
    except ValueError:
        pass
    out = s.train({"epochs": 1, "n_per_kind": 6, "horizon": 4,
                   "step_weight_scheme": "back"})
    assert out["step_weight_scheme"] == "back"
