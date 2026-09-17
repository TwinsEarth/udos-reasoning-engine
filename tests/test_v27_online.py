"""
v2.7.0.dev1 在线增量适配与漂移触发再校准闭环 (udos.online.OnlineAdapter) 测试
================================================================================
锚点纪律:
    * 无漂移 (分布内观测) => 不触发、不改状态、日志为空;
    * 漂移 (X*5) => 触发再校准, 日志追加一条;
    * 默认 enable_finetune=False => 触发后 weights_modified=False, 模型权重逐位不变;
    * opt-in enable_finetune=True => weights_modified=True, 权重确实改变;
    * adaptation_log 字段完整; 与 ood.StreamingDriftDetector 接口一致。
"""
import copy

import pytest
import torch

from udos import __version__
from udos.online import OnlineAdapter
from udos.ood import StreamingDriftDetector
from udos.persistence import load_predictor
from udos.dynamics import build_parametric_dataset

CKPT = "checkpoints/predictor_v2.7.0.pt"


@pytest.fixture(scope="module")
def predictor():
    model, _ = load_predictor(CKPT)
    return model


@pytest.fixture(scope="module")
def ref_and_drift():
    ref = build_parametric_dataset(n_per_kind=12, n_steps=14, window=6,
                                   horizon=4, dt=0.5, seed=501)
    cal = build_parametric_dataset(n_per_kind=8, n_steps=14, window=6,
                                   horizon=4, dt=0.5, seed=502)
    return ref, cal


def test_version():
    assert __version__ == "5.5.5"


def _snapshot_weights(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _same_weights(a, b):
    return all(torch.equal(a[k], b[k]) for k in a)


def test_no_drift_no_trigger(predictor, ref_and_drift):
    ref, cal = ref_and_drift
    ad = OnlineAdapter(ref.X, drift_window=32)
    # 分布内观测 => 窗口均值贴近参考 => 不触发
    ad.observe(ref.X[:20])
    before = _snapshot_weights(predictor)
    out = ad.check_and_adapt(predictor, cal)
    assert out["adapted"] is False
    assert out["reason"] == "no_drift"
    assert ad.adaptation_log == []
    # 不触发 => 权重逐位不变
    assert _same_weights(before, _snapshot_weights(predictor))


def test_drift_triggers_recalibration(predictor, ref_and_drift):
    ref, cal = ref_and_drift
    ad = OnlineAdapter(ref.X, drift_window=32)
    # 远外推观测 => 窗口均值漂移远超阈值
    ad.observe(ref.X[:24] + 15.0)
    assert ad.is_drifted() is True
    out = ad.check_and_adapt(predictor, cal)
    assert out["adapted"] is True
    assert out["weights_modified"] is False
    assert len(ad.adaptation_log) == 1


def test_default_does_not_modify_weights(predictor, ref_and_drift):
    ref, cal = ref_and_drift
    ad = OnlineAdapter(ref.X, drift_window=32)
    ad.observe(ref.X[:24] + 15.0)
    before = _snapshot_weights(predictor)
    out = ad.check_and_adapt(predictor, cal)
    assert out["adapted"] is True
    assert out["weights_modified"] is False
    # 默认只重跑 PAVA (校准器/半宽), 模型权重逐位不变
    assert _same_weights(before, _snapshot_weights(predictor))


def test_opt_in_finetune_changes_weights(predictor, ref_and_drift):
    ref, cal = ref_and_drift
    ad = OnlineAdapter(ref.X, drift_window=32, enable_finetune=True,
                       finetune_epochs=2, finetune_lr=1e-2)
    ad.observe(ref.X[:24] + 15.0)
    before = _snapshot_weights(predictor)
    out = ad.check_and_adapt(predictor, cal)
    assert out["adapted"] is True
    assert out["weights_modified"] is True
    # 微调后权重确实改变
    assert not _same_weights(before, _snapshot_weights(predictor))


def test_adaptation_log_complete(predictor, ref_and_drift):
    ref, cal = ref_and_drift
    ad = OnlineAdapter(ref.X, drift_window=32)
    ad.observe(ref.X[:24] + 15.0)
    ad.check_and_adapt(predictor, cal)
    assert len(ad.adaptation_log) == 1
    e = ad.adaptation_log[0]
    for k in ("timestamp", "trigger_reason", "drift_score",
              "before_ece", "after_ece", "weights_modified"):
        assert k in e, f"日志缺字段 {k}"
    assert e["weights_modified"] is False
    assert e["drift_score"] > e["threshold"]


def test_reset_clears_state(predictor, ref_and_drift):
    ref, cal = ref_and_drift
    ad = OnlineAdapter(ref.X, drift_window=32)
    ad.observe(ref.X[:24] + 15.0)
    ad.check_and_adapt(predictor, cal)
    assert len(ad.adaptation_log) == 1
    ad.reset()
    assert ad.adaptation_log == []
    assert ad.detector.n_window == 0


def test_compatible_with_streaming_detector(ref_and_drift):
    ref, _ = ref_and_drift
    ad = OnlineAdapter(ref.X, drift_window=16)
    assert isinstance(ad.detector, StreamingDriftDetector)
    assert ad.detector.fitted is True
    assert ad.detector.threshold_ == ad.detector.threshold_  # 非 nan 已拟合
