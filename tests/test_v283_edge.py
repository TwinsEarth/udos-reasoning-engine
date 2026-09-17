"""
v2.8.3 Patch 精修: 边界条件单测
==================================
覆盖五类边界 (均为显式守卫/退化路径, 不静默 inf 传播):
    1. loop 空 window / 非有限 window -> ValueError;
    2. multitask 头 latent 维度不匹配 (头期望 > encode 产出) -> RuntimeError;
    3. future_state horizon=0 -> PhysicalLoopRunner.__init__ 显式 ValueError;
    4. feedback 极端偏差率 (dev_threshold 极小) -> correction.triggered, 但默认不改权重;
    5. 服务端点未训练态 -> ServiceNotReady (HTTP 409 语义)。
"""
import pytest
import torch

from udos import __version__
from udos.server import UDOSService
from udos.persistence import load_predictor
from udos.physical_loop import PhysicalLoopRunner
from udos.multitask import MultiTaskHead, SpatialCoordHead
from udos.dynamics import build_parametric_dataset

CKPT = "checkpoints/predictor_v2.8.0.pt"


@pytest.fixture(scope="module")
def predictor():
    m, _ = load_predictor(CKPT)
    return m


@pytest.fixture(scope="module")
def window_batch():
    ds = build_parametric_dataset(n_per_kind=8, n_steps=14, window=6,
                                  horizon=4, dt=0.5, seed=9090)
    return ds.X[:1], ds.P[:1]


def test_version():
    assert __version__ == "5.5.5"


# 1. 空 window / 非有限 window
def test_loop_empty_window_rejected(predictor, window_batch):
    loop = PhysicalLoopRunner(predictor, horizon=2)
    # B=0
    with pytest.raises(ValueError):
        loop.run(torch.zeros(0, 6, predictor.raw_dim))
    # W=0
    with pytest.raises(ValueError):
        loop.run(torch.zeros(1, 0, predictor.raw_dim))


def test_loop_nonfinite_window_rejected(predictor, window_batch):
    loop = PhysicalLoopRunner(predictor, horizon=2)
    bad = window_batch[0].clone()
    bad[0, -1, 0] = float("nan")
    with pytest.raises(ValueError):
        loop.run(bad, scene_params=window_batch[1])


# 2. multitask 头维度不匹配
def test_multitask_head_dim_mismatch_raises(predictor, window_batch):
    wb, pb = window_batch
    # encode 产出 latent_dim=8, 但 SpatialCoordHead 期望 latent_dim=32 => matmul 不匹配
    mth = MultiTaskHead(predictor, latent_dim=8, enable=True)
    mth.register_head("big", SpatialCoordHead(32, n_pts=4))
    with pytest.raises(RuntimeError):
        mth.forward(wb, scene_params=pb)


# 3. future_state horizon=0
def test_loop_horizon_zero_rejected(predictor):
    with pytest.raises(ValueError):
        PhysicalLoopRunner(predictor, horizon=0)
    with pytest.raises(ValueError):
        PhysicalLoopRunner(predictor, horizon=-3)


# 4. feedback 极端偏差率
def test_feedback_extreme_deviation_triggers_no_weight_change(predictor, window_batch):
    wb, pb = window_batch
    loop = PhysicalLoopRunner(predictor, horizon=2)
    loop.dev_threshold = -1.0   # 极端: 任何偏差 (含 0) 都触发
    out = loop.run(wb, scene_params=pb, candidate_actions=[{}])
    corr = out["loop_state"]["correction"]
    assert corr["triggered"] is True
    assert corr["reason"] == "deviation"
    # 默认 opt-in 关 => 绝不改权重/校准
    assert corr["weights_modified"] is False
    assert corr["recalibrated"] is False


# 5. 服务端点未训练态
def test_endpoints_not_trained():
    svc = UDOSService(preset="small")
    with pytest.raises(Exception):  # ServiceNotReady -> 409
        svc.loop_step({"window": [[0.0] * 6] * 6})
    with pytest.raises(Exception):
        svc.multitask_predict({"window": [[0.0] * 6] * 6})
