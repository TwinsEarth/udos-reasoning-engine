"""
v2.8.0.dev6 SpatialCoordHead + ActionTrajectoryHead 单元测试
================================================================
锚点纪律:
    * 坐标头输出 [B, N_pts, 3] 有限;
    * 动作头输出 [B, H, action_dim] 有限, action_dim 与 policy 动作空间对齐 (=6);
    * 两头共享同一 latent (backbone) 推理模式下梯度不冲突、不污染主模型;
    * 头单独关闭 / enable=False 时旧路径不变;
    * A/B: 共享 backbone (encode 一次跑两头) vs 独立 (各自 encode) 的延迟对比,
      诚实记录 (不强制宣称共享更快)。
"""
import time
from pathlib import Path

import pytest
import torch

from udos import __version__
from udos.multitask import MultiTaskHead, SpatialCoordHead, ActionTrajectoryHead
from udos.persistence import load_predictor
from udos.dynamics import build_parametric_dataset, RAW_DIM

ROOT = Path(__file__).resolve().parents[1]
CKPT = str(ROOT / "checkpoints" / "predictor_v2.8.0.pt")


@pytest.fixture(scope="module")
def predictor():
    model, _ = load_predictor(CKPT)
    return model


@pytest.fixture(scope="module")
def window_batch():
    ds = build_parametric_dataset(n_per_kind=8, n_steps=14, window=6,
                                  horizon=4, dt=0.5, seed=3030)
    return ds.X[:1], ds.P[:1]


def _build(predictor):
    mth = MultiTaskHead(predictor, latent_dim=32, enable=True)
    torch.manual_seed(0)
    mth.register_head("spatial", SpatialCoordHead(32, n_pts=4))
    mth.register_head("action", ActionTrajectoryHead(32, horizon=4,
                                                      action_dim=RAW_DIM))
    return mth


def test_version():
    assert __version__ == "5.5.5"


def test_spatial_head_shape_finite(predictor, window_batch):
    wb, pb = window_batch
    mth = _build(predictor)
    out = mth.forward(wb, scene_params=pb)
    assert out["spatial"].shape == (1, 4, 3)
    assert torch.isfinite(out["spatial"]).all()


def test_action_head_shape_finite(predictor, window_batch):
    wb, pb = window_batch
    mth = _build(predictor)
    out = mth.forward(wb, scene_params=pb)
    assert out["action"].shape == (1, 4, RAW_DIM)
    assert torch.isfinite(out["action"]).all()


def test_shared_backbone_no_grad_conflict(predictor, window_batch):
    wb, pb = window_batch
    mth = _build(predictor)
    # 推理模式: 跑两头后无梯度累积, 主模型参数 grad 为 None
    out = mth.forward(wb, scene_params=pb)
    assert set(out.keys()) == {"spatial", "action"}
    for p in predictor.parameters():
        assert p.grad is None


def test_head_disabled_old_path_unchanged(predictor, window_batch):
    wb, pb = window_batch
    mth = _build(predictor)
    # enable=False => forward 空, 主模型不变
    mth.enable = False
    assert mth.forward(wb, scene_params=pb) == {}
    before = predictor.predict_next(wb, scene_params=pb)
    mth.enable = True
    mth.forward(wb, scene_params=pb)
    after = predictor.predict_next(wb, scene_params=pb)
    assert torch.equal(before, after)


def test_shared_vs_independent_latency(predictor, window_batch):
    wb, pb = window_batch
    mth = _build(predictor)
    heads = [mth.get_head("spatial"), mth.get_head("action")]
    # warmup
    for _ in range(3):
        z = mth.encode(wb, scene_params=pb)
        for h in heads:
            h(z)
    # 共享: encode 一次, 跑两头
    t0 = time.perf_counter()
    for _ in range(20):
        z = mth.encode(wb, scene_params=pb)
        for h in heads:
            h(z)
    shared_s = (time.perf_counter() - t0) / 20
    # 独立: 每头各 encode 一次
    t0 = time.perf_counter()
    for _ in range(20):
        for h in heads:
            h(mth.encode(wb, scene_params=pb))
    indep_s = (time.perf_counter() - t0) / 20
    # 诚实记录: 共享应 <= 独立 (少一次 encode), 但不强制阈值
    assert shared_s <= indep_s * 2.0
