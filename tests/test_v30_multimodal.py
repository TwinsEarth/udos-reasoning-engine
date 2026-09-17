"""
v3.0.0 FutureMultimodalHead 单元测试
=====================================
锚点纪律:
    * RGB 代理 [B,H,rgb_dim=8] / 深度代理 [B,H,depth_dim=4] /
      对象 mask 代理 [B,H,mask_dim=4] 三模态输出有限;
    * 三模态共享同一份 latent (backbone), 推理模式下不污染主模型梯度;
    * 各模态独立可开关 (use_rgb/use_depth/use_mask);
    * 默认关 (MultiTaskHead.enable=False) 时旧 predict_next 路径逐位一致;
    * 与 multitask 注册框架兼容; config_dict/load_config 往返一致。
"""
from pathlib import Path

import pytest
import torch

from udos import __version__
from udos.multitask import MultiTaskHead
from udos.future_multimodal import FutureMultimodalHead
from udos.persistence import load_predictor
from udos.dynamics import build_parametric_dataset

ROOT = Path(__file__).resolve().parents[1]
CKPT = str(ROOT / "checkpoints" / "predictor_v2.9.0.pt")


@pytest.fixture(scope="module")
def predictor():
    model, _ = load_predictor(CKPT)
    return model


@pytest.fixture(scope="module")
def window_batch():
    ds = build_parametric_dataset(n_per_kind=8, n_steps=14, window=6,
                                  horizon=4, dt=0.5, seed=3030)
    return ds.X[:1], ds.P[:1]


def _build(predictor, **kw):
    mth = MultiTaskHead(predictor, latent_dim=32, enable=True)
    torch.manual_seed(0)
    mth.register_head("future_mm", FutureMultimodalHead(32, horizon=4, **kw))
    return mth


def test_version():
    assert __version__ == "5.5.5"


def test_three_modality_shapes_finite(predictor, window_batch):
    wb, pb = window_batch
    mth = _build(predictor)
    out = mth.forward(wb, scene_params=pb)
    mm = out["future_mm"]
    assert mm["rgb"].shape == (1, 4, 8)
    assert mm["depth"].shape == (1, 4, 4)
    assert mm["mask"].shape == (1, 4, 4)
    for k in ("rgb", "depth", "mask"):
        assert torch.isfinite(mm[k]).all(), f"{k} 含非有限值"


def test_shared_backbone_no_grad_conflict(predictor, window_batch):
    wb, pb = window_batch
    mth = _build(predictor)
    out = mth.forward(wb, scene_params=pb)
    assert "future_mm" in out
    for p in predictor.parameters():
        assert p.grad is None


def test_modality_independent_toggle(predictor, window_batch):
    wb, pb = window_batch
    # 只开 RGB
    mth = _build(predictor, use_depth=False, use_mask=False)
    mm = mth.forward(wb, scene_params=pb)["future_mm"]
    assert set(mm.keys()) == {"rgb"}
    # 只开 depth
    mth = _build(predictor, use_rgb=False, use_mask=False)
    mm = mth.forward(wb, scene_params=pb)["future_mm"]
    assert set(mm.keys()) == {"depth"}
    # 只开 mask
    mth = _build(predictor, use_rgb=False, use_depth=False)
    mm = mth.forward(wb, scene_params=pb)["future_mm"]
    assert set(mm.keys()) == {"mask"}


def test_head_disabled_old_path_unchanged(predictor, window_batch):
    wb, pb = window_batch
    mth = _build(predictor)
    mth.enable = False
    assert mth.forward(wb, scene_params=pb) == {}
    before = predictor.predict_next(wb, scene_params=pb)
    mth.enable = True
    mth.forward(wb, scene_params=pb)
    after = predictor.predict_next(wb, scene_params=pb)
    assert torch.equal(before, after)


def test_config_roundtrip():
    torch.manual_seed(0)
    head = FutureMultimodalHead(32, horizon=4, rgb_dim=8, depth_dim=4,
                                mask_dim=4, use_rgb=False)
    cfg = head.config_dict()
    head2 = FutureMultimodalHead(32, horizon=2, rgb_dim=8, depth_dim=4,
                                 mask_dim=4)
    head2.load_config(cfg)
    assert head2.horizon == 4
    assert head2.use_rgb is False
    assert head2.rgb_dim == 8


def test_bad_horizon_rejected():
    with pytest.raises(ValueError):
        FutureMultimodalHead(32, horizon=0)
