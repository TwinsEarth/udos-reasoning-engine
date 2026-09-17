"""
v3.0.0.dev2 MaskProxyHead 单元测试
====================================
锚点纪律:
    * mask 形状 [B,H,N_obj=4] 有限, 值严格在 [0,1];
    * mask 与 RGB/深度在同一 H 步对齐 (同 horizon);
    * 空场景 (scene_params=None) 守卫正常; 提供 scene_params 时阈值偏置生效;
    * 头单独关闭 (use_mask=False) 不影响 RGB/深度。
"""
from pathlib import Path

import pytest
import torch

from udos import __version__
from udos.future_multimodal import (RGBProxyHead, DepthProxyHead,
                                    MaskProxyHead, FutureMultimodalHead)
from udos.persistence import load_predictor
from udos.dynamics import build_parametric_dataset

ROOT = Path(__file__).resolve().parents[1]
CKPT = str(ROOT / "checkpoints" / "predictor_v2.9.0.pt")


@pytest.fixture(scope="module")
def predictor():
    model, _ = load_predictor(CKPT)
    return model


def test_version():
    assert __version__ == "5.5.5"


def test_mask_shape_finite_range():
    torch.manual_seed(0)
    head = MaskProxyHead(32, horizon=4, mask_dim=4)
    z = torch.randn(3, 32)
    m = head(z)
    assert m.shape == (3, 4, 4)
    assert torch.isfinite(m).all()
    assert (m >= 0).all() and (m <= 1).all()


def test_mask_aligned_with_rgb_depth_H():
    """mask 与 RGB/深度共享同一 horizon H。"""
    torch.manual_seed(0)
    H = 4
    rgb = RGBProxyHead(32, horizon=H, rgb_dim=8)
    dep = DepthProxyHead(32, horizon=H, depth_dim=4)
    msk = MaskProxyHead(32, horizon=H, mask_dim=4)
    z = torch.randn(2, 32)
    assert rgb(z).size(1) == dep(z).size(1) == msk(z).size(1) == H


def test_empty_scene_guard():
    """scene_params=None 时退化为纯 latent 软 mask (仍 [0,1])。"""
    torch.manual_seed(0)
    head = MaskProxyHead(32, horizon=4, mask_dim=4)
    z = torch.randn(2, 32)
    m_none = head(z, scene_params=None)
    assert m_none.shape == (2, 4, 4)
    assert (m_none >= 0).all() and (m_none <= 1).all()


def test_scene_threshold_bias_effect():
    """提供 scene_params 时, 高场景值对应对象的 mask 均值应系统性更高。"""
    torch.manual_seed(0)
    head = MaskProxyHead(32, horizon=4, mask_dim=4, threshold=0.0,
                         scene_bias_scale=5.0)
    z = torch.zeros(16, 32)  # 零 latent => mask 主要由 scene 偏置决定
    sp_low = torch.full((16, 4), -2.0)
    sp_high = torch.full((16, 4), 2.0)
    m_low = head(z, scene_params=sp_low).mean()
    m_high = head(z, scene_params=sp_high).mean()
    assert m_high > m_low
    assert (m_high <= 1).all() and (m_low >= 0).all()


def test_mask_individual_toggle(predictor):
    ds = build_parametric_dataset(n_per_kind=8, n_steps=14, window=6,
                                  horizon=4, dt=0.5, seed=505)
    wb, pb = ds.X[:1], ds.P[:1]
    # 关 mask, 开 RGB+深度
    fmh = FutureMultimodalHead(32, horizon=4, use_mask=False)
    latent = predictor.obs_encoder(wb).mean(dim=1)
    out = fmh(latent, scene_params=pb)
    assert set(out.keys()) == {"rgb", "depth"}
    # 全开关 mask
    fmh2 = FutureMultimodalHead(32, horizon=4)
    out2 = fmh2(latent, scene_params=pb)
    assert set(out2.keys()) == {"rgb", "depth", "mask"}
    assert (out2["mask"] >= 0).all() and (out2["mask"] <= 1).all()


def test_bad_mask_dim_rejected():
    with pytest.raises(ValueError):
        MaskProxyHead(32, horizon=4, mask_dim=0)
