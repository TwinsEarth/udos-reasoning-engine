"""
v3.0.0.dev1 RGBProxyHead + DepthProxyHead 单元测试
====================================================
锚点纪律:
    * RGB 代理头输出 [B,H,rgb_dim=8] 有限 (颜色统计通道);
    * 深度代理头输出 [B,H,depth_dim=4] 有限 (深度排序通道);
    * 投影可逆性检查 (合成数据): 验证头是确定性仿射/线性投影 (可叠加、非退化);
    * 两头独立损失 + 联合推理; 头单独关闭不影响另一头与旧 predictor 输出。
"""
from pathlib import Path

import pytest
import torch

from udos import __version__
from udos.future_multimodal import RGBProxyHead, DepthProxyHead
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


def test_rgb_head_shape_finite():
    torch.manual_seed(0)
    head = RGBProxyHead(32, horizon=4, rgb_dim=8)
    z = torch.randn(3, 32)
    out = head(z)
    assert out.shape == (3, 4, 8)
    assert torch.isfinite(out).all()


def test_depth_head_shape_finite():
    torch.manual_seed(0)
    head = DepthProxyHead(32, horizon=4, depth_dim=4)
    z = torch.randn(3, 32)
    out = head(z)
    assert out.shape == (3, 4, 4)
    assert torch.isfinite(out).all()


def test_projection_affine_superposition():
    """投影可逆性/线性性检查 (合成数据): 仿射映射满足
    f(a+b) - f(0) == (f(a)-f(0)) + (f(b)-f(0)), 且非退化。"""
    torch.manual_seed(0)
    for Head in (RGBProxyHead, DepthProxyHead):
        head = Head(32, horizon=4)
        head.eval()
        a = torch.randn(2, 32)
        b = torch.randn(2, 32)
        zero = torch.zeros(2, 32)
        fa, fb, f0 = head(a), head(b), head(zero)
        # 仿射叠加性
        lhs = head(a + b) - f0
        rhs = (fa - f0) + (fb - f0)
        assert torch.allclose(lhs, rhs, atol=1e-5), f"{Head.__name__} 不满足仿射叠加"
        # 非退化: 不同 latent 应给出不同输出 (投影不是常数)
        assert not torch.allclose(head(a), head(-a), atol=1e-3)


def test_independent_loss_and_joint_inference():
    """两头独立可算损失, 也可联合推理。"""
    torch.manual_seed(0)
    rgb = RGBProxyHead(32, horizon=4, rgb_dim=8)
    dep = DepthProxyHead(32, horizon=4, depth_dim=4)
    z = torch.randn(4, 32)
    rgb_t = torch.randn(4, 4, 8)
    dep_t = torch.randn(4, 4, 4)
    rgb_loss = ((rgb(z) - rgb_t) ** 2).mean()
    dep_loss = ((dep(z) - dep_t) ** 2).mean()
    assert torch.isfinite(rgb_loss) and torch.isfinite(dep_loss)
    # 各自损失独立 (互不依赖对方权重)
    rgb_loss.backward()
    assert rgb.proj.weight.grad is not None
    assert dep.proj.weight.grad is None


def test_heads_no_conflict_with_predictor(predictor):
    ds = build_parametric_dataset(n_per_kind=8, n_steps=14, window=6,
                                  horizon=4, dt=0.5, seed=404)
    wb, pb = ds.X[:1], ds.P[:1]
    before = predictor.predict_next(wb, scene_params=pb)
    torch.manual_seed(0)
    rgb = RGBProxyHead(32, horizon=4)
    dep = DepthProxyHead(32, horizon=4)
    latent = predictor.obs_encoder(wb).mean(dim=1)
    _ = rgb(latent)
    _ = dep(latent)
    after = predictor.predict_next(wb, scene_params=pb)
    assert torch.equal(before, after)


def test_bad_dims_rejected():
    with pytest.raises(ValueError):
        RGBProxyHead(32, horizon=0)
    with pytest.raises(ValueError):
        DepthProxyHead(32, horizon=4, depth_dim=0)
