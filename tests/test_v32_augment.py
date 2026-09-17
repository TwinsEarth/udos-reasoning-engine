"""
v3.2.0 节点41: SyntheticEgoAugmenter 合成多视角数据增强
========================================================
锚点纪律:
    * 增强输出形状正确;
    * 视角旋转几何一致 (R^T R = I, 可逆往返);
    * 扰动/噪声幅度可控 (参数=0 逐位不变, 越大偏离越大);
    * (X,Y) 成对增强标签对齐 (几何变换在 X/Y 间一致);
    * 空数据集守卫。
"""
import math

import torch

from udos.ego_data import SyntheticEgoAugmenter, rotation_xy, time_warp
from udos.dynamics import build_parametric_dataset


def _toy_pair(n=8, W=6, H=4):
    ds = build_parametric_dataset(n_per_kind=4, n_steps=W + H + 2,
                                  window=W, horizon=H, dt=0.5, seed=7)
    return ds.X[:n], ds.Y[:n]


def test_output_shape():
    X, Y = _toy_pair()
    aug = SyntheticEgoAugmenter(view_rotate_deg=15.0, traj_perturb=0.02,
                                noise_sigma=0.01, time_scale=1.1, seed=1)
    Xp, Yp = aug.augment_pair(X, Y)
    assert Xp.shape == X.shape
    assert Yp.shape == Y.shape
    assert torch.isfinite(Xp).all() and torch.isfinite(Yp).all()


def test_rotation_is_orthonormal():
    R = rotation_xy(math.radians(37.0))
    assert torch.allclose(R @ R.T, torch.eye(2), atol=1e-6)
    assert torch.allclose(R.T @ R, torch.eye(2), atol=1e-6)
    # 行列式 +1 (纯旋转)
    assert abs(float(torch.det(R)) - 1.0) < 1e-6


def test_view_roundtrip_invertible():
    X, Y = _toy_pair()
    aug = SyntheticEgoAugmenter(view_rotate_deg=25.0,
                                view_translate_xyz=(0.3, -0.2, 0.1))
    Xp, Yp = aug.augment_pair(X, Y)
    # reverse_view 应把增强后的视图几何还原回原始 (扰动/噪声=0, 时间缩放=1)
    back = aug.reverse_view(Xp)
    assert torch.allclose(back, X, atol=1e-5), (back - X).abs().max()


def test_zero_params_bit_identical():
    X, Y = _toy_pair()
    aug = SyntheticEgoAugmenter()  # 全 0 / 1
    Xp, Yp = aug.augment_pair(X, Y)
    assert torch.equal(Xp, X)
    assert torch.equal(Yp, Y)


def test_perturb_controllable():
    X, Y = _toy_pair()
    small = SyntheticEgoAugmenter(traj_perturb=0.001, noise_sigma=0.0, seed=3)
    large = SyntheticEgoAugmenter(traj_perturb=0.5, noise_sigma=0.0, seed=3)
    s = (small.augment_pair(X, Y)[0] - X).abs().mean()
    l = (large.augment_pair(X, Y)[0] - X).abs().mean()
    assert l > s
    # 参数 0 => 完全不变
    none_ = SyntheticEgoAugmenter(traj_perturb=0.0, seed=3)
    assert torch.equal(none_.augment_pair(X, Y)[0], X)


def test_time_scale_shape_preserved():
    X, Y = _toy_pair()
    for s in (0.7, 1.0, 1.3):
        aug = SyntheticEgoAugmenter(time_scale=s)
        Xp, Yp = aug.augment_pair(X, Y)
        assert Xp.shape == X.shape and Yp.shape == Y.shape
    assert torch.equal(time_warp(X, 1.0), X)


def test_label_alignment_view_consistency():
    """视图旋转必须同时作用于 X 末帧与 Y 首帧 (世界帧连续)。"""
    X, Y = _toy_pair()
    aug = SyntheticEgoAugmenter(view_rotate_deg=40.0)
    Xp, Yp = aug.augment_pair(X, Y)
    # 相邻两帧 (X 末帧, Y 首帧) 的速度差在原视角 vs 增强视角应满足同一刚体变换
    d0 = (Y[:, 0, :] - X[:, -1, :])
    d0_rot = aug.apply_view(d0)  # 平移分量应抵消, 只剩旋转速度部分
    d1 = (Yp[:, 0, :] - Xp[:, -1, :])
    # 平移 (常数) 在差分中抵消; 速度分量被旋转, 位置差分也被旋转
    assert torch.allclose(d1, d0_rot, atol=1e-4), (d1 - d0_rot).abs().max()


def test_empty_dataset_guard():
    aug = SyntheticEgoAugmenter(view_rotate_deg=10.0)
    empty = torch.zeros(0, 6, 6)
    empty_y = torch.zeros(0, 4, 6)
    try:
        aug.augment_pair(empty, empty_y)
        assert False, "应抛 ValueError"
    except ValueError:
        pass


def test_bad_constructor_args():
    try:
        SyntheticEgoAugmenter(traj_perturb=-1.0)
        assert False
    except ValueError:
        pass
    try:
        SyntheticEgoAugmenter(time_scale=0.0)
        assert False
    except ValueError:
        pass
