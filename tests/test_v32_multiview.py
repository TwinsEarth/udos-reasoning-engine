"""
v3.2.0.dev1 节点42: MultiViewGenerator 多视图生成与视图一致性
================================================================
锚点纪律:
    * 多视图生成形状正确 [K,T,6];
    * 视图变换可逆性 (R_k^T R_k = I);
    * 跨视图一致性误差 ~0 (对应关系矩阵解析已知);
    * 空场景 / 越界视角守卫。
"""
import torch

from udos.ego_data import MultiViewGenerator
from udos.dynamics import build_parametric_dataset


def _base(n=4):
    ds = build_parametric_dataset(n_per_kind=3, n_steps=12, window=6,
                                  horizon=4, dt=0.5, seed=11)
    return ds.X[0]  # [6,6]


def test_generate_shape():
    g = MultiViewGenerator(n_views=4, angle_step_deg=45.0)
    views = g.generate(_base())
    assert views.shape == (4, 6, 6)
    assert torch.isfinite(views).all()
    # 第 0 视角 (angle 0) 应与基准逐位一致
    assert torch.allclose(views[0], _base(), atol=1e-6)


def test_view_invertibility():
    g = MultiViewGenerator(n_views=5, angle_step_deg=30.0)
    base = _base()
    err = g.invertibility_error(base)
    assert err < 1e-5


def test_cross_view_consistency():
    g = MultiViewGenerator(n_views=4, angle_step_deg=45.0)
    views = g.generate(_base())
    err = g.cross_view_consistency(views)
    assert err < 1e-5


def test_correspondence_matrix():
    g = MultiViewGenerator(n_views=3, angle_step_deg=60.0)
    # 自对应矩阵 C[k,k] = I
    I = g.correspondence(1, 1)
    assert torch.allclose(I, torch.eye(2), atol=1e-6)
    # C[k,j] 把 view_k 映到 view_j: 用单一点验证
    pt = torch.tensor([[1.0, 0.0, 0.0, 0.5, 0.0, 0.0]])
    vk = g.apply_view(pt, 0)
    vj = g.apply_view(pt, 2)
    mapped = _rotate_xy_manual(vk, g.correspondence(0, 2))
    assert torch.allclose(mapped, vj, atol=1e-6)


def _rotate_xy_manual(seq, R):
    out = seq.clone()
    out[..., 0:2] = out[..., 0:2] @ R.T
    out[..., 3:5] = out[..., 3:5] @ R.T
    return out


def test_view_angles():
    g = MultiViewGenerator(n_views=4, angle_step_deg=45.0)
    assert g.view_angles_deg() == [0.0, 45.0, 90.0, 135.0]


def test_bad_views_guard():
    try:
        MultiViewGenerator(n_views=1)
        assert False
    except ValueError:
        pass
    g = MultiViewGenerator(n_views=2)
    try:
        g.apply_view(_base(), 5)
        assert False
    except ValueError:
        pass


def test_empty_scene_guard():
    g = MultiViewGenerator(n_views=3)
    try:
        g.generate(torch.zeros(0, 6))
        assert False
    except ValueError:
        pass
