"""v3.5.0.dev4 多视角正交投影一致性测试。"""
import numpy as np
import pytest

from udos import __version__
from udos.spatial import OrthographicView, multiview_consistency_error


def test_version():
    assert __version__ == "5.5.5"


def test_project_unproject_roundtrip():
    cam = OrthographicView(eye=[5.0, 0.0, 0.0], look_at=[0, 0, 0])
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.5, -0.5], [-2.0, 2.0, 1.0]])
    proj = cam.project(pts)
    assert proj["uv"].shape == (3, 2)
    # 逐点用各自深度反投影 (精确)
    back = cam.unproject(proj["uv"], proj["depth"])
    assert np.allclose(back, pts, atol=1e-12)


def test_multiview_consistency():
    cam_a = OrthographicView(eye=[5.0, 0.0, 0.0], look_at=[0, 0, 0])
    cam_b = OrthographicView(eye=[0.0, 5.0, 2.0], look_at=[0, 0, 0],
                             up_hint=(0, 0, 1))
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.5], [-1.0, 0.0, 0.2]])
    err = multiview_consistency_error(cam_a, cam_b, pts)
    assert err < 1e-10, err


def test_view_orthonormal_basis():
    cam = OrthographicView(eye=[3.0, 0.0, 1.0], look_at=[0, 0, 0])
    # right/up/forward 两两正交且单位长
    M = np.stack([cam.right, cam.up, cam.forward], axis=1)
    assert np.allclose(M.T @ M, np.eye(3), atol=1e-10)


def test_view_guards():
    with pytest.raises(ValueError):
        OrthographicView(eye=[0, 0, 0], look_at=[0, 0, 0])   # 视点重合
    with pytest.raises(ValueError):
        OrthographicView(eye=[0, 0, 1], look_at=[0, 0, 0],
                         up_hint=(0, 0, 1))                  # up 与视线平行
    cam = OrthographicView(eye=[5, 0, 0], look_at=[0, 0, 0])
    with pytest.raises(ValueError):
        cam.project([[0, 0]])  # shape 错


def test_view_extreme_angle():
    """极端视角 (贴地斜视) 仍可逆。"""
    cam = OrthographicView(eye=[10.0, 10.0, 0.01], look_at=[0, 0, 0])
    pts = np.array([[0.5, -0.5, 0.0], [0.0, 0.0, 0.0]])
    proj = cam.project(pts)
    back = cam.unproject(proj["uv"], proj["depth"])
    assert np.allclose(back, pts, atol=1e-10)
