"""v3.5.3 SFM 边界精修: 零体积/共面碰撞/极端坐标/空查询/数值退化。"""
import math

import numpy as np
import pytest

from udos import __version__
from udos.spatial import (SpatialObject, SpatialScene, SpatialTransform,
                          OrthographicView)
from udos.scene_graph import SceneGraph
from udos.collision import CollisionDetector
from udos.spatial_query import SpatialQueryEngine
from udos.occupancy import OccupancyGrid


def test_version():
    assert __version__ == "5.5.5"


def test_zero_radius_rejected():
    """零体积物体在构造期显式拒绝 (不进入运行期)。"""
    with pytest.raises(ValueError):
        SpatialObject("zero", [0, 0, 0], radius=0.0)
    with pytest.raises(ValueError):
        SpatialObject("neg", [0, 0, 0], radius=-0.1)


def test_coplanar_touching_collision():
    """两球恰好相切 (中心距 = r1+r2) 视为接触。"""
    cd = CollisionDetector()
    a = SpatialObject("a", [0, 0, 0], radius=0.5)
    b = SpatialObject("b", [1.0, 0, 0], radius=0.5)  # 中心距 1.0 = 和
    r = cd.ball_ball(a, b)
    assert r["contact"] is True
    assert r["penetration"] == pytest.approx(0.0, abs=1e-9)


def test_extreme_coordinates():
    """大坐标/近边界数值稳定 (不溢出、有限)。"""
    sc = SpatialScene([
        SpatialObject("big", [1e6, -1e6, 0.0], radius=1.0),
        SpatialObject("small", [1e6 + 2.0, -1e6, 0.0], radius=1.0),
    ])
    cd = CollisionDetector()
    contacts = cd.detect_contacts(sc)
    assert len(contacts) == 1  # 中心距 2.0 = 2r
    # 变换极端缩放不报错
    t = SpatialTransform.scaling([1e-3, 1e3, 1.0])
    out = sc.apply_transform(t)
    assert np.isfinite(out.position_matrix()).all()


def test_ray_tangent_to_sphere():
    """射线与球相切 (disc=0) 的退化情形不崩。"""
    r = SpatialQueryEngine.ray_sphere(
        origin=[0, -1.0, 0], direction=[1, 0, 0], center=[0, 0, 0], radius=1.0)
    # 切线: 距离球心 1 = 半径, 恰切
    assert r["hit"] in (True, False)
    assert r["t_near"] is None or r["t_near"] >= 0


def test_empty_query_guards():
    q = SpatialQueryEngine(SpatialScene())
    with pytest.raises(ValueError):
        q.box_query([0, 0, 0], [1, 1, 1])
    g = SceneGraph(SpatialScene())
    with pytest.raises(ValueError):
        g.edges()
    occ = OccupancyGrid(resolution=4)
    assert occ.n_occupied == 0


def test_degenerate_rotation_identity_roundtrip():
    """0 角度旋转 (恒等) 可逆。"""
    t = SpatialTransform.rotation_from_axis_angle([0, 0, 1], 0.0)
    pts = np.random.RandomState(0).randn(5, 3)
    assert t.roundtrip_error(pts) < 1e-12


def test_multiview_degenerate_up_rejected():
    with pytest.raises(ValueError):
        OrthographicView(eye=[0, 0, 5], look_at=[0, 0, 0], up_hint=[0, 0, 1])


def test_occupancy_point_on_boundary():
    """查询点恰在网格边界 -> 夹到合法体素, 不越界报错。"""
    g = OccupancyGrid(bounds=((0, 0, 0), (1, 1, 1)), resolution=8)
    # 恰在 max 边界
    assert g.is_occupied([1.0, 0.5, 0.5]) in (True, False)


def test_scene_graph_coplanar_no_above():
    """两物体水平等高 => above/below 均为假。"""
    sc = SpatialScene([
        SpatialObject("a", [0, 0, 0.0], radius=0.3),
        SpatialObject("b", [0.5, 0.0, 0.0], radius=0.3),
    ])
    g = SceneGraph(sc, near_dist=0.4)
    r = g.relate("a", "b")
    assert r["above"] is False
    assert r["below"] is False
