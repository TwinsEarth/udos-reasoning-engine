"""v3.5.0.dev5 空间查询引擎测试: 射线/视线/区域/范围。"""
import numpy as np
import pytest

from udos import __version__
from udos.spatial import SpatialObject, SpatialScene
from udos.spatial_query import SpatialQueryEngine


def test_version():
    assert __version__ == "5.5.5"


def _scene():
    return SpatialScene([
        SpatialObject("target", [5.0, 0.0, 0.0], radius=0.5),
        SpatialObject("wall", [2.5, 0.0, 0.0], radius=0.4),     # 挡住视线
        SpatialObject("far", [10.0, 10.0, 10.0], radius=0.5),
    ])


def test_ray_sphere_hit():
    r = SpatialQueryEngine.ray_sphere(
        origin=[0, 0, 0], direction=[1, 0, 0], center=[5, 0, 0], radius=0.5)
    assert r["hit"] is True
    assert r["t_near"] == pytest.approx(4.5)


def test_ray_sphere_miss():
    r = SpatialQueryEngine.ray_sphere(
        origin=[0, 0, 0], direction=[0, 1, 0], center=[5, 0, 0], radius=0.5)
    assert r["hit"] is False


def test_raycast_nearest():
    q = SpatialQueryEngine(_scene())
    hit = q.raycast([0, 0, 0], [1, 0, 0])
    # wall 在 2.5, target 在 5; 最近命中应为 wall
    assert hit["hit"] is True
    assert hit["object_id"] == "wall"


def test_raycast_no_hit():
    q = SpatialQueryEngine(_scene())
    hit = q.raycast([0, 0, 0], [0, 1, 0])
    assert hit["hit"] is False


def test_line_of_sight_blocked():
    q = SpatialQueryEngine(_scene())
    res = q.line_of_sight("target", "far")
    # target(5,0,0) 到 far(10,10,10) 不经过 wall(2.5,0,0) -> 应可见
    assert res["visible"] is True
    # 从原点方向看: target 到 wall 之间无遮挡
    res2 = q.line_of_sight("wall", "target")
    assert res2["visible"] is True


def test_line_of_sight_blocked_by_middle():
    sc = SpatialScene([
        SpatialObject("a", [0, 0, 0], radius=0.3),
        SpatialObject("block", [5.0, 0.0, 0.0], radius=0.4),
        SpatialObject("b", [10.0, 0.0, 0.0], radius=0.3),
    ])
    q = SpatialQueryEngine(sc)
    res = q.line_of_sight("a", "b")
    assert res["visible"] is False
    assert res["blocker"] == "block"


def test_range_search():
    q = SpatialQueryEngine(_scene())
    res = q.range_search([0, 0, 0], radius=3.0)
    ids = [r["object_id"] for r in res]
    assert "wall" in ids and "target" not in ids
    # 按距离升序
    dists = [r["distance"] for r in res]
    assert dists == sorted(dists)


def test_box_query():
    q = SpatialQueryEngine(_scene())
    ids = q.box_query((-1, -1, -1), (3, 1, 1))
    assert ids == ["wall"]


def test_empty_guard_and_validation():
    q = SpatialQueryEngine(SpatialScene())
    with pytest.raises(ValueError):
        q.raycast([0, 0, 0], [1, 0, 0])
    with pytest.raises(ValueError):
        q.range_search([0, 0, 0], radius=1.0)
    q2 = SpatialQueryEngine(_scene())
    with pytest.raises(ValueError):
        q2.ray_sphere([0, 0, 0], [0, 0, 0], [0, 0, 0], 1.0)  # 零方向
    with pytest.raises(ValueError):
        q2.box_query((1, 1, 1), (0, 0, 0))  # min>max
