"""v3.5.0.dev1 SceneGraph 场景图测试: 关系计算 / 图遍历 / 空图守卫。"""
import pytest

from udos import __version__
from udos.spatial import SpatialObject, SpatialScene
from udos.scene_graph import SceneGraph, RELATIONS


def _toy_scene():
    return SpatialScene([
        SpatialObject("table", [0.0, 0.0, 0.0], radius=0.5),
        SpatialObject("cup", [0.0, 0.0, 1.2], radius=0.2),       # 在桌上(above)
        SpatialObject("box", [3.0, 0.0, 0.0], radius=0.6),      # 远(far)
        SpatialObject("ball", [0.3, 0.0, 0.0], radius=0.15),   # 近(near)
    ])


def test_version():
    assert __version__ == "5.5.5"


def test_above_below():
    g = SceneGraph(_toy_scene(), near_dist=1.0)
    r = g.relate("cup", "table")
    assert r["above"] is True
    assert r["below"] is False
    r2 = g.relate("table", "cup")
    assert r2["below"] is True
    assert r2["above"] is False


def test_left_right():
    sc = SpatialScene([
        SpatialObject("L", [-2.0, 0, 0], radius=0.3),
        SpatialObject("R", [2.0, 0, 0], radius=0.3),
    ])
    g = SceneGraph(sc)
    assert g.relate("L", "R")["left"] is True
    assert g.relate("L", "R")["right"] is False
    assert g.relate("R", "L")["right"] is True


def test_near_far():
    g = SceneGraph(_toy_scene(), near_dist=1.0, far_dist=2.0)
    assert g.relate("table", "ball")["near"] is True      # 0.3
    assert g.relate("table", "box")["far"] is True        # 3.0
    assert g.relate("table", "cup")["near"] is False      # 1.2


def test_inside():
    sc = SpatialScene([
        SpatialObject("jar", [0, 0, 0], radius=1.0),
        SpatialObject("coin", [0.2, 0.0, 0.0], radius=0.1),
    ])
    g = SceneGraph(sc)
    r = g.relate("coin", "jar")
    assert r["inside"] is True
    # 反向不成立
    assert g.relate("jar", "coin")["inside"] is False


def test_edges_and_query():
    g = SceneGraph(_toy_scene(), near_dist=1.0, far_dist=2.0)
    # 4 节点有向边数 = 4*3 = 12
    assert len(g.edges()) == 12
    near_edges = g.edges(relation="near")
    assert all(e["near"] for e in near_edges)
    q = g.query("near")
    assert isinstance(q, list)
    assert all(set(e) == {"subject", "object"} for e in q)
    # neighbors
    nb = g.neighbors("table", "near")
    assert "ball" in nb


def test_graph_validation():
    with pytest.raises(ValueError):
        SceneGraph("not-a-scene")
    with pytest.raises(ValueError):
        SceneGraph(_toy_scene(), near_dist=-1.0)
    g = SceneGraph(_toy_scene())
    with pytest.raises(ValueError):
        g.relate("cup", "cup")
    with pytest.raises(ValueError):
        g.relate("cup", "ghost")
    with pytest.raises(ValueError):
        g.edges(relation="on_top")  # 未知关系


def test_empty_graph_guard():
    g = SceneGraph(SpatialScene())
    assert len(g) == 0
    with pytest.raises(ValueError):
        g.edges()
    with pytest.raises(ValueError):
        g.query("near")


def test_relations_constant():
    assert set(RELATIONS) == {"above", "below", "left", "right",
                              "near", "far", "inside"}
