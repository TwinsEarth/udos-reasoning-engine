"""v3.5.0.dev3 碰撞检测 + 最近邻测试。"""
import pytest

from udos import __version__
from udos.spatial import SpatialObject, SpatialScene
from udos.collision import CollisionDetector, NearestNeighbor


def test_version():
    assert __version__ == "5.5.5"


def test_ball_ball_contact():
    cd = CollisionDetector()
    a = SpatialObject("a", [0, 0, 0], radius=0.5)
    b = SpatialObject("b", [0.8, 0, 0], radius=0.5)   # 中心距0.8 < 1.0
    r = cd.ball_ball(a, b)
    assert r["contact"] is True
    assert r["penetration"] == pytest.approx(0.2)
    c = SpatialObject("c", [5.0, 0, 0], radius=0.5)
    assert cd.ball_ball(a, c)["contact"] is False


def test_ball_box():
    cd = CollisionDetector()
    # 球心在盒内
    r = cd.ball_box([0.0, 0.0, 0.0], 0.3, [-1, -1, -1], [1, 1, 1])
    assert r["contact"] is True
    # 球心在盒外远处
    r2 = cd.ball_box([5.0, 0.0, 0.0], 0.3, [-1, -1, -1], [1, 1, 1])
    assert r2["contact"] is False
    with pytest.raises(ValueError):
        cd.ball_box([0, 0, 0], 0.3, [1, 1, 1], [0, 0, 0])  # min>max


def test_detect_contacts_scene():
    sc = SpatialScene([
        SpatialObject("a", [0, 0, 0], radius=0.5),
        SpatialObject("b", [0.6, 0, 0], radius=0.5),   # 与 a 接触
        SpatialObject("c", [10, 10, 10], radius=0.5),   # 远离
    ])
    cd = CollisionDetector()
    contacts = cd.detect_contacts(sc)
    assert len(contacts) == 1
    assert {contacts[0]["a"], contacts[0]["b"]} == {"a", "b"}


def test_continuous_contact_event():
    cd = CollisionDetector()
    prev = SpatialScene([
        SpatialObject("a", [0, 0, 0], radius=0.5),
        SpatialObject("b", [5.0, 0, 0], radius=0.5),
    ])
    cur = SpatialScene([
        SpatialObject("a", [0, 0, 0], radius=0.5),
        SpatialObject("b", [0.6, 0, 0], radius=0.5),   # 新进入接触
    ])
    events = cd.continuous_contact(prev, cur)
    assert len(events) == 1
    assert {events[0]["a"], events[0]["b"]} == {"a", "b"}


def test_contacts_guard():
    cd = CollisionDetector()
    with pytest.raises(ValueError):
        cd.detect_contacts(SpatialScene())       # <2
    with pytest.raises(ValueError):
        cd.detect_contacts(SpatialScene([SpatialObject("a", [0, 0, 0])]))


def test_nearest_neighbor():
    sc = SpatialScene([
        SpatialObject("a", [0, 0, 0]),
        SpatialObject("b", [1.0, 0, 0]),
        SpatialObject("c", [10.0, 0, 0]),
    ])
    nn = NearestNeighbor(sc)
    res = nn.query([0.1, 0, 0], k=2)
    assert [r["object_id"] for r in res] == ["a", "b"]
    # 以 b 自身为查询点, 自动剔除自身
    own = nn.nearest_to_object("b", k=1)
    assert own[0]["object_id"] == "a"


def test_nearest_guard():
    nn = NearestNeighbor(SpatialScene())
    with pytest.raises(ValueError):
        nn.query([0, 0, 0])
    sc = SpatialScene([SpatialObject("only", [0, 0, 0])])
    nn2 = NearestNeighbor(sc)
    with pytest.raises(ValueError):
        nn2.nearest_to_object("only")  # 剔除自身后为空
    with pytest.raises(ValueError):
        nn2.query([0, 0, 0], k=0)
