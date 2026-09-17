"""v3.5.0.dev2 OccupancyGrid + DistanceField 测试。"""
import pytest

from udos import __version__
from udos.spatial import SpatialObject
from udos.occupancy import OccupancyGrid, DistanceField


def test_version():
    assert __version__ == "5.5.5"


def test_occupy_ball():
    g = OccupancyGrid(bounds=((-1, -1, -1), (1, 1, 1)), resolution=16)
    n = g.occupy_object(SpatialObject("o", [0.0, 0.0, 0.0], radius=0.4))
    assert n > 0
    assert g.n_occupied == n
    # 球心占据
    assert g.is_occupied([0.0, 0.0, 0.0]) is True
    # 远处自由
    assert g.is_occupied([0.9, 0.9, 0.9]) is False


def test_occupy_out_of_bounds_raises():
    g = OccupancyGrid(bounds=((0, 0, 0), (1, 1, 1)), resolution=8)
    with pytest.raises(ValueError):
        g.is_occupied([5.0, 0.5, 0.5])


def test_invalid_bounds_resolution():
    with pytest.raises(ValueError):
        OccupancyGrid(bounds=((1, 1, 1), (0, 0, 0)))  # hi<lo
    with pytest.raises(ValueError):
        OccupancyGrid(resolution=0)
    with pytest.raises(ValueError):
        OccupancyGrid().occupy_object("not-object")


def test_distance_field_sign():
    g = OccupancyGrid(bounds=((-1, -1, -1), (1, 1, 1)), resolution=16)
    g.occupy_object(SpatialObject("o", [0.0, 0.0, 0.0], radius=0.3))
    df = DistanceField(g)
    # 占据点 sdf 为负
    assert df.at([0.0, 0.0, 0.0]) < 0
    # 远离障碍 sdf 为正
    assert df.at([0.9, 0.9, 0.9]) > 0
    # sdf 形状
    assert df.sdf.shape == (16, 16, 16)


def test_distance_field_gradient_monotonic():
    """随距离障碍越远, sdf 单调不降 (自由侧)。"""
    g = OccupancyGrid(bounds=((-1, -1, -1), (1, 1, 1)), resolution=24)
    g.occupy_object(SpatialObject("o", [0.0, 0.0, 0.0], radius=0.2))
    df = DistanceField(g)
    vals = [df.at([d, 0.0, 0.0]) for d in (0.3, 0.5, 0.7, 0.9)]
    assert vals == sorted(vals)


def test_empty_occupancy_guard():
    g = OccupancyGrid(resolution=8)
    assert g.n_occupied == 0
    with pytest.raises(ValueError):
        DistanceField(g).at([0.0, 0.0, 0.0])


def test_voxel_center_roundtrip():
    g = OccupancyGrid(bounds=((0, 0, 0), (1, 1, 1)), resolution=8)
    c = g.voxel_center((0, 0, 0))
    assert c.shape == (3,)
    with pytest.raises(ValueError):
        g.voxel_center((99, 0, 0))
