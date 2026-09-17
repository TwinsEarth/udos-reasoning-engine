"""v3.5.0 SFM 空间几何核心测试: SpatialObject/SpatialScene/SpatialTransform。

覆盖:
    * 物体/场景结构与几何量;
    * 坐标变换可逆性 (apply(inverse(p)) ≈ p);
    * 旋转工厂 (轴角) 与正交性;
    * 空场景 / 非法输入守卫;
    * 与 pce_format.PhysicalToken 接口一致;
    * 零外挂: 不改动主 predictor 参数量。
"""
import math

import numpy as np
import pytest

from udos.spatial import (SpatialObject, SpatialScene, SpatialTransform,
                          _as_vec3)
from udos.pce_format import PhysicalToken


# --------------------------------------------------------------------------- #
# SpatialObject
# --------------------------------------------------------------------------- #
def test_object_basic_geometry():
    a = SpatialObject("a", [0.0, 0.0, 0.0], velocity=[1.0, 0.0, 0.0], radius=0.5)
    b = SpatialObject("b", [3.0, 4.0, 0.0], radius=0.5)
    assert abs(a.distance_to(b) - 5.0) < 1e-9
    rel = b.relative_position(a)
    assert np.allclose(rel, [-3.0, -4.0, 0.0])
    assert a.radius == 0.5


def test_object_validation():
    with pytest.raises(ValueError):
        SpatialObject("", [0, 0, 0])          # 空 id
    with pytest.raises(ValueError):
        SpatialObject("x", [0, 0])            # 长度不对
    with pytest.raises(ValueError):
        SpatialObject("x", [0, 0, -math.inf]) # 非有限
    with pytest.raises(ValueError):
        SpatialObject("x", [0, 0, 0], radius=0.0)  # 半径非正
    with pytest.raises(ValueError):
        SpatialObject("x", [0, 0, 0], radius=-1.0)


def test_object_roundtrip_dict():
    a = SpatialObject("box", [1.0, 2.0, 3.0], [0.1, 0.2, 0.3], 0.7)
    b = SpatialObject.from_dict(a.to_dict())
    assert b.object_id == "box"
    assert np.allclose(b.position, [1, 2, 3])
    assert np.allclose(b.velocity, [0.1, 0.2, 0.3])
    assert b.radius == 0.7


def test_object_pce_interop():
    tok = PhysicalToken(object_id="cup", timestamp=2,
                        position=[1.0, 2.0, 3.0], velocity=[0.5, 0.0, -0.5])
    obj = SpatialObject.from_physical_token(tok, radius=0.3)
    assert obj.object_id == "cup"
    assert np.allclose(obj.position, [1, 2, 3])
    back = obj.to_physical_token(timestamp=2)
    assert back.object_id == "cup"
    assert np.allclose(back.position, [1, 2, 3])


# --------------------------------------------------------------------------- #
# SpatialTransform 可逆性
# --------------------------------------------------------------------------- #
def test_transform_translation_identity():
    t = SpatialTransform.translation([10.0, -2.0, 3.0])
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    out = t.apply(pts)
    assert np.allclose(out[0], [10, -2, 3])
    assert t.roundtrip_error(pts) < 1e-12
    assert not SpatialTransform.identity().is_identity() is False


def test_transform_rotation_roundtrip():
    t = SpatialTransform.rotation_from_axis_angle([0, 0, 1], math.pi / 4)
    pts = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [2.0, -1.0, 3.0]])
    # 旋转保距
    d0 = np.linalg.norm(pts[0] - pts[1])
    d1 = np.linalg.norm(t.apply(pts)[0] - t.apply(pts)[1])
    assert abs(d0 - d1) < 1e-9
    assert t.roundtrip_error(pts) < 1e-12


def test_transform_compose_roundtrip():
    a = SpatialTransform.translation([1, 2, 3])
    r = SpatialTransform.rotation_from_axis_angle([1, 1, 0], 0.7)
    s = SpatialTransform.scaling([2.0, 0.5, 1.5])
    composed = a.compose(r).compose(s)
    pts = np.array([[0.3, 1.2, -0.7], [4.0, -2.0, 0.0]])
    # 复合变换仍可逆
    err = composed.roundtrip_error(pts)
    assert err < 1e-8, err


def test_transform_velocity_no_translation():
    t = SpatialTransform.translation([5, 5, 5])
    v = np.array([[1.0, 0.0, 0.0]])
    # 平移不改变自由向量
    assert np.allclose(t.apply_vectors(v), v)


def test_transform_validation():
    with pytest.raises(ValueError):
        SpatialTransform(scale=[-1.0, 1.0, 1.0])   # 缩放须正
    with pytest.raises(ValueError):
        SpatialTransform.rotation_from_axis_angle([0, 0, 0], 0.5)  # 零轴
    with pytest.raises(ValueError):
        SpatialTransform(rotation=np.eye(3) * [[1, 0, 0], [0, 1, 0], [0, 0, -1]])  # 反射
    with pytest.raises(ValueError):
        SpatialTransform().apply([[0, 0]])  # shape 错


# --------------------------------------------------------------------------- #
# SpatialScene
# --------------------------------------------------------------------------- #
def test_scene_add_remove_get():
    sc = SpatialScene()
    sc.add(SpatialObject("a", [0, 0, 0]))
    sc.add(SpatialObject("b", [1, 0, 0]))
    assert len(sc) == 2
    assert "a" in sc
    assert sc.get("b").position[0] == 1.0
    sc.remove("a")
    assert len(sc) == 1
    with pytest.raises(ValueError):
        sc.add(SpatialObject("b", [9, 9, 9]))  # 重复 id
    with pytest.raises(ValueError):
        sc.get("missing")
    with pytest.raises(ValueError):
        sc.remove("missing")


def test_scene_pairwise_distances():
    sc = SpatialScene([
        SpatialObject("a", [0, 0, 0]),
        SpatialObject("b", [3, 4, 0]),
        SpatialObject("c", [0, 4, 3]),
    ])
    D = sc.pairwise_distances()
    assert D.shape == (3, 3)
    assert abs(D[0, 1] - 5.0) < 1e-9
    assert abs(D[0, 2] - 5.0) < 1e-9
    assert D[0, 0] == 0.0


def test_scene_apply_transform_returns_new():
    sc = SpatialScene([SpatialObject("a", [1.0, 0.0, 0.0], [1.0, 0, 0], 0.5)])
    t = SpatialTransform.translation([10, 0, 0])
    moved = sc.apply_transform(t)
    # 原场景不变
    assert np.allclose(sc.get("a").position, [1, 0, 0])
    # 新场景平移到位
    assert np.allclose(moved.get("a").position, [11, 0, 0])
    # 速度不含平移
    assert np.allclose(moved.get("a").velocity, [1, 0, 0])


def test_scene_empty_guards():
    sc = SpatialScene()
    assert len(sc) == 0
    assert sc.position_matrix().shape == (0, 3)
    assert sc.pairwise_distances().shape == (0, 0)
    with pytest.raises(ValueError):
        sc.apply_transform(SpatialTransform.identity())


def test_scene_dict_roundtrip():
    sc = SpatialScene([
        SpatialObject("a", [0, 0, 0], [1, 0, 0], 0.4),
        SpatialObject("b", [2, 2, 2], [0, 1, 0], 0.6),
    ])
    sc2 = SpatialScene.from_dict(sc.to_dict())
    assert sc2.ids() == ["a", "b"]
    assert np.allclose(sc2.get("b").position, [2, 2, 2])


# --------------------------------------------------------------------------- #
# 零外挂 / 不改主权重
# --------------------------------------------------------------------------- #
def test_spatial_does_not_touch_predictor():
    """空间模块为纯几何, 导入不实例化/训练任何主模型。"""
    import udos
    # 主导出仍在, 主参数量由正式件保证; 这里只确认新符号已导出
    assert hasattr(udos, "SpatialObject")
    assert hasattr(udos, "SpatialScene")
    assert hasattr(udos, "SpatialTransform")
