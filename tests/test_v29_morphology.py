"""
v2.9.0.dev1 形态配置体系与预设形态库 (MorphologyLibrary) 单测
==============================================================
锚点纪律:
    * 3 种合成预设形态参数正确 (prime_u_60dof / arm_7dof / gripper_4dof);
    * 形态库整体序列化 / 反序列化往返一致;
    * 形态间兼容性检查正确;
    * 未知形态显式 KeyError 守卫;
    * 与 ActionRetargeter 接口一致 (可由两个预设直接构造重定向器)。
analogy, not reproduction: 预设形态均为合成占位本体。
"""
import pytest
import torch

from udos.retargeting import (MorphologyConfig, MorphologyLibrary,
                              ActionRetargeter)


@pytest.fixture(scope="module")
def lib():
    return MorphologyLibrary()


def test_presets_correct(lib):
    names = lib.names()
    assert set(names) == {"prime_u_60dof", "arm_7dof", "gripper_4dof"}
    p = lib.get("prime_u_60dof")
    assert p.dof == 60 and p.control_freq == 100.0
    assert p.low.shape == (60,) and torch.all(p.low <= p.high)
    a = lib.get("arm_7dof")
    assert a.dof == 7 and a.control_freq == 500.0
    g = lib.get("gripper_4dof")
    assert g.dof == 4 and g.control_freq == 1000.0


def test_serialization_roundtrip(lib):
    d = lib.to_dict()
    lib2 = MorphologyLibrary.from_dict(d)
    assert set(lib2.names()) == set(lib.names())
    for n in lib.names():
        c1, c2 = lib.get(n), lib2.get(n)
        assert c1.dof == c2.dof and c1.control_freq == c2.control_freq
        assert torch.equal(c1.low, c2.low) and torch.equal(c1.high, c2.high)
    # 往返库仍可构造重定向器
    rt = ActionRetargeter(lib2.get("arm_7dof"), lib2.get("gripper_4dof"))
    out = rt.retarget(torch.randn(2, 7))
    assert out.shape == (2, 4) and torch.isfinite(out).all()


def test_compatibility(lib):
    assert lib.are_compatible("arm_7dof", "gripper_4dof")
    assert lib.are_compatible("prime_u_60dof", "arm_7dof")
    # dof=0 自定义形态与任何形态都不兼容
    zero = MorphologyConfig(0, 100.0)
    assert not zero.compatible_with(lib.get("arm_7dof"))


def test_unknown_morphology_guard(lib):
    with pytest.raises(KeyError):
        lib.get("not_a_robot")
    # 非法注册
    with pytest.raises(ValueError):
        lib.register("", "not-a-config")


def test_retargeter_interface_consistent(lib):
    # 60dof -> 4dof 大跨度重定向, 输出有限且在限位内
    rt = ActionRetargeter(lib.get("prime_u_60dof"),
                           lib.get("gripper_4dof"))
    src = torch.randn(3, 60) * 10.0     # 故意超限位
    out = rt.retarget(src)
    assert out.shape == (3, 4)
    assert bool((out >= -1.0).all()) and bool((out <= 1.0).all())
