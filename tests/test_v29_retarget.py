"""
v2.9.0 形态无关动作重定向 (retargeting) 单测
==============================================
锚点纪律:
    * MorphologyConfig 校验 (DOF/频率/限位) 正确;
    * DOF 映射端点对齐、形状正确、恒等维逐位一致;
    * 重定向后动作严格落在目标关节限位内 (clamp);
    * 空动作 / 维度不符 / 非有限值显式 ValueError;
    * MorphologyConfig / ActionRetargeter 序列化往返一致 (save/load 兼容)。
analogy, not reproduction: 不同维动作向量代理不同机器人形态。
"""
import pytest
import torch

from udos.retargeting import (MorphologyConfig, ActionRetargeter,
                              map_dof)


def _arm(dof=7, freq=100.0):
    lim = [[-1.0, 1.0]] * dof
    return MorphologyConfig(dof=dof, control_freq=freq, joint_limits=lim,
                            name="arm", kinematics={"links": dof})


def test_morphology_config_validation():
    cfg = _arm(7, 100.0)
    assert cfg.dof == 7 and cfg.control_freq == 100.0
    assert cfg.low.shape == (7,) and cfg.high.shape == (7,)
    # dof < 0
    with pytest.raises(ValueError):
        MorphologyConfig(-1, 100.0)
    # freq <= 0
    with pytest.raises(ValueError):
        MorphologyConfig(7, 0.0)
    # lo > hi
    with pytest.raises(ValueError):
        MorphologyConfig(2, 100.0, [[1.0, -1.0]])
    # limits 长度不符
    with pytest.raises(ValueError):
        MorphologyConfig(3, 100.0, [[-1, 1], [-1, 1]])


def test_dof_mapping_identity_and_shape():
    x = torch.randn(5, 6)
    # 恒等维逐位一致
    y = map_dof(x, 6, 6)
    assert torch.equal(x, y)
    # 6 -> 9: 形状 + 端点对齐
    z = map_dof(x, 6, 9)
    assert z.shape == (5, 9)
    # 端点: target 第 0/8 列对应 source 第 0/5 列
    assert torch.equal(z[:, 0], x[:, 0])
    assert torch.equal(z[:, -1], x[:, -1])
    # 单目标关节退化
    s = map_dof(x, 6, 1)
    assert s.shape == (5, 1) and torch.equal(s[:, 0], x[:, -1])


def test_retarget_dof_mapping_and_clamp():
    src = _arm(6, 60.0)
    tgt = MorphologyConfig(4, 120.0, [[-0.5, 0.5]] * 4, name="gripper")
    rt = ActionRetargeter(src, tgt)
    # 构造一个明显超限位的动作
    act = torch.tensor([[10.0, -10.0, 0.0, 0.3, -0.2, 0.1]])
    out = rt.retarget(act)
    assert out.shape == (1, 4)
    # 严格落在限位内
    assert bool((out >= -0.5).all()) and bool((out <= 0.5).all())
    # 端点映射保持: target 首关节 <- source 首关节
    assert torch.allclose(out[0, 0],
                          torch.clamp(torch.tensor(10.0), -0.5, 0.5))


def test_retarget_empty_and_dim_guard():
    src, tgt = _arm(6), _arm(4)
    rt = ActionRetargeter(src, tgt)
    with pytest.raises(ValueError):
        rt.retarget(torch.empty(0, 6))          # 空动作
    with pytest.raises(ValueError):
        rt.retarget(torch.randn(1, 5))          # 末维不符
    with pytest.raises(ValueError):
        rt.retarget(torch.tensor([[float("nan")] * 6]))  # 非有限


def test_dof_zero_not_retargetable():
    zero = MorphologyConfig(0, 100.0)
    assert zero.dof == 0 and zero.low.shape == (0,)
    with pytest.raises(ValueError):
        ActionRetargeter(zero, _arm(4))
    # 兼容性: dof=0 与任何形态都不兼容
    assert not zero.compatible_with(_arm(4))
    assert _arm(6).compatible_with(_arm(4))


def test_serialization_roundtrip():
    cfg = _arm(7, 100.0)
    d = cfg.to_dict()
    cfg2 = MorphologyConfig.from_dict(d)
    assert cfg2.dof == cfg.dof and cfg2.control_freq == cfg.control_freq
    assert torch.equal(cfg2.low, cfg.low) and torch.equal(cfg2.high, cfg.high)
    # retargeter 往返
    rt = ActionRetargeter(_arm(6, 60.0), _arm(4, 120.0))
    act = torch.randn(3, 6)
    ref = rt.retarget(act)
    rt2 = ActionRetargeter.from_config_dict(rt.config_dict())
    assert torch.allclose(rt2.retarget(act), ref)
