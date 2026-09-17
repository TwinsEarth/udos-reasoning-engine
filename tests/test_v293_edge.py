"""
v2.9.3 边界精修单测
=====================
边界覆盖:
    * retargeting DOF=0 (占位形态不可重定向, 显式 ValueError);
    * affordance 零物体 (N_parts=0 显式 ValueError);
    * spatial_relation 单物体 (n_obj<2 显式 ValueError);
    * 频率比极端值 (100x / 0.01x) 重采样不崩溃、有限、端点一致;
    * 服务端点未训练态 -> 409 (ServiceNotReady)。
analogy, not reproduction。
"""
import pytest
import torch

from udos.retargeting import MorphologyConfig, ActionRetargeter
from udos.affordance import AffordanceScorer
from udos.multitask import SpatialRelationHead
from udos.server import UDOSService


def test_retarget_dof_zero_edge():
    zero = MorphologyConfig(0, 100.0)
    assert zero.dof == 0
    with pytest.raises(ValueError):
        ActionRetargeter(zero, MorphologyConfig(4, 100, [[-1, 1]] * 4))
    with pytest.raises(ValueError):
        ActionRetargeter(MorphologyConfig(4, 100, [[-1, 1]] * 4), zero)


def test_affordance_zero_objects_edge():
    scorer = AffordanceScorer()
    with pytest.raises(ValueError):
        scorer.score(torch.zeros(1, 6), torch.zeros(1, 0, 6))


def test_spatial_relation_single_object_edge():
    with pytest.raises(ValueError):
        SpatialRelationHead(32, n_obj=1)
    # 两物体最小合法仍工作
    head = SpatialRelationHead(32, n_obj=2)
    out = head(torch.randn(1, 32))
    assert out.shape == (1, 2, 2, 5)


def test_extreme_frequency_ratio():
    src = MorphologyConfig(6, 60.0, [[-2, 2]] * 6)
    tgt = MorphologyConfig(6, 60.0, [[-2, 2]] * 6)
    rt = ActionRetargeter(src, tgt)
    traj = torch.randn(11, 6)
    # 100x 过采样
    up = rt.resample(traj, 60.0, 6000.0)
    assert torch.isfinite(up).all() and up.shape[0] > 100
    assert torch.allclose(up[0], traj[0], atol=1e-6)
    # 0.01x 欠采样 (至少 2 帧, 端点一致)
    down = rt.resample(traj, 60.0, 0.6)
    assert down.shape[0] >= 2 and torch.isfinite(down).all()
    assert torch.allclose(down[0], traj[0], atol=1e-6)
    assert torch.allclose(down[-1], traj[-1], atol=1e-6)


def test_server_endpoints_untrained_409():
    svc = UDOSService(preset="small")
    with pytest.raises(Exception):   # ServiceNotReady -> 409
        svc.retarget_convert({"actions": [[0.0] * 7], "source": "arm_7dof",
                              "target": "gripper_4dof"})
    with pytest.raises(Exception):
        svc.affordance_score({"state": [0.0] * 6, "objects": [[[0, 0, 0, 0, 0, 0]]]})
