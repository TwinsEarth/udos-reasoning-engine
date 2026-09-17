"""
v2.9.0.dev5 空间关系头 (SpatialRelationHead) 单测
==================================================
锚点纪律:
    * 关系矩阵 [B,N_obj,N_obj,5] 形状有限;
    * 对称性约束: right(i,j) == left(j,i) (上下同理);
    * 接触/左右关系在可控坐标下正确;
    * 头关闭/移除时旧预测路径不变; 与 multitask 框架兼容 (register/remove)。
analogy, not reproduction: 用 latent 投影坐标的几何关系代理物体间空间关系。
"""
from pathlib import Path

import pytest
import torch

from udos.multitask import MultiTaskHead, SpatialRelationHead
from udos.persistence import load_predictor
from udos.dynamics import build_parametric_dataset

ROOT = Path(__file__).resolve().parents[1]
CKPT = str(ROOT / "checkpoints" / "predictor_v2.9.0.pt")


@pytest.fixture(scope="module")
def predictor():
    m, _ = load_predictor(CKPT)
    return m


def test_relation_matrix_shape(predictor):
    mth = MultiTaskHead(predictor, latent_dim=32, enable=True)
    head = SpatialRelationHead(32, n_obj=4)
    mth.register_head("spatial_rel", head)
    ds = build_parametric_dataset(n_per_kind=8, n_steps=14, window=6,
                                  horizon=4, dt=0.5, seed=505)
    out = mth.forward(ds.X[:1], scene_params=ds.P[:1])
    rel = out["spatial_rel"]
    assert rel.shape == (1, 4, 4, 5)
    assert torch.isfinite(rel).all()


def test_symmetry_constraint(predictor):
    head = SpatialRelationHead(32, n_obj=4)
    latent = torch.randn(2, 32)
    rel = head(latent)   # [2,4,4,5]
    right = rel[..., 0]
    left = rel[..., 1]
    up = rel[..., 2]
    down = rel[..., 3]
    # right(i,j) == left(j,i)
    assert torch.equal(right, left.transpose(-1, -2))
    assert torch.equal(up, down.transpose(-1, -2))


def test_relations_correct_on_controlled_coords(predictor):
    head = SpatialRelationHead(32, n_obj=3, contact_thr=0.5)
    # 锁死线性投影: W=0, bias = 固定 3 物体坐标 [o0=(0,0), o1=(1,0), o2=(0,1)]
    with torch.no_grad():
        head.coord_proj.weight.zero_()
        head.coord_proj.bias.copy_(torch.tensor(
            [0.0, 0.0, 1.0, 0.0, 0.0, 1.0]))
    rel = head(torch.randn(1, 32))[0]   # [3,3,5]
    right, left, up, down, contact = rel.unbind(-1)
    # o1 在 o0 右侧
    assert right[0, 1] == 1.0 and left[0, 1] == 0.0
    assert left[1, 0] == 1.0 and right[1, 0] == 0.0
    # o2 在 o0 上方
    assert up[0, 2] == 1.0 and down[2, 0] == 1.0
    # 距离均=1 > contact_thr => 无接触 (对角自身 contact=1)
    assert contact[0, 1] == 0.0 and contact[1, 0] == 0.0
    assert contact[0, 0] == 1.0


def test_head_toggle_old_path_unchanged(predictor):
    x = torch.randn(1, 6, 6)
    sp = torch.randn(1, 4)
    before = predictor.predict_next(x, scene_params=sp)
    mth = MultiTaskHead(predictor, latent_dim=32, enable=True)
    mth.register_head("spatial_rel", SpatialRelationHead(32, n_obj=4))
    mth.forward(x, scene_params=sp)
    mth.remove_head("spatial_rel")
    after = predictor.predict_next(x, scene_params=sp)
    assert torch.equal(before, after)
    # 默认 enable=False => 不跑任何头
    mth_off = MultiTaskHead(predictor, latent_dim=32, enable=False)
    assert mth_off.forward(x, scene_params=sp) == {}


def test_n_obj_guard():
    with pytest.raises(ValueError):
        SpatialRelationHead(32, n_obj=1)   # 单物体关系矩阵无定义
