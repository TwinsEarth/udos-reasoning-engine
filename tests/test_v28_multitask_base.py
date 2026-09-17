"""
v2.8.0.dev5 MultiTaskHead 骨架单元测试
==========================================
锚点纪律:
    * 头注册/查询/列表/删除;
    * 空头/未注册查询守卫;
    * encode(window, scene_params) -> latent[B, latent_dim] 有限;
    * 默认 enable=False 时 forward 返回 {} 且主模型 predict_next 逐位不变;
    * config_dict / load_config 往返一致 (save/load 头配置)。
"""
from pathlib import Path

import pytest
import torch

from udos import __version__
from udos.multitask import MultiTaskHead
from udos.persistence import load_predictor
from udos.dynamics import build_parametric_dataset

ROOT = Path(__file__).resolve().parents[1]
CKPT = str(ROOT / "checkpoints" / "predictor_v2.8.0.pt")


@pytest.fixture(scope="module")
def predictor():
    model, _ = load_predictor(CKPT)
    return model


@pytest.fixture(scope="module")
def window_batch():
    ds = build_parametric_dataset(n_per_kind=8, n_steps=14, window=6,
                                  horizon=4, dt=0.5, seed=2020)
    return ds.X[:1], ds.P[:1]


def test_version():
    assert __version__ == "5.5.5"


def test_head_register_list_get(predictor, window_batch):
    wb, pb = window_batch
    mth = MultiTaskHead(predictor, latent_dim=32)
    dummy = lambda latent: latent.sum(dim=-1)
    mth.register_head("dummy", dummy)
    assert "dummy" in mth.list_heads()
    assert mth.get_head("dummy") is dummy
    mth.remove_head("dummy")
    assert "dummy" not in mth.list_heads()


def test_unknown_head_guard(predictor):
    mth = MultiTaskHead(predictor)
    with pytest.raises(KeyError):
        mth.get_head("nope")
    with pytest.raises(ValueError):
        mth.register_head("", lambda x: x)
    with pytest.raises(ValueError):
        mth.register_head("bad", "not_callable")


def test_encode_shape_finite(predictor, window_batch):
    wb, pb = window_batch
    mth = MultiTaskHead(predictor, latent_dim=32)
    z = mth.encode(wb, scene_params=pb)
    assert z.shape == (1, 32)
    assert torch.isfinite(z).all()
    # 2D 输入自动升维
    z2 = mth.encode(wb[0], scene_params=pb)
    assert z2.shape == (1, 32)


def test_latent_dim_configurable(predictor, window_batch):
    wb, pb = window_batch
    for ld in [8, 16, 64]:
        mth = MultiTaskHead(predictor, latent_dim=ld)
        assert mth.encode(wb, scene_params=pb).shape == (1, ld)


def test_default_disabled_path_unchanged(predictor, window_batch):
    wb, pb = window_batch
    mth = MultiTaskHead(predictor, latent_dim=32)
    # 默认 enable=False => forward 空
    assert mth.forward(wb, scene_params=pb) == {}
    # 主模型 predict_next 逐位不变
    before = predictor.predict_next(wb, scene_params=pb)
    mth.encode(wb, scene_params=pb)
    after = predictor.predict_next(wb, scene_params=pb)
    assert torch.equal(before, after)


def test_enabled_routes_to_registered_head(predictor, window_batch):
    wb, pb = window_batch
    mth = MultiTaskHead(predictor, latent_dim=32, enable=True)
    mth.register_head("half", lambda latent: latent * 0.5)
    out = mth.forward(wb, scene_params=pb)
    assert set(out.keys()) == {"half"}
    assert torch.allclose(out["half"], mth.encode(wb, scene_params=pb) * 0.5)


def test_config_save_load(predictor, window_batch):
    wb, pb = window_batch
    mth = MultiTaskHead(predictor, latent_dim=16, enable=True)
    mth.register_head("a", lambda x: x)
    cfg = mth.config_dict()
    # 新实例 load_config 恢复元数据
    mth2 = MultiTaskHead(predictor, latent_dim=32)
    mth2.load_config(cfg)
    assert mth2.latent_dim == 16
    assert mth2.enable is True
    assert mth2.list_heads() == []   # 头对象需调用方重新 register (只恢复名/配置)
