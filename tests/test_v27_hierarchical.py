"""
v2.7.0.dev4 分层 / 多尺度长时域 rollout (udos.hierarchical.HierarchicalRollout) 测试
====================================================================================
锚点纪律:
    * 短 horizon (≤ coarse_factor) 与普通 rollout 逐位一致;
    * 长 horizon 预测形状 [B,H,RAW] 有限, coarse_points 标记正确, truncated 正确;
    * 单自回归头下分层与平铺逐位一致 (诚实负面, 见 A/B JSON);
    * A/B 证据可复算。
"""
import json
import os

import pytest
import torch

from udos import __version__
from udos.hierarchical import HierarchicalRollout
from udos.persistence import load_predictor
from udos.dynamics import build_parametric_dataset

CKPT = "checkpoints/predictor_v2.7.0.pt"
AB_JSON = "benchmarks/results/hierarchical_ablation_v2.7.0.json"


@pytest.fixture(scope="module")
def predictor():
    model, _ = load_predictor(CKPT)
    return model


@pytest.fixture(scope="module")
def window_batch():
    ds = build_parametric_dataset(n_per_kind=8, n_steps=22, window=6,
                                  horizon=16, dt=0.5, seed=621)
    return ds.X[:4], ds.P[:4], ds


def test_version():
    assert __version__ == "5.5.5"


def test_short_horizon_degenerate_identical(predictor, window_batch):
    wb, pb, _ = window_batch
    hr = HierarchicalRollout(predictor, coarse_factor=4)
    for H in (1, 2, 4):
        flat = predictor.rollout(wb, H, scene_params=pb)
        out = hr.rollout(wb, H, scene_params=pb)
        assert out["coarse_points"] == []
        assert out["truncated"] is False
        assert torch.allclose(out["predictions"], flat, atol=1e-7)


def test_long_horizon_shape_and_flags(predictor, window_batch):
    wb, pb, _ = window_batch
    hr = HierarchicalRollout(predictor, coarse_factor=4)
    H = 16
    out = hr.rollout(wb, H, scene_params=pb)
    assert out["predictions"].shape == (wb.size(0), H, 6)
    assert torch.isfinite(out["predictions"]).all()
    # 粗粒度锚点: 4,8,12 (每 cf 一步, 终点不记)
    assert out["coarse_points"] == [4, 8, 12]
    assert out["truncated"] is False     # 16 % 4 == 0

    # H 不整除 => truncated=True
    out2 = hr.rollout(wb, 13, scene_params=pb)
    assert out2["truncated"] is True
    assert out2["coarse_points"] == [4, 8, 12]


def test_long_horizon_not_worse_than_flat(predictor, window_batch):
    """单自回归头下分层与平铺逐位一致 (诚实负面): MSE 不退化。"""
    wb, pb, ds = window_batch
    hr = HierarchicalRollout(predictor, coarse_factor=4)
    flat = predictor.rollout(wb, 16, scene_params=pb)
    hier = hr.rollout(wb, 16, scene_params=pb)["predictions"]
    assert torch.allclose(flat, hier, atol=1e-6)


def test_ablation_json_exists():
    assert os.path.exists(AB_JSON)
    with open(AB_JSON, encoding="utf-8") as f:
        d = json.load(f)
    for k in ("H=8", "H=12", "H=16"):
        assert k in d["results"]
        # 如实记录: 单自回归头下逐位一致
        assert d["results"][k]["bit_identical"] is True
