"""
v3.0.2 Patch 精修: 3.0 线边界加固
======================================
锚点纪律:
    * multimodal H=0 显式守卫;
    * eval_suite 零维度/空分数守卫;
    * mask 维度=0 守卫;
    * alignment_loss NaN 防护 (常数/零方差);
    * 服务端点未训练态 409;
    * 五维权重和不为 1 自动归一化。
"""
import pytest
import torch

from udos import __version__
from udos.future_multimodal import (FutureMultimodalHead, RGBProxyHead,
                                    DepthProxyHead, MaskProxyHead,
                                    CrossModalAlignmentLoss)
from udos.eval_suite import FiveDimensionEvaluator
from udos.server import UDOSService
from udos.persistence import load_predictor

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
CKPT = str(ROOT / "checkpoints" / "predictor_v3.0.0.pt")


def test_version():
    assert __version__ == "5.5.5"


def test_multimodal_h_zero_rejected():
    with pytest.raises(ValueError):
        FutureMultimodalHead(32, horizon=0)
    with pytest.raises(ValueError):
        RGBProxyHead(32, horizon=0)
    with pytest.raises(ValueError):
        DepthProxyHead(32, horizon=0)
    with pytest.raises(ValueError):
        MaskProxyHead(32, horizon=0)


def test_mask_dim_zero_rejected():
    with pytest.raises(ValueError):
        MaskProxyHead(32, horizon=4, mask_dim=0)


def test_all_modalities_off_returns_empty():
    head = FutureMultimodalHead(32, horizon=4, use_rgb=False,
                                use_depth=False, use_mask=False)
    assert head(torch.randn(2, 32)) == {}


def test_eval_suite_zero_dimension_guard():
    # 空分数 => composite 缺维度守卫
    class _Stub:
        DIMENSIONS = FiveDimensionEvaluator.DIMENSIONS
    ev = FiveDimensionEvaluator(load_predictor(CKPT)[0])
    with pytest.raises(ValueError):
        ev.composite_score({})            # 空分数
    with pytest.raises(ValueError):
        ev.composite_score({"visual_spatial_perception": 50.0})  # 缺其余维度


def test_weights_sum_not_one_normalized():
    ev = FiveDimensionEvaluator(load_predictor(CKPT)[0])
    scores = {d: 100.0 for d in FiveDimensionEvaluator.DIMENSIONS}
    w = {d: 2.0 for d in FiveDimensionEvaluator.DIMENSIONS}  # 和=10, 非 1
    c = ev.composite_score(scores, weights=w)
    assert abs(c - 100.0) < 1e-6
    # 权重和<=0 守卫
    with pytest.raises(ValueError):
        ev.composite_score(scores, weights={d: 0.0 for d in
                                            FiveDimensionEvaluator.DIMENSIONS})


def test_alignment_loss_nan_protection():
    loss_fn = CrossModalAlignmentLoss()
    # 常数输入 (零方差) 不产生 NaN
    rgb = torch.ones(4, 4, 8)
    depth = torch.ones(4, 4, 4)
    mask = torch.full((4, 4, 4), 0.5)
    l = loss_fn(rgb, depth, mask)
    assert torch.isfinite(l) and not torch.isnan(l)
    # 错误形状守卫
    with pytest.raises(ValueError):
        loss_fn(torch.randn(4, 8), depth, mask)


def test_service_endpoints_untrained_409():
    svc = UDOSService(preset="small")
    with pytest.raises(Exception):
        svc.future_predict({"window": [[[0.0] * 6] * 6]})
    with pytest.raises(Exception):
        svc.eval_5d()
