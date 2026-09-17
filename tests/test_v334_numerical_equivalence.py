"""v3.3.4 hardening: 数值逐位等价锚点测试。

日志改造是纯附加行为, 不得改变任何计算结果。本测试在加日志后对核心推理路径
跑两次相同输入, 断言输出逐位 (bit-exact) 等价; 并验证配置/序列化路径确定性。

覆盖: 单次 predict、批量 predict、internalize+reason、calibration fit 确定性。
"""

from __future__ import annotations

import json

import pytest
import torch

from udos.ctm_engine import CTMConfig
from udos.dynamics import RAW_DIM, SCENE_PARAM_DIM, build_parametric_dataset
from udos.training import PhysicsPredictor
from udos.server import UDOSService
from udos.calibration import fit_predictor_calibration
from udos.pce_format import PCEParser


def _tiny_predictor():
    cfg = CTMConfig(iterations=4, d_model=32, d_input=16, heads=2,
                    n_synch_out=8, n_synch_action=6, memory_length=4,
                    nlm_hidden=8, out_dims=16, certainty_threshold=0.0)
    m = PhysicsPredictor(cfg, scene_param_dim=SCENE_PARAM_DIM)
    m.eval()
    return m


def test_predict_bitexact_twice():
    """同一输入连续两次 predict_next 必须逐位相同。"""
    torch.manual_seed(0)
    m = _tiny_predictor()
    x = torch.randn(4, 6, RAW_DIM)
    with torch.no_grad():
        a = m.predict_next(x)
        b = m.predict_next(x)
    assert torch.equal(a, b), "predict_next 两次输出必须 bit-exact"


def test_predict_batch_matches_single():
    """批量入口与逐笔入口数值一致 (容差内, 与既有契约一致)。"""
    torch.manual_seed(0)
    m = _tiny_predictor()
    x = torch.randn(3, 6, RAW_DIM)
    with torch.no_grad():
        batched = m.predict_next(x)
        singles = torch.stack([m.predict_next(x[i:i + 1])[0]
                               for i in range(x.size(0))])
    assert torch.allclose(batched, singles, atol=1e-6)


def test_reason_bitexact_twice():
    """internalize + reason 两次推理预测向量逐位相同。"""
    torch.manual_seed(0)
    svc = UDOSService(preset="small")
    from demos.scene_factory import build_factory_scene
    scene = json.loads(PCEParser.dumps(build_factory_scene(n_steps=6)))
    svc.internalize({"scene": scene})
    with torch.no_grad():
        r1 = svc.reason({"scene": scene, "query": "probe", "horizon": 2})
        r2 = svc.reason({"scene": scene, "query": "probe", "horizon": 2})
    assert r1["prediction_vector"] == r2["prediction_vector"]
    assert r1["certainty_trajectory"] == r2["certainty_trajectory"]


def test_calibration_fit_deterministic():
    """同一数据两次 fit_predictor_calibration 报告逐位一致。"""
    torch.manual_seed(0)
    m = _tiny_predictor()
    ds = build_parametric_dataset(n_per_kind=8, n_steps=14, window=6,
                                  horizon=2, dt=0.5, seed=123)
    with torch.no_grad():
        c1, rep1, hw1 = fit_predictor_calibration(m, ds)
        c2, rep2, hw2 = fit_predictor_calibration(m, ds)
    assert rep1 == rep2, "校准报告应确定性"
    for a, b in zip(hw1, hw2):
        assert torch.equal(a, b)
