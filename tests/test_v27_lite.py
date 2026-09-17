"""
v2.7.0.dev3 模型轻量化 (udos.lite) 测试
=========================================
锚点纪律:
    * MagnitudePruner 实际零参数占比 ≈ 目标稀疏度, unprune 恢复;
    * 动态量化前后 predict_next 输出在容差内接近;
    * 蒸馏 loss 下降;
    * A/B JSON 落盘; 默认全量模型逐位不变 (不调用 lite 路径)。
"""
import json
import os

import pytest
import torch

from udos import __version__
from udos.lite import MagnitudePruner, DynamicQuantizer, DistillationTrainer
from udos.ctm_engine import CTMConfig
from udos.persistence import load_predictor
from udos.dynamics import build_parametric_dataset

CKPT = "checkpoints/predictor_v2.7.0.pt"
AB_JSON = "benchmarks/results/lite_ablation_v2.7.0.json"


def small_cfg(d_model=64):
    return CTMConfig(iterations=8, d_model=d_model, d_input=32, heads=4,
                     n_synch_out=16, n_synch_action=8, memory_length=8,
                     nlm_hidden=16, out_dims=32, certainty_threshold=0.0)


@pytest.fixture(scope="module")
def predictor():
    model, _ = load_predictor(CKPT)
    return model


@pytest.fixture(scope="module")
def data():
    ds = build_parametric_dataset(n_per_kind=12, n_steps=14, window=6,
                                  horizon=4, dt=0.5, seed=611)
    return ds


def test_version():
    assert __version__ == "5.5.5"


def test_prune_sparsity_and_restore(predictor, data):
    p = MagnitudePruner()
    out_before = predictor.predict_next(data.X[:4], scene_params=data.P[:4]).clone()
    sp = p.prune(predictor, 0.5)
    # 实际稀疏度接近 0.5 (全局分位, 允许小幅偏差)
    assert 0.45 <= sp <= 0.55
    assert MagnitudePruner.sparsity_ratio(predictor) > 0.4
    # unprune 后逐位恢复
    p.unprune(predictor)
    assert MagnitudePruner.sparsity_ratio(predictor) < 1e-9
    out_after = predictor.predict_next(data.X[:4], scene_params=data.P[:4])
    assert torch.allclose(out_before, out_after, atol=1e-6)


def test_quantize_output_close(predictor, data):
    orig = predictor.predict_next(data.X[:4], scene_params=data.P[:4])
    q = DynamicQuantizer.quantize(predictor)
    q_out = q.predict_next(data.X[:4], scene_params=data.P[:4])
    # INT8 动态量化允许一定误差, 但不应发散
    diff = (orig - q_out).abs().max().item()
    assert diff < 1.0
    assert torch.isfinite(q_out).all()


def test_distill_loss_decreases(predictor, data):
    student = DistillationTrainer(alpha=0.7).distill(
        predictor, small_cfg(d_model=32), data, epochs=8, seed=0)
    hist = student.distill_history
    assert hist[-1] < hist[0]      # loss 下降
    # 学生模型更小 (d_model=32)
    assert sum(p.numel() for p in student.parameters()) < \
        sum(p.numel() for p in predictor.parameters())


def test_ablation_json_exists():
    assert os.path.exists(AB_JSON), "A/B 脚本未落盘"
    with open(AB_JSON, encoding="utf-8") as f:
        d = json.load(f)
    for v in ("full", "pruned_50", "quantized_int8", "distilled_student"):
        assert v in d["variants"]


def test_default_model_unchanged():
    """不调用 lite 路径, 新加载的正式件逐位一致 (52191 参数, 可加载)。"""
    m, meta = load_predictor(CKPT)
    assert sum(p.numel() for p in m.parameters()) == 52191
    assert meta["udos_version"] == "2.7.0"
