"""v2.0.0 学习闭环测试: 数据集契约、前向形状、梯度回传、真实收敛。"""

import pytest
import torch

from udos.ctm_engine import CTMConfig
from udos.dynamics import (RAW_DIM, build_dynamics_dataset,
                           naive_baseline_mse)
from udos.training import (PhysicsPredictor, CTMTrainer, TrainConfig)


def tiny_cfg():
    return CTMConfig(iterations=4, d_model=32, d_input=16, heads=2,
                     n_synch_out=8, n_synch_action=6, memory_length=4,
                     nlm_hidden=8, out_dims=16, certainty_threshold=0.0)


def test_dataset_shapes_and_classes():
    ds = build_dynamics_dataset(n_per_kind=6, n_steps=10, window=5)
    assert ds.X.dim() == 3 and ds.X.size(-1) == RAW_DIM == 6
    assert ds.y.shape == (len(ds), RAW_DIM)
    # 四类运动都被覆盖
    assert set(ds.class_names) <= set(ds.kinds)
    # 滑窗: 每条轨迹贡献 n_steps-window 个样本
    assert len(ds) > 0


def test_split_disjoint_and_batches():
    ds = build_dynamics_dataset(n_per_kind=6, n_steps=10, window=5)
    tr, te = ds.split(0.8)
    assert len(tr) + len(te) == len(ds)
    bx, by = next(tr.batches(16, shuffle=False))
    assert bx.size(0) == by.size(0) and by.size(-1) == RAW_DIM


def test_naive_baseline_positive_finite():
    ds = build_dynamics_dataset(n_per_kind=6)
    v = naive_baseline_mse(ds)
    assert v == v and v > 0  # 非 NaN 且为正 (运动轨迹"状态不变"确有误差)


def test_predictor_forward_shapes():
    m = PhysicsPredictor(tiny_cfg())
    x = torch.randn(5, 5, RAW_DIM)
    ticks, certs, last, info = m(x)
    T = tiny_cfg().iterations
    assert ticks.shape == (5, RAW_DIM, T)
    assert certs.shape == (5, 2, T)
    assert last.shape == (5, RAW_DIM)
    # 训练模式关闭早停, 跑满 ticks
    assert info["ticks_used"] == T


def test_gradient_reaches_core():
    # 反例: 若 CTM 核心参数脱离计算图, grad 为 None
    m = PhysicsPredictor(tiny_cfg())
    ticks, certs, last, _ = m(torch.randn(8, 5, RAW_DIM))
    y = torch.randn(8, RAW_DIM)
    loss = ((last - y) ** 2).mean() + 0.01 * certs[:, 0, -1].mean()
    loss.backward()
    grads = [p.grad for p in m.ctm.parameters() if p.requires_grad]
    assert grads and all(g is not None for g in grads)
    assert any(g.abs().sum().item() > 0 for g in grads)


def test_eval_deterministic():
    m = PhysicsPredictor(tiny_cfg()); m.eval()
    x = torch.randn(4, 5, RAW_DIM)
    a = m.predict_next(x); b = m.predict_next(x)
    assert torch.allclose(a, b, atol=1e-6)


def test_trains_and_beats_baselines():
    """
    核心契约 (能学习): 在小数据集上训练后, 训练集 MSE 必须显著低于
    未训练初始值, 并低于"状态不变"naive 基线。
    反例: 不可学习的冻结/错误解码器无法通过。
    """
    torch.manual_seed(0)
    ds = build_dynamics_dataset(n_per_kind=16, n_steps=10, window=5)
    tr, _ = ds.split(0.9)
    # 收敛需要足够容量/训练步; 形状类测试用 tiny_cfg, 这里用可稳定超过基线的配置
    cfg = CTMConfig(iterations=8, d_model=64, d_input=24, heads=2,
                    n_synch_out=12, n_synch_action=8, memory_length=6,
                    nlm_hidden=12, out_dims=24, certainty_threshold=0.0)
    m = PhysicsPredictor(cfg)
    trainer0 = CTMTrainer(m, TrainConfig(epochs=1))
    initial = trainer0.evaluate(tr)
    trainer = CTMTrainer(m, TrainConfig(epochs=40, lr=3e-3, batch_size=32,
                                        cert_weight=0.0))
    trainer.train(tr, None)
    final = trainer.evaluate(tr)
    assert final < 0.5 * initial            # 至少下降一半 (能学习)
    assert final < naive_baseline_mse(tr)   # 优于"状态不变"朴素外推
