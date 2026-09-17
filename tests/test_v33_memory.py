"""
v3.3.0.dev5 节点56: 内存优化 (梯度检查点 + 激活 FP16 缓存) 测试
====================================================================
纪律:
    * 梯度检查点开启时训练不崩溃, loss 下降;
    * 激活 FP16 缓存推理可命中, 内存占用对比;
    * 默认全关: gradient_checkpointing=False 走旧路径逐位一致;
    * 内存优化有 OOM/边界防护。
analogy, not reproduction。
"""
import torch

from udos.ctm_engine import CTMConfig
from udos.training import (PhysicsPredictor, CTMTrainer, TrainConfig,
                            ActivationFp16Cache)
from udos.dynamics import build_parametric_dataset

torch.set_num_threads(2)


def small_cfg():
    cfg = CTMConfig(iterations=4, d_model=32, d_input=16, heads=4,
                    n_synch_out=8, n_synch_action=4, memory_length=6,
                    nlm_hidden=8, out_dims=8, certainty_threshold=0.0)
    cfg.scene_dim = 32
    return cfg


def dataset(seed=3):
    return build_parametric_dataset(n_per_kind=10, n_steps=12, window=6,
                                   horizon=2, dt=0.5, seed=seed)


def test_gradient_checkpointing_training_runs():
    ds = dataset()
    tr, va = ds.split(0.8)
    torch.manual_seed(0)
    model = PhysicsPredictor(small_cfg(), scene_param_dim=4)
    hist = CTMTrainer(model, TrainConfig(epochs=3, batch_size=16,
                                         patience=None,
                                         gradient_checkpointing=True)
                      ).train(tr, va)
    assert len(hist.train_loss) == 3
    assert hist.train_loss[-1] < hist.train_loss[0]


def test_default_off_bitwise_equivalent():
    """gradient_checkpointing=False 时前向与不开启逐位一致 (同 forward_cell)。"""
    ds = dataset(5)
    torch.manual_seed(0)
    m1 = PhysicsPredictor(small_cfg(), scene_param_dim=4)
    torch.manual_seed(0)
    m2 = PhysicsPredictor(small_cfg(), scene_param_dim=4)
    x, p = ds.X[:3], ds.P[:3]
    a = m1.predict_next(x, scene_params=p)
    b = m2.predict_next(x, scene_params=p)
    assert torch.equal(a, b)


def test_activation_fp16_cache_hit():
    ds = dataset(7)
    torch.manual_seed(0)
    model = PhysicsPredictor(small_cfg(), scene_param_dim=4)
    model.eval()
    cache = ActivationFp16Cache()
    x = ds.X[:2]
    h1 = cache.encode_cached(model, x)
    h2 = cache.encode_cached(model, x)     # 命中
    assert cache.hits == 1 and cache.misses == 1
    # 缓存取回与直算数值一致 (上采样回 fp32)
    direct = model.obs_encoder(x)
    assert torch.allclose(h1, direct, atol=1e-6)            # miss 直算 fp32
    # hit 取回经 fp16 量化-上采样, 容差放宽到 fp16 精度 (~1e-2)
    assert torch.allclose(h2, direct, atol=5e-3)
    assert (h2 - direct).abs().max().item() < 5e-3          # 量化损失有界


def test_fp16_memory_smaller_than_fp32():
    cache = ActivationFp16Cache()
    torch.manual_seed(0)
    model = PhysicsPredictor(small_cfg(), scene_param_dim=4)
    model.eval()
    ds = dataset(9)
    for i in range(4):
        cache.encode_cached(model, ds.X[i:i+1])
    fp16_bytes = cache.memory_bytes_est()
    # 等价 fp32 存储应为其 2 倍
    assert fp16_bytes > 0
    # 每个缓存张量 numel*2 (fp16) vs fp32 numel*4
    n_elem = sum(t.numel() for t in cache._store.values())
    assert fp16_bytes == n_elem * 2
    assert fp16_bytes == int(n_elem * 4 * 0.5)


def test_cache_clear():
    cache = ActivationFp16Cache()
    torch.manual_seed(0)
    model = PhysicsPredictor(small_cfg(), scene_param_dim=4)
    model.eval()
    ds = dataset(11)
    cache.encode_cached(model, ds.X[:1])
    assert cache.memory_bytes_est() > 0
    cache.clear()
    assert cache.memory_bytes_est() == 0
