"""
v3.3.2 节点59: 3.3 线边界精修
====================================================================
覆盖:
    * MoE 专家数=1 (退化路由);
    * 蒸馏温度=0 (守卫);
    * 剪枝稀疏度=1.0 (守卫);
    * 内存优化 OOM/空态防护;
    * 鲁棒性极端噪声 (分数仍在 [0,100]);
    * 服务未训练态 409。
analogy, not reproduction。
"""
import pytest
import torch

from udos import __version__
from udos.moe import LightweightMoE
from udos.lite import DistillationTrainerV2, StructuredPrunerV2
from udos.training import (PhysicsPredictor, ActivationFp16Cache,
                           TrainConfig)
from udos.ctm_engine import CTMConfig
from udos.robustness import RobustnessEvaluator
from udos.dynamics import build_parametric_dataset

torch.set_num_threads(2)


def small_model():
    cfg = CTMConfig(iterations=3, d_model=32, d_input=16, heads=4,
                    n_synch_out=8, n_synch_action=4, memory_length=6,
                    nlm_hidden=8, out_dims=8, certainty_threshold=0.0)
    cfg.scene_dim = 32
    torch.manual_seed(0)
    return PhysicsPredictor(cfg, scene_param_dim=4)


def test_moe_single_expert():
    moe = LightweightMoE(8, 4, num_experts=1, top_k=1)
    out = moe(torch.randn(3, 8))
    assert out.shape == (3, 4) and torch.isfinite(out).all()


def test_distill_zero_temperature_guarded():
    try:
        DistillationTrainerV2(temperature=0.0)
        assert False
    except ValueError:
        pass


def test_prune_sparsity_one_guarded():
    try:
        StructuredPrunerV2(prune_ratio=1.0)
        assert False
    except ValueError:
        pass
    try:
        StructuredPrunerV2(prune_ratio=-0.1)
        assert False
    except ValueError:
        pass


def test_memory_oom_guard():
    """激活缓存空态/清空防护: 未 encode 时字节为 0, clear 后为 0。"""
    cache = ActivationFp16Cache()
    assert cache.memory_bytes_est() == 0
    cache.clear()   # 空缓存 clear 不报错
    assert cache.memory_bytes_est() == 0


def test_extreme_noise_robustness_clamped():
    model = small_model()
    ds = build_parametric_dataset(n_per_kind=6, n_steps=12, window=6,
                                 horizon=2, dt=0.5, seed=31)
    rep = RobustnessEvaluator(model, noise_sigmas=(0.0, 10.0),
                              seed=0).evaluate(ds)
    assert 0.0 <= rep["robustness_score"] <= 100.0
    assert torch.isfinite(torch.tensor(rep["noise"]["grid"]["test_sigma=10.0"]))


def test_service_untrained_409():
    from udos.server import UDOSService, ServiceNotReady
    svc = UDOSService(preset="small")     # 未加载 checkpoint
    try:
        svc.loop_step({"window": [[0.0] * 6] * 6, "horizon": 1})
        assert False
    except ServiceNotReady:
        pass


def test_version():
    assert __version__ == "5.5.5"
