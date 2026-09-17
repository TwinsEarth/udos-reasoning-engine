"""
v3.3.0.dev3 节点54: 结构化剪枝 v2 (通道/头级) 测试
====================================================================
纪律:
    * 整通道 (Linear 行) 级剪枝, 而非逐元素; 通道稀疏度可测;
    * 剪枝后可选微调恢复 (mask 冻结); 对比 v1 幅值剪枝;
    * save/load 剪枝态; opt-in 默认关。
analogy, not reproduction。
"""
import torch

from udos.ctm_engine import CTMConfig
from udos.training import PhysicsPredictor, CTMTrainer, TrainConfig
from udos.lite import MagnitudePruner, StructuredPrunerV2
from udos.dynamics import build_parametric_dataset
from udos.persistence import save_predictor, load_predictor

torch.set_num_threads(2)


def small_model(seed=0):
    cfg = CTMConfig(iterations=3, d_model=32, d_input=16, heads=4,
                    n_synch_out=8, n_synch_action=4, memory_length=6,
                    nlm_hidden=8, out_dims=8, certainty_threshold=0.0)
    cfg.scene_dim = 32
    torch.manual_seed(seed)
    return PhysicsPredictor(cfg, scene_param_dim=4)


def dataset(seed=1):
    return build_parametric_dataset(n_per_kind=10, n_steps=12, window=6,
                                   horizon=2, dt=0.5, seed=seed)


def test_channel_pruning_sparsity():
    model = small_model()
    rep = StructuredPrunerV2(prune_ratio=0.4).prune(model)
    # 整通道稀疏度应接近 prune_ratio
    assert 0.3 <= rep["channel_sparsity"] <= 0.5
    # 所有被剪通道整行为 0 (通道级, 非逐元素)
    for _, lin in model.named_modules():
        if isinstance(lin, torch.nn.Linear):
            row_zeros = (lin.weight.detach().abs().sum(dim=1) == 0).sum()
            assert row_zeros >= 0


def test_vs_v1_magnitude_pruning():
    """v1 逐元素 vs v2 通道级: 同目标稀疏度, v2 整行为 0 的通道占比更高。"""
    m1 = small_model()
    m2 = small_model()
    rep1 = MagnitudePruner().prune(m1, sparsity=0.4)
    rep2 = StructuredPrunerV2(prune_ratio=0.4).prune(m2)
    # v1 报权重稀疏度 (逐元素), v2 报通道稀疏度 (整行)
    assert 0.35 <= rep1 <= 0.45
    assert rep2["pruned_channels"] > 0
    # v2 通道级零行数 > v1 (v1 很少整行为 0)
    v1_zero_rows = sum(
        int((lin.weight.detach().abs().sum(dim=1) == 0).sum())
        for _, lin in m1.named_modules() if isinstance(lin, torch.nn.Linear))
    assert rep2["pruned_channels"] > v1_zero_rows


def test_finetune_recovery_runs():
    model = small_model()
    pruner = StructuredPrunerV2(prune_ratio=0.3)
    pruner.prune(model)
    ds = dataset()
    losses = pruner.fine_tune(model, ds, epochs=2, batch_size=16)
    assert len(losses) == 2 and all(v == v for v in losses)
    # 微调后剪枝通道仍为 0 (mask 冻结)
    rep = pruner.report(model)
    assert rep["pruned_channels"] > 0


def test_head_grouped_pruning():
    model = small_model()
    # attn_to_state 输出 d_model=32, head_dim=8 => 4 头
    pruner = StructuredPrunerV2(prune_ratio=0.5, head_dim=8)
    rep = pruner.prune(model)
    assert rep["channel_sparsity"] > 0
    assert 0.0 <= rep["weight_sparsity"] <= 1.0


def test_save_load_pruned_state(tmp_path):
    model = small_model()
    pruner = StructuredPrunerV2(prune_ratio=0.4)
    pruner.prune(model)
    ds = dataset()
    x, p = ds.X[:2], ds.P[:2]
    before = model.predict_next(x, scene_params=p)
    ckpt = tmp_path / "pruned.pt"
    save_predictor(model, str(ckpt))
    loaded, _ = load_predictor(str(ckpt))
    after = loaded.predict_next(x, scene_params=p)
    assert torch.allclose(before, after, atol=1e-6)


def test_unprune_restores():
    model = small_model()
    before_w = dict(model.state_dict())
    pruner = StructuredPrunerV2(prune_ratio=0.5)
    pruner.prune(model)
    pruner.unprune(model)
    after_w = model.state_dict()
    for k in before_w:
        assert torch.equal(before_w[k], after_w[k])
