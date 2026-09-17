import torch

from udos.gpm_engine import (
    GPMConfig, PhysicsHypernetwork, LoRAInjector, TinyBaseModel,
    infer_dims_from_model, aggregate_loras,
)


def _setup(tiny_scene):
    base = TinyBaseModel(hidden=32, n_layers=2)
    targets = ("down_proj", "gate_proj", "up_proj")
    dims = infer_dims_from_model(base, targets)
    # down: 128->32, gate/up: 32->128 (hidden*4)
    assert dims["down_proj"] == (128, 32)
    assert dims["gate_proj"] == (32, 128)
    cfg = GPMConfig(feature_dim=32, latent_size=32, n_latents=8, lora_rank=4,
                    target_modules=targets, layer_indices=(0, 1), dims=dims,
                    num_pre_head_layers=1, heads=2, init_scaler_b_zero=False)
    hyper = PhysicsHypernetwork(cfg)
    return base, hyper


def test_lora_shapes(tiny_scene, torch_seed):
    base, hyper = _setup(tiny_scene)
    lora = hyper(tiny_scene)
    A_down, B_down = lora.AB["down_proj"]
    # [n_layers, r, d_in/d_out]
    assert A_down.shape == (2, 4, 128)
    assert B_down.shape == (2, 4, 32)
    assert lora.num_params() > 0


def test_chunk_aggregation(tiny_scene, torch_seed):
    from udos.gpm_engine import PhysicsContextEncoder
    _, hyper = _setup(tiny_scene)
    # 三份完全相同的分块, 共享同一编码器 => 平均必等于其中一份
    enc = PhysicsContextEncoder(32, 32, attr_keys=sorted(
        {k for t in tiny_scene.tokens for k in t.attributes}))
    c1, c2, c3 = tiny_scene, tiny_scene, tiny_scene
    c1.scene_id, c2.scene_id, c3.scene_id = "c1", "c2", "c3"
    merged = hyper.forward_chunked([c1, c2, c3], encoder=enc)
    single = hyper(tiny_scene, encoder=enc)
    for m in merged.AB:
        assert torch.allclose(merged.AB[m][0], single.AB[m][0], atol=1e-6)


def test_weighted_aggregate(torch_seed):
    from udos.gpm_engine import LoRASet
    A = torch.ones(2, 3, 4)
    s1 = LoRASet({"m": (A, A)}, [0, 1])
    s2 = LoRASet({"m": (3 * A, 3 * A)}, [0, 1])
    out = aggregate_loras([s1, s2], weights=[1.0, 3.0])
    # 1*0.25 + 3*0.75 = 2.5
    assert torch.allclose(out.AB["m"][0], 2.5 * A)


def test_inject_and_lossless_reset(tiny_scene, torch_seed):
    base, hyper = _setup(tiny_scene)
    lora = hyper(tiny_scene)
    injector = LoRAInjector(base, scaling=0.1)
    probe = torch.randn(2, 5, 32)
    with torch.no_grad():
        y0 = base(probe).clone()
        injector.inject(lora)
        y1 = base(probe).clone()
        assert injector.active is not None
        injector.reset()
        y2 = base(probe).clone()
    # reset 必须零误差
    assert torch.equal(y0, y2)
    # 注入后输出发生改变 (LoRA 生效)
    assert not torch.allclose(y0, y1)


def test_zero_scaler_b_init_means_no_delta(tiny_scene, torch_seed):
    # scaler_B 默认初始化为 0 => LoRA 增量为 0 (LoRA 惯例, 训练稳定起点)
    base = TinyBaseModel(hidden=32, n_layers=2)
    targets = ("down_proj", "gate_proj", "up_proj")
    cfg = GPMConfig(
        feature_dim=32, latent_size=32, n_latents=8, lora_rank=4,
        target_modules=targets, layer_indices=(0, 1),
        dims=infer_dims_from_model(base, targets),
        num_pre_head_layers=1, heads=2)  # init_scaler_b_zero 默认 True
    hyper = PhysicsHypernetwork(cfg)
    lora = hyper(tiny_scene)
    for _, B in lora.AB.values():
        assert torch.all(B == 0)
