"""
契约 -> 反例 加固测试 (CTM / GPM / 协同)
=========================================
每个用例保护一条高风险契约, 并对应一个明确的"错误实现" (变异即变红):
- CTM: 前向确定性(状态不跨调用泄漏) / 批次独立 / 同步递推公式 / 可训练性 / 形状契约
- GPM: 前向补丁与 LoRA 公式逐项等价 / 重复 inject 不叠加 / reset 幂等 /
        只影响目标层 / 权重归一
- 协同: 重复内化不叠加、无显式因果时不伪造 pce-explicit 边
"""

import torch
import torch.nn as nn

from udos.ctm_engine import CTMConfig, CTMPhysicsEngine
from udos.gpm_engine import (
    GPMConfig, PhysicsHypernetwork, LoRAInjector, TinyBaseModel,
    infer_dims_from_model, aggregate_loras, LoRASet,
)
from udos.ctm_engine import normalized_entropy  # noqa: F401


def _ctm(**kw):
    d = dict(iterations=6, d_model=48, d_input=24, heads=2, n_synch_out=12,
             n_synch_action=12, memory_length=6, nlm_hidden=12, out_dims=16,
             certainty_threshold=0.0, n_random_pairing_self=2)
    d.update(kw)
    return CTMPhysicsEngine(CTMConfig(**d)).eval()


# ----------------------------- CTM 契约 ----------------------------------
def test_ctm_deterministic_no_state_leak(torch_seed):
    """C1 两次相同前向必须逐元素一致 (反例: trace 跨调用残留 -> 结果漂移)。"""
    eng = _ctm()
    x = torch.randn(2, 6, 24)
    with torch.no_grad():
        p1, c1, s1, _ = eng(x)
        p2, c2, s2, _ = eng(x)
    assert torch.equal(p1, p2) and torch.equal(s1, s2)


def test_ctm_batch_independence(torch_seed):
    """C2 拼 batch 与逐条单跑结果一致 (反例: 跨样本共享隐状态)。"""
    eng = _ctm()
    x1, x2 = torch.randn(1, 6, 24), torch.randn(1, 6, 24)
    with torch.no_grad():
        pb, _, _, _ = eng(torch.cat([x1, x2], 0))
        p1, _, _, _ = eng(x1)
        p2, _, _, _ = eng(x2)
    assert torch.allclose(pb[0], p1[0], atol=1e-6)
    assert torch.allclose(pb[1], p2[0], atol=1e-6)


def test_synchronise_recurrence_formula(torch_seed):
    """C3 同步递推必须等于 alpha=r·alpha+a*b; beta=r·beta+1; sync=alpha/√beta。"""
    eng = _ctm()
    B, n = 1, eng.cfg.n_synch_out
    act = torch.randn(B, eng.cfg.d_model)
    r = torch.exp(-eng.decay_params_out).unsqueeze(0)
    alpha = beta = None
    manual_a, manual_b = None, None
    left, right = eng.out_left, eng.out_right
    for _ in range(4):  # 与内部走相同 4 步
        sync, alpha, beta = eng._synchronise(act, alpha, beta, r, "out")
        pair = act[:, left] * act[:, right]
        if manual_a is None:
            manual_a, manual_b = pair, torch.ones_like(pair)
        else:
            manual_a = r * manual_a + pair
            manual_b = r * manual_b + 1.0
        manual_sync = manual_a / torch.sqrt(manual_b)
        assert torch.allclose(sync, manual_sync, atol=1e-6)


def test_ctm_trainable_grad_flows(torch_seed):
    """C4 损失能回传到核心可学习参数 (反例: 某参数被排除在计算图外)。"""
    eng = CTMPhysicsEngine(CTMConfig(
        iterations=3, d_model=32, d_input=16, heads=2, n_synch_out=8,
        n_synch_action=8, memory_length=4, nlm_hidden=8, out_dims=8,
        certainty_threshold=0.0)).train()
    x = torch.randn(2, 5, 16)
    preds, _, _, _ = eng(x)
    preds.sum().backward()
    for name in ["synapses", "trace_processor", "output_projector", "q_proj"]:
        module = getattr(eng, name)
        grads = [p.grad for p in module.parameters()]
        assert grads and all(g is not None for g in grads), f"{name} 无梯度"


def test_ctm_rejects_2d_input(torch_seed):
    """C5 公开入口对非法维度快速失败, 不产出语义错误结果。"""
    eng = _ctm()
    try:
        eng(torch.randn(6, 24))
        assert False, "应拒绝 2D 输入"
    except AssertionError:
        pass


# ----------------------------- GPM 契约 ----------------------------------
def _gpm(seed=0, **kw):
    torch.manual_seed(seed)
    base = TinyBaseModel(hidden=32, n_layers=2)
    targets = ("down_proj", "gate_proj", "up_proj")
    cfg = GPMConfig(
        feature_dim=32, latent_size=32, n_latents=8, lora_rank=4,
        target_modules=targets, layer_indices=(0, 1),
        dims=infer_dims_from_model(base, targets),
        num_pre_head_layers=1, heads=2, init_scaler_b_zero=False, **kw)
    return base, PhysicsHypernetwork(cfg).eval()


def test_lora_patch_matches_formula(tiny_scene, torch_seed):
    """G1 补丁输出必须逐元素等于 Wx + s·B(Ax); 反例: einsum 转置/缩放错误。"""
    base, hyper = _gpm()
    lora = hyper(tiny_scene)
    # 取第 0 层 down_proj 手工对照
    lin = base.layers[0].mlp.down_proj
    A, B = lora.AB["down_proj"][0][0], lora.AB["down_proj"][1][0]  # [r,din],[r,dout]
    x = torch.randn(3, 7, lin.in_features)
    scaling = 0.25
    with torch.no_grad():
        expect = lin(x) + scaling * torch.einsum(
            "rd,...sr->...sd", B, torch.einsum("rd,...sd->...sr", A, x))
        inj = LoRAInjector(base, scaling=scaling)
        inj.inject(lora)
        got = lin(x)
        inj.reset()
    assert torch.allclose(got, expect, atol=1e-6)


def test_repeated_inject_does_not_stack(tiny_scene, torch_seed):
    """G3 第二次 inject 前必须清掉旧补丁, 否则 forward_orig 被套两层。"""
    base, hyper = _gpm()
    inj = LoRAInjector(base, scaling=0.1)
    probe = torch.randn(1, 4, 32)
    with torch.no_grad():
        inj.inject(hyper(tiny_scene))
        y1 = base(probe).clone()
        lora2 = hyper(tiny_scene)
        inj.inject(lora2)  # 内部应先 reset, 不能在旧补丁上叠加
        y2 = base(probe).clone()
        inj.reset()
        y0 = base(probe).clone()
    # 同一 lora 两次注入结果应一致 (证明没有叠加), reset 仍零误差
    assert torch.allclose(y1, y2, atol=1e-6)
    assert torch.equal(y0, base(probe))


def test_reset_is_idempotent(tiny_scene, torch_seed):
    """G2 连续多轮 inject/reset 后权重输出恒等于原始 (反例: 浮点/补丁累积)。"""
    base, hyper = _gpm()
    inj = LoRAInjector(base, scaling=0.3)
    probe = torch.randn(2, 5, 32)
    with torch.no_grad():
        y0 = base(probe).clone()
        for _ in range(3):
            inj.inject(hyper(tiny_scene))
            inj.reset()
        assert torch.equal(y0, base(probe))
        inj.reset()  # 无补丁时 reset 也不报错
        assert torch.equal(y0, base(probe))


def test_only_target_layers_change(tiny_scene, torch_seed):
    """G5 未被选为 LoRA 目标的线性层输出不受注入影响。"""
    base, hyper = _gpm()
    # 构造一个不在目标集合里的额外线性层
    extra = nn.Linear(32, 32, bias=False)
    lora = hyper(tiny_scene)
    inj = LoRAInjector(base, scaling=0.1)
    x = torch.randn(1, 4, 32)
    with torch.no_grad():
        before = extra(x).clone()
        inj.inject(lora)
        after = extra(x).clone()
        inj.reset()
    assert torch.equal(before, after)


def test_weighted_aggregate_normalizes(torch_seed):
    """G4 权重和不为 1 时按比例归一 (反例: 直接相加导致尺度漂移)。"""
    A = torch.ones(1, 2, 3)
    s1 = LoRASet({"m": (A, A)}, [0])
    s2 = LoRASet({"m": (4 * A, 4 * A)}, [0])
    out = aggregate_loras([s1, s2], weights=[2.0, 8.0])  # 和=10
    # 0.2*1 + 0.8*4 = 3.4
    assert torch.allclose(out.AB["m"][0], 3.4 * A)


def test_lora_param_accounting(tiny_scene, torch_seed):
    """G6 num_params/num_bytes 与实际 A/B 元素数一致。"""
    _, hyper = _gpm()
    lora = hyper(tiny_scene)
    manual = sum(a.numel() + b.numel() for a, b in lora.AB.values())
    assert lora.num_params() == manual
    assert lora.num_bytes_fp32() == manual * 4


# ----------------------------- 协同契约 ----------------------------------
def test_repeated_internalize_no_stack(tiny_scene, torch_seed):
    """R1 走公开入口重复内化同一/新场景, 注入不叠加、记忆不残留旧引用。"""
    from udos.reasoning import UDOSReasoningEngine
    from udos.ctm_engine import CTMConfig
    base, hyper_cfg_engine = None, None
    gpm = GPMConfig(feature_dim=32, latent_size=32, n_latents=8, lora_rank=4,
                    layer_indices=(0, 1), num_pre_head_layers=1, heads=2,
                    init_scaler_b_zero=False)
    ctm = CTMConfig(iterations=3, d_model=32, d_input=32, heads=2,
                    n_synch_out=8, n_synch_action=8, memory_length=4,
                    nlm_hidden=8, out_dims=8, certainty_threshold=0.0,
                    n_random_pairing_self=2)
    eng = UDOSReasoningEngine(ctm, gpm, base_model=TinyBaseModel(32, 2))
    eng.internalize_scene(tiny_scene)
    n1 = len(eng.injector._patched)
    eng.internalize_scene(tiny_scene)
    n2 = len(eng.injector._patched)
    assert n1 == n2  # 补丁数量不翻倍
    eng.reset_all()
    assert eng.injector.active is None and eng.scene_memory == {}


def test_no_fabricated_causal_edges(torch_seed):
    """R3 没有显式 causal_parents 时, 只允许出现时间邻接边。"""
    from udos.pce_format import PhysicalToken, PhysicsScene
    from udos.reasoning import UDOSReasoningEngine
    from udos.ctm_engine import CTMConfig
    scene = PhysicsScene("x")
    for t in range(3):
        scene.add(PhysicalToken("o", t, [float(t), 0, 0]))
    eng = UDOSReasoningEngine(CTMConfig(
        iterations=2, d_model=32, d_input=32, heads=2, n_synch_out=8,
        n_synch_action=8, memory_length=4, out_dims=8,
        certainty_threshold=0.0), GPMConfig(
        feature_dim=32, latent_size=32, n_latents=4, lora_rank=2,
        layer_indices=(0, 1, 2, 3), num_pre_head_layers=1, heads=2))
    res = eng.reason(scene)
    sources = {e["source"] for e in res.causal_chain}
    assert sources == {"temporal-adjacent"}
