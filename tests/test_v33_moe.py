"""
v3.3.0.dev1 节点52: 轻量 MoE 任务路由测试
====================================================================
纪律:
    * MoE 前向形状有限; 门控确实选 top-k 专家; 总参数量可控;
    * 空输入/非法 top_k 守卫;
    * opt-in 默认关: MoE 为独立外挂, 不挂载进 PhysicsPredictor, 主模型输出逐位不变。
analogy, not reproduction。
"""
import torch

from udos.moe import LightweightMoE
from udos.training import PhysicsPredictor
from udos.ctm_engine import CTMConfig
from udos.dynamics import build_parametric_dataset

torch.set_num_threads(2)


def make_moe():
    torch.manual_seed(0)
    return LightweightMoE(in_dim=16, out_dim=8, num_experts=4, top_k=2, seed=123)


def test_forward_shape_finite():
    moe = make_moe()
    x = torch.randn(5, 16)
    out = moe(x)
    assert out.shape == (5, 8)
    assert torch.isfinite(out).all()


def test_gate_selects_topk_experts():
    moe = make_moe()
    x = torch.randn(4, 16)
    out, idx, w = moe(x, return_routing=True)
    assert idx.shape == (4, 2) and w.shape == (4, 2)
    # 门控权重归一
    assert torch.allclose(w.sum(dim=-1), torch.ones(4), atol=1e-5)
    # 选出的专家确为 router logits 的 top-2
    logits = moe.router(x)
    true_top = logits.topk(2, dim=-1).values
    # 与 idx 对应行比较
    picked = torch.gather(logits, 1, idx)
    assert torch.allclose(picked.sort(dim=-1, descending=True).values,
                          true_top, atol=1e-5)


def test_param_count_controlled():
    moe = make_moe()
    # 专家: E*(in*out + out); router: in*E + E
    expect = 4 * (16 * 8 + 8) + (16 * 4 + 4)
    assert moe.num_parameters() == expect
    # 参数量随专家数线性增长, 可控
    m_fewer = LightweightMoE(16, 8, num_experts=2, top_k=1)
    assert m_fewer.num_parameters() < moe.num_parameters()


def test_topk_eq_num_experts_uses_all():
    moe = LightweightMoE(8, 4, num_experts=3, top_k=3)
    x = torch.randn(3, 8)
    out = moe(x)
    assert out.shape == (3, 4) and torch.isfinite(out).all()


def test_empty_and_bad_input_guard():
    moe = make_moe()
    try:
        moe(torch.randn(0, 16))
        assert False
    except ValueError:
        pass
    try:
        moe(torch.randn(3, 7))   # 维度不匹配
        assert False
    except ValueError:
        pass
    try:
        LightweightMoE(8, 4, num_experts=2, top_k=5)
        assert False
    except ValueError:
        pass


def test_optin_default_off_predictor_unchanged():
    """MoE 为独立外挂, 不进 PhysicsPredictor 默认前向 => 主模型输出逐位不变。"""
    torch.manual_seed(0)
    cfg = CTMConfig(iterations=4, d_model=32, d_input=16, heads=4,
                    n_synch_out=8, n_synch_action=4, memory_length=6,
                    nlm_hidden=8, out_dims=8, certainty_threshold=0.0)
    cfg.scene_dim = 32
    model = PhysicsPredictor(cfg, scene_param_dim=4)
    model.eval()
    ds = build_parametric_dataset(n_per_kind=2, n_steps=12, window=6,
                                  horizon=2, dt=0.5, seed=5)
    before = model.predict_next(ds.X[:3], scene_params=ds.P[:3])
    # 构造 MoE 但不挂载 => 模型不变
    _ = LightweightMoE(16, 8, num_experts=4, top_k=2)
    after = model.predict_next(ds.X[:3], scene_params=ds.P[:3])
    assert torch.equal(before, after)
