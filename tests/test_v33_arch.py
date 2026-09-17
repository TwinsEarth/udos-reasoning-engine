"""
v3.3.0 节点51: CTM 架构精炼 (残差连接 + 层归一化 + 可配置初始化) 测试
====================================================================
纪律:
    * 默认配置 (residual=False / act_norm=False / init_mode="legacy") 与旧架构
      逐位等价 —— 两次同种子构造 state_dict 全相等、前向逐元素相等;
    * residual / act_norm / init_mode 均为 opt-in, 开启后有限且与关闭时不同;
    * 新配置经 save_predictor/load_predictor round-trip; 轻量训练不崩溃。
analogy, not reproduction: 纯架构算子, 不涉及任何外部预训练权重。
"""
import torch

from udos.ctm_engine import CTMConfig, CTMPhysicsEngine
from udos.training import PhysicsPredictor, CTMTrainer, TrainConfig
from udos.dynamics import build_parametric_dataset
from udos.persistence import save_predictor, load_predictor

torch.set_num_threads(2)


def small_engine_cfg(**kw):
    base = dict(iterations=6, d_model=32, d_input=16, heads=4,
                n_synch_out=8, n_synch_action=4, memory_length=6,
                nlm_hidden=8, out_dims=8, certainty_threshold=0.0,
                scene_dim=None)
    base.update(kw)
    return CTMConfig(**base)


def make_input(seed=0, B=4, S=5):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, S, 16, generator=g)


def test_default_config_bitwise_equivalent():
    """默认配置两次同种子构造: state_dict 全相等 + 前向逐元素相等。"""
    torch.manual_seed(123)
    a = CTMPhysicsEngine(small_engine_cfg())
    torch.manual_seed(123)
    b = CTMPhysicsEngine(small_engine_cfg())
    sd_a, sd_b = a.state_dict(), b.state_dict()
    assert set(sd_a.keys()) == set(sd_b.keys())
    for k in sd_a:
        assert torch.equal(sd_a[k], sd_b[k]), f"参数 {k} 不等"
    x = make_input()
    pa, *_ = a.eval()(x)
    pb, *_ = b.eval()(x)
    assert torch.equal(pa, pb)


def test_explicit_off_equals_default():
    """显式 residual=False/act_norm=False/init_mode='legacy' 与缺省一致。"""
    torch.manual_seed(7)
    off = CTMPhysicsEngine(small_engine_cfg(residual=False, act_norm=False,
                                            init_mode="legacy"))
    torch.manual_seed(7)
    def_ = CTMPhysicsEngine(small_engine_cfg())
    for k in off.state_dict():
        assert torch.equal(off.state_dict()[k], def_.state_dict()[k])
    x = make_input(1)
    po, *_ = off.eval()(x)
    pd, *_ = def_.eval()(x)
    assert torch.equal(po, pd)


def test_residual_optin_forward():
    torch.manual_seed(0)
    off = CTMPhysicsEngine(small_engine_cfg())
    torch.manual_seed(0)
    on = CTMPhysicsEngine(small_engine_cfg(residual=True))
    x = make_input(2)
    po, *_ = off.eval()(x)
    pn, *_ = on.eval()(x)
    assert torch.isfinite(pn).all()
    # 开启残差应改变表示 (非逐位相同)
    assert not torch.allclose(po, pn, atol=1e-6)


def test_act_norm_optin_forward():
    torch.manual_seed(0)
    on = CTMPhysicsEngine(small_engine_cfg(act_norm=True))
    assert isinstance(on.act_norm, torch.nn.LayerNorm)
    # act_norm 开启后多一组 LayerNorm 参数 => 与默认 state_dict 键数不同
    torch.manual_seed(0)
    off = CTMPhysicsEngine(small_engine_cfg())
    assert len(on.state_dict()) == len(off.state_dict()) + 2  # weight + bias
    x = make_input(3)
    pn, *_ = on.eval()(x)
    assert torch.isfinite(pn).all()


def test_init_mode_configurable():
    for mode in ("xavier", "he"):
        torch.manual_seed(0)
        m = CTMPhysicsEngine(small_engine_cfg(init_mode=mode))
        assert torch.isfinite(m.output_projector.weight).all()
        # 与 legacy 的 output_projector 权重应不同
        torch.manual_seed(0)
        leg = CTMPhysicsEngine(small_engine_cfg(init_mode="legacy"))
        assert not torch.equal(m.output_projector.weight,
                               leg.output_projector.weight)
    # 非法 init_mode
    try:
        CTMPhysicsEngine(small_engine_cfg(init_mode="bogus"))
        assert False
    except ValueError:
        pass


def test_save_load_new_config(tmp_path):
    """带 residual=True 的新配置经 save/load 重建并前向一致。"""
    torch.manual_seed(5)
    cfg = small_engine_cfg(iterations=4, residual=True, act_norm=True)
    cfg.scene_dim = 32
    model = PhysicsPredictor(cfg, scene_param_dim=4)
    model.eval()
    ds = build_parametric_dataset(n_per_kind=2, n_steps=12, window=6,
                                  horizon=2, dt=0.5, seed=11)
    x, p = ds.X[:2], ds.P[:2]
    out_before = model.predict_next(x, scene_params=p)
    ckpt = tmp_path / "arch_optin.pt"
    save_predictor(model, str(ckpt))
    loaded, meta = load_predictor(str(ckpt))
    assert loaded.ctm.cfg.residual is True
    assert loaded.ctm.cfg.act_norm is True
    out_after = loaded.predict_next(x, scene_params=p)
    assert torch.allclose(out_before, out_after, atol=1e-6)
    assert torch.isfinite(out_after).all()


def test_training_not_crash():
    """轻量训练几步不崩溃 (正式口径由 build_v330 脚本完成)。"""
    torch.manual_seed(0)
    ds = build_parametric_dataset(n_per_kind=8, n_steps=12, window=6,
                                  horizon=2, dt=0.5, seed=21)
    tr, va = ds.split(0.8)
    cfg = small_engine_cfg(iterations=4, residual=True, act_norm=True)
    cfg.scene_dim = 32
    model = PhysicsPredictor(cfg, scene_param_dim=4)
    hist = CTMTrainer(model, TrainConfig(epochs=2, batch_size=16,
                                         patience=None)).train(tr, va)
    assert len(hist.train_loss) == 2
    assert all(v == v for v in hist.train_loss)  # finite
