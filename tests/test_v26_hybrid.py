"""
v2.6.0 混合物理修正 (learned-residual 混合物理修正) 单元测试
==========================================================
锚点纪律:
    * 不挂载 hybrid / hybrid_weight=0 时, predict_next 与 v2.5.2 逐位一致。
    * attach 后, 在匀速运动上 hybrid 输出的运动学残差应低于纯模型输出
      (euler 骨架对匀速段是解析精确的)。
    * save/load 后 hybrid 输出逐位一致; 旧 checkpoint 无 hybrid 键 => None。
"""
import torch

from udos import __version__
from udos.ctm_engine import CTMConfig
from udos.dynamics import (build_parametric_dataset, kinematic_residual,
                           traj_uniform, _raw)
from udos.hybrid import HybridPhysicsCorrector
from udos.training import PhysicsPredictor, CTMTrainer, TrainConfig
from udos.persistence import save_predictor, load_predictor


def _cfg():
    return CTMConfig(iterations=8, d_model=64, d_input=32, heads=4,
                     n_synch_out=16, n_synch_action=8, memory_length=8,
                     nlm_hidden=16, out_dims=32, certainty_threshold=0.0)


def _make_model():
    torch.manual_seed(0)
    return PhysicsPredictor(_cfg(), scene_param_dim=4)


def _uniform_window(v0=1.3, dt=0.5, n=10, window=6):
    traj = traj_uniform(n, dt, v0=v0, x0=0.0)
    raw = torch.tensor([_raw(p, v) for p, v in traj], dtype=torch.float32)
    return raw[:window].unsqueeze(0), dt


def test_version():
    assert __version__ == "5.5.5"


def test_hybrid_param_count():
    h = HybridPhysicsCorrector()
    # (18*32+32) + (32*6+6) = 576+32+192+6 = 806
    assert h.n_params == 806


def test_hybrid_disabled_equivalence():
    """不 attach hybrid 时, predict_next 与底层 forward 末值逐位一致;
    attach 后 hybrid=False 仍走旧路径 (不计算 euler/residual)。"""
    model = _make_model()
    ds = build_parametric_dataset(n_per_kind=4, n_steps=14, window=6,
                                  horizon=2, dt=0.5, seed=7)
    xb, pb = ds.X[:4], ds.P[:4]
    ref = model.predict_next(xb, scene_params=pb)
    # 底层 forward 末值
    raw_out = model(xb, scene_params=pb)[2]
    assert torch.allclose(ref, raw_out, atol=1e-6)

    # attach 后 hybrid=False 仍逐位一致
    model.attach_hybrid(HybridPhysicsCorrector())
    out_off = model.predict_next(xb, scene_params=pb, hybrid=False)
    assert torch.allclose(out_off, ref, atol=1e-6)
    assert model.hybrid is not None
    model.detach_hybrid()
    assert model.hybrid is None


def test_hybrid_disabled_mode_returns_input():
    """corrector.forward(enabled=False) 原样返回 model_pred, 逐位一致。"""
    c = HybridPhysicsCorrector()
    mp = torch.randn(3, 6)
    last = torch.randn(3, 6)
    out = c(mp, last, 0.5, enabled=False)
    assert torch.equal(out, mp)


def test_hybrid_reduces_kinematic_residual():
    """匀速运动上, zero-init 残差 => hybrid≈euler 骨架, 运动学残差远低于随机模型输出。"""
    model = _make_model()
    window, dt = _uniform_window(v0=1.3)
    pb = torch.zeros(1, 4)
    model_pred = model.predict_next(window, scene_params=pb)
    prev = window[:, -1, :]
    resid_model = kinematic_residual(model_pred, prev, dt).item()

    # zero-init 残差: hybrid_pred = euler_pred (匀速段解析精确)
    c = HybridPhysicsCorrector()
    with torch.no_grad():
        c.mlp[-1].weight.zero_()
        c.mlp[-1].bias.zero_()
    model.attach_hybrid(c)
    hybrid_pred = model.predict_next(window, scene_params=pb, hybrid=True, dt=dt)
    resid_hybrid = kinematic_residual(hybrid_pred, prev, dt).item()

    assert resid_hybrid < resid_model
    # 匀速段 euler 骨架位置残差应接近 0
    assert resid_hybrid < 1e-4


def test_hybrid_save_load(tmp_path):
    """save/load 后 hybrid 输出逐位一致; 主权重不含 hybrid 键。"""
    model = _make_model()
    c = HybridPhysicsCorrector()
    model.attach_hybrid(c)
    ds = build_parametric_dataset(n_per_kind=4, n_steps=14, window=6,
                                  horizon=2, dt=0.5, seed=11)
    xb, pb = ds.X[:4], ds.P[:4]
    out_before = model.predict_next(xb, scene_params=pb, hybrid=True)

    ckpt = tmp_path / "hyb.pt"
    save_predictor(model, ckpt)
    loaded, meta = load_predictor(ckpt)
    assert loaded.hybrid is not None
    out_after = loaded.predict_next(xb, scene_params=pb, hybrid=True)
    assert torch.allclose(out_before, out_after, atol=1e-6)
    # 未开 hybrid 时与旧路径一致
    assert torch.allclose(loaded.predict_next(xb, scene_params=pb),
                          model.predict_next(xb, scene_params=pb), atol=1e-6)


def test_hybrid_ablation_weight_zero_keeps_old_path():
    """hybrid_weight=0 且已挂载 hybrid 时, 损失走旧路径: hybrid 参数无梯度,
    loss 数值与 detach_hybrid 时一致。"""
    model = _make_model()
    c = HybridPhysicsCorrector()
    model.attach_hybrid(c)
    ds = build_parametric_dataset(n_per_kind=8, n_steps=14, window=6,
                                  horizon=2, dt=0.5, seed=13)
    xb, pb, yb = ds.X[:8], ds.P[:8], ds.Y[:8]

    cfg_on = TrainConfig(hybrid_weight=0.0)
    tr_on = CTMTrainer(model, cfg_on)
    tr_on.opt.zero_grad()
    mse, phys, cert_pen = tr_on._parametric_loss(xb, pb, yb, ds.dt, epoch=0)
    assert float(tr_on._last_hyb_loss.detach()) == 0.0   # 未启用, hybrid 损失为零
    loss_on = mse + cfg_on.cert_weight * cert_pen
    loss_on.backward()
    # hybrid 参数不应有梯度
    for p in c.parameters():
        assert p.grad is None

    # 与 detach 后对比 loss 数值一致
    model.detach_hybrid()
    tr_off = CTMTrainer(model, TrainConfig(hybrid_weight=0.0))
    mse2, phys2, cert2 = tr_off._parametric_loss(xb, pb, yb, ds.dt, epoch=0)
    assert float(tr_off._last_hyb_loss.detach()) == 0.0
    assert torch.allclose(mse, mse2, atol=1e-6)
