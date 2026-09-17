"""v7.0.2 探针：确定性可观测运动学通道 + 统一图零初始化 + 服务 horizon 键修复。

最高风险契约（每条配一个能被错误实现击穿的断言）：
- 匀速窗：三维加速度向量 a_lin≈0，弹簧/碰撞不误报。
- 加速窗：a_lin 精确等于 a·d（向量口径，根治标量 accel_a 伪负结果）。
- 弹簧窗：沿 PCA 主轴 + 带截距 q̈=−ω²q 反演 ω，短窗召回且对非弹簧零误报。
- 碰撞窗：跨碰撞帧的速度跳变被检出，非碰撞零误报。
- 运动学编码器末层零初始化 => 未训练模型仍严格恒等（不破坏 M1 稳定起点）。
- 旧 v7.0.1 checkpoint（config 无 use_kinematics）加载时不构建运动学通道（向后兼容）。
- 服务校准器缓存键含 horizon：不同视界得到形状正确、各自独立的区间（修复 v7.0.1 bug）。
"""
import torch

from udos7 import WorldModelCore
from udos7.kinematics import KIN_DIM, kinematic_features
from udos7.dynamics import (build_split, traj3d_accel, traj3d_spring,
                            traj3d_collision, traj3d_uniform, three_way_splits)
from udos7.scene import SceneChannel
from udos7.model import WorldModelCore as WMC
from udos7.train import TrainConfig, fit
from udos7.persistence import save_worldmodel
from udos7.metrics import kinematic_recovery
from udos7.server import V7Service

DT = 0.5
W = 6


def _win(states):
    return states[:W].unsqueeze(0)


def test_uniform_zero_accel_no_false_positives():
    d = torch.tensor([0.6, 0.8, 0.0])
    st = traj3d_uniform(8, DT, v0=1.3, d=d, x0=torch.zeros(3))
    f = kinematic_features(_win(st), DT)
    assert f.shape == (1, KIN_DIM)
    assert torch.allclose(f[0, 3:6], torch.zeros(3), atol=1e-5)
    assert f[0, 7].item() == 0.0       # omega invalid
    assert f[0, 9].item() == 0.0       # jump invalid
    assert torch.isfinite(f).all()


def test_accel_vector_exactly_recovered():
    d = torch.tensor([0.577, -0.577, 0.577])
    d = d / d.norm()
    a = 1.25
    st = traj3d_accel(8, DT, v0=0.4, a=a, d=d, x0=torch.zeros(3))
    f = kinematic_features(_win(st), DT)
    # 错误实现（漏除/多除一个 dt）会被 atol=0.02 击穿
    assert torch.allclose(f[0, 3:6], a * d, atol=0.02), f[0, 3:6]
    assert f[0, 7].item() == 0.0       # 恒定加速度不是弹簧


def test_spring_omega_recovered_and_no_false_positive():
    d = torch.tensor([0.0, 0.6, 0.8])
    omega = 1.2
    st = traj3d_spring(8, DT, amp=1.5, omega=omega, phi=0.3, d=d, x0=torch.zeros(3))
    f = kinematic_features(_win(st), DT)
    assert f[0, 7].item() == 1.0
    assert abs(f[0, 6].item() - omega) < 0.06
    # 非弹簧（匀速/加速）零误报
    for kind in ("uniform", "accel"):
        ds = build_split(seed=2026, n_traj_per_kind=16)
        m = ds.kind_mask(kind)
        K = kinematic_features(ds.X[m], DT)
        assert float(K[:, 7].mean()) == 0.0, kind


def test_collision_jump_detected_only_when_straddling():
    # v1=2 撞上静止 v2=0：约第 3 帧交换速度 => 窗内含跳变
    st = traj3d_collision(8, DT, x1=-2.0, v1=2.0, x2=0.5, v2=0.0)
    f = kinematic_features(_win(st), DT)
    assert f[0, 9].item() == 1.0
    assert f[0, 8].item() < 0.0          # 跳变后速度由 2 变 0，符号为负
    # 非碰撞零误报
    ds = build_split(seed=2026, n_traj_per_kind=16)
    for kind in ("uniform", "accel", "spring"):
        m = ds.kind_mask(kind)
        K = kinematic_features(ds.X[m], DT)
        assert float(K[:, 9].mean()) == 0.0, kind


def test_rejects_bad_shape_and_nonfinite():
    try:
        kinematic_features(torch.randn(W, 6), DT)
        assert False, "应拒绝二维输入"
    except ValueError:
        pass
    bad = torch.randn(1, W, 6)
    bad[0, 2, 1] = float("nan")
    try:
        kinematic_features(bad, DT)
        assert False, "应拒绝 NaN"
    except ValueError:
        pass


def test_kin_encoder_zero_init_keeps_untrained_identity():
    torch.manual_seed(0)
    m = WorldModelCore(window=W)
    assert m.scene.kin_encoder is not None
    x = torch.randn(4, W, 6)
    blind = m(x)
    cond = m(x, explicit=torch.randn(4, 4))
    # 运动学/场景桥/残差头全部零初始化 => 未训练严格恒等
    assert torch.allclose(blind, x[:, -1, :], atol=1e-6)
    assert torch.allclose(cond, x[:, -1, :], atol=1e-6)
    assert torch.count_nonzero(m.scene.kin_encoder.weight) == 0


def test_scene_context_carries_kinematics_and_toggle():
    ch = SceneChannel(window=W, use_kinematics=True)
    x = torch.randn(3, W, 6)
    sc = ch(x)
    assert sc.kinematics.shape == (3, KIN_DIM)
    ch0 = SceneChannel(window=W, use_kinematics=False)
    sc0 = ch0(x)
    assert ch0.kin_encoder is None and sc0.kinematics is None


def test_persistence_roundtrip_and_legacy_config_compat(tmp_path):
    torch.manual_seed(0)
    m = WMC(window=W, hidden=32, n_layers=1, use_kinematics=True)
    p = save_worldmodel(m, tmp_path / "k.pt")
    from udos7.persistence import load_worldmodel
    m2, ckpt = load_worldmodel(p)
    assert ckpt["config"]["use_kinematics"] is True
    assert m2.scene.kin_encoder is not None
    # 模拟 v7.0.1 旧 config（无 use_kinematics）=> 不构建运动学通道
    legacy = {"model_state": WMC(window=W, hidden=32, n_layers=1,
                                use_kinematics=False).state_dict(),
              "config": {"window": W, "hidden": 32, "n_layers": 1,
                         "scene_dim": 32}}
    lp = tmp_path / "legacy.pt"
    torch.save(legacy, lp)
    m3, _ = load_worldmodel(lp)
    assert m3.scene.kin_encoder is None


def test_kinematic_recovery_on_heldout():
    ds = build_split(seed=2026, n_traj_per_kind=64)
    r = kinematic_recovery(ds)
    assert r["accel_vector"]["skill"] > 0.95
    assert r["uniform_accel_falsepos_norm"] < 1e-4
    assert r["spring_omega"]["recall"] > 0.9
    assert r["spring_omega"]["false_positive_rate_non_spring"] == 0.0
    assert r["collision_jump"]["false_positive_rate_non_collision"] == 0.0


def test_service_calibrator_keyed_by_horizon(tmp_path):
    sp = three_way_splits(n_traj_per_kind=8)
    torch.manual_seed(0)
    m = WMC(window=W, hidden=32, n_layers=1)
    fit(m, sp["train"], sp["val"],
        TrainConfig(epochs=10, batch=64, lr=3e-3, patience=10, seed=0))
    ck = save_worldmodel(m, tmp_path / "h.pt", {"evidence_grade": "cpu-proto"})
    svc = V7Service(checkpoint=ck, n_traj_per_kind=8).load()
    ds = sp["test"]
    payload = {"window": ds.X[0].tolist(), "alpha": 0.1}
    o2 = svc.interval({**payload, "horizon": 2})
    o4 = svc.interval({**payload, "horizon": 4})
    assert len(o2["lower"][0]) == 2 and len(o4["lower"][0]) == 4
    # 两种 horizon 必须分别缓存（v7.0.1 仅按 use_explicit 缓存会串用带宽）
    assert (False, 2) in svc._calib and (False, 4) in svc._calib
