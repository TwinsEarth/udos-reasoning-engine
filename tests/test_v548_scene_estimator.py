"""
v5.4.8 场景隐藏参数估计器 (Scene Param Estimator)
=================================================
在只有观测窗口 [W,6]=[pos3,vel3]、没有 GPM 显式场景参数时, 用**确定性经典
运动学** (无需训练) 反演 4 个隐藏场景参数, 并对每个槽位给出物理可观测性掩码:

    v0          窗口局部初速度 (速度本就在 raw 中, 恒可观测; 匀加速下为
                窗口起点速度 = 全局 v0 + a*s*dt, 非轨迹全局 v0)
    accel_a     速度对时间最小二乘斜率; 仅当速度线性 (lin_r2>=0.95, 即
                匀速/匀加速) 标记可观测; 弹簧/碰撞的非常数加速标记不可观测
    spring_omega 由简谐关系 a=-w^2 x (位置二阶差分) 反演; 仅当
                (速度非线性 lin_r2<0.95) 且 (w2>=0.09) 且 (简谐拟合相对
                残差 rr<=0.10) 才标记可观测, 否则诚实置 NaN
    other_v2    碰撞对方速度, 主体窗口物理上看不到 -> 恒不可观测, 恒 NaN

在生成器 (W=6, dt=0.5) 上实测工作点 (精度优先, 三 seed):
    弹簧 omega 召回约 0.78 (短窗弧度不足是真实可辨识性天花板),
    匀速/匀加速/碰撞的 omega 误报率为 0。
5.5.0 才引入联合训练的学习型估计头; 本版只做可解释的经典估计与诚实掩码。
"""

import math

import pytest
import torch

from udos.dynamics import (
    SCENE_PARAM_DIM,
    build_parametric_dataset,
    traj_accel,
    traj_collision,
    traj_spring,
    traj_uniform,
    _raw,
)
from udos.scene_estimator import SceneEstimate, estimate_scene_params


def _window(traj, start=0, w=6):
    raw = torch.tensor([_raw(p, v) for p, v in traj], dtype=torch.float32)
    return raw[start:start + w]


def test_uniform_slots(torch_seed):
    w = _window(traj_uniform(12, 0.5, v0=1.3))
    est = estimate_scene_params(w, 0.5)
    assert est.values.shape == (1, SCENE_PARAM_DIM)
    assert est.observable.dtype == torch.bool
    assert est.observable[0, 0].item() is True            # v0 恒可观测
    assert est.values[0, 0].item() == pytest.approx(1.3, abs=1e-4)
    assert est.values[0, 1].item() == pytest.approx(0.0, abs=1e-4)  # a=0
    assert est.observable[0, 1].item() is True
    # 反例: 匀速不得臆测弹簧频率
    assert est.observable[0, 2].item() is False
    assert math.isnan(est.values[0, 2].item())


def test_accel_slope_and_local_intercept(torch_seed):
    # 窗口从第 3 帧开始: 局部初速度 c = v0 + a*s*dt = 0.5 + 1.0*3*0.5 = 2.0
    w = _window(traj_accel(12, 0.5, v0=0.5, a=1.0), start=3)
    est = estimate_scene_params(w, 0.5)
    assert est.observable[0, 1].item() is True
    assert est.values[0, 1].item() == pytest.approx(1.0, abs=2e-3)
    assert est.values[0, 0].item() == pytest.approx(2.0, abs=2e-3)
    assert est.observable[0, 2].item() is False
    assert math.isnan(est.values[0, 2].item())


def test_spring_omega_recovered(torch_seed):
    w = _window(traj_spring(12, 0.5, amp=1.5, omega=1.2, phi=0.3))
    est = estimate_scene_params(w, 0.5)
    assert est.observable[0, 2].item() is True
    assert est.values[0, 2].item() == pytest.approx(1.2, abs=0.06)
    # 弹簧不是常数加速 regime
    assert est.observable[0, 1].item() is False


def test_collision_never_hallucinates_omega_or_v2(torch_seed):
    traj = traj_collision(14, 0.5, x1=-2.0, v1=1.5, x2=1.5, v2=0.3)
    raw = torch.tensor([_raw(p, v) for p, v in traj], dtype=torch.float32)
    for s in range(raw.size(0) - 6):
        est = estimate_scene_params(raw[s:s + 6], 0.5)
        assert est.observable[0, 2].item() is False       # 不把碰撞拐点当弹簧
        assert math.isnan(est.values[0, 2].item())
        assert est.observable[0, 3].item() is False       # v2 永远看不到
        assert math.isnan(est.values[0, 3].item())


def test_other_v2_never_observable_on_dataset(torch_seed):
    # 最高风险契约: 四类运动里 other_v2 一律不可观测、一律 NaN (不允许编造)
    ds = build_parametric_dataset(n_per_kind=64, seed=42)
    est = estimate_scene_params(ds.X, 0.5)
    assert est.observable[:, 3].sum().item() == 0
    assert torch.isnan(est.values[:, 3]).all().item()


def test_omega_precision_recall_operating_point(torch_seed):
    # 锁定实测工作点: 非弹簧零误报, 弹簧召回 >=0.70 (W=6 短窗的诚实上限)
    for seed in (42, 7, 2026):
        ds = build_parametric_dataset(n_per_kind=96, seed=seed)
        est = estimate_scene_params(ds.X, 0.5)
        om = est.observable[:, 2]
        for kind in ("uniform", "accel", "collision"):
            rate = om[ds.kind_mask(kind)].float().mean().item()
            assert rate == 0.0, f"seed{seed} {kind} omega 误报 {rate}"
        recall = om[ds.kind_mask("spring")].float().mean().item()
        assert recall >= 0.70, f"seed{seed} spring 召回 {recall} 低于 0.70"


def test_batch_matches_single(torch_seed):
    ds = build_parametric_dataset(n_per_kind=16, seed=11)
    batch = estimate_scene_params(ds.X[:8], 0.5)
    for i in range(8):
        one = estimate_scene_params(ds.X[i:i + 1], 0.5)
        mb = torch.isnan(batch.values[i])
        mo = torch.isnan(one.values[0])
        assert torch.equal(mb, mo)
        finite = ~mb
        assert torch.allclose(batch.values[i][finite],
                              one.values[0][finite], atol=1e-5)
        assert torch.equal(batch.observable[i], one.observable[0])


def test_values_filled_replaces_nan_only(torch_seed):
    w = _window(traj_uniform(12, 0.5, v0=1.0))
    est = estimate_scene_params(w, 0.5)
    filled = est.values_filled(0.0)
    assert not torch.isnan(filled).any()
    # 可观测槽位值不被改动
    assert filled[0, 0].item() == pytest.approx(est.values[0, 0].item(), abs=1e-7)
    assert filled[0, 3].item() == 0.0


def test_rejects_short_window(torch_seed):
    with pytest.raises(ValueError):
        estimate_scene_params(torch.zeros(2, 6), 0.5)        # W<3
    with pytest.raises(ValueError):
        estimate_scene_params(torch.zeros(3, 2, 6), 0.5)     # 批量 W=2


def test_rejects_nan_and_bad_dt(torch_seed):
    w = _window(traj_uniform(12, 0.5, v0=1.0)).clone()
    w[0, 0] = float("nan")
    with pytest.raises(ValueError):
        estimate_scene_params(w, 0.5)
    good = _window(traj_uniform(12, 0.5, v0=1.0))
    with pytest.raises(ValueError):
        estimate_scene_params(good, 0.0)    # dt 必须为正


def test_rejects_bad_shape(torch_seed):
    with pytest.raises(ValueError):
        estimate_scene_params(torch.zeros(6), 0.5)          # 1D
    with pytest.raises(ValueError):
        estimate_scene_params(torch.zeros(2, 6, 5), 0.5)    # 末维 != 6
