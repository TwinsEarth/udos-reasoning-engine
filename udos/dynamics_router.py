"""
四类运动识别 + 估计路由 + 类型相关不确定性 (v5.5.2)
==================================================
在 v5.4.8 经典场景参数估计、v5.5.0 学习型场景头、v5.5.1 蒙特卡洛扇形之上,
本模块做两件事:

1. **运动类型路由 (确定性, 无需训练)**: 仅凭观测窗口 [B,W,6] 判别四类合成
   运动 (uniform / accel / spring / collision), 并按类型给出"槽位干净"的
   4 维场景参数, 供主预测员消费。关键判别是**模型选择**而非单一阈值:

       简谐模型   acc = -omega^2 x   的相对残差 rr
       匀加速模型 acc = const        的相对残差 rc
   仅当简谐模型显著不劣于 (rr < rc) 且通过门限时才判 spring, 从而在弹簧
   召回拉满的同时对 accel 零误判 (短窗内二次位置轨迹会"看起来像"简谐)。

   碰撞用帧间速度跳变的稳健统计 (max > k*median + floor) 识别; 不含跳变
   的碰撞窗口在局部就是匀速, 归 uniform 对下游无害 (碰撞 P 槽0=v1)。

2. **类型条件不确定性**: KindSpecificParamErrorModel 按真实类型拟合逐槽位
   bias/std, 推理时按*预测*类型绑定采样; fit_kind_conformal_inflation 在
   独立校准集上对每个类型分别求 split-conformal 膨胀因子, 修正 v5.5.1
   全局高斯导致的分类型欠/过覆盖 (加速欠覆盖、碰撞过覆盖)。

诚实边界:
  * 路由只识别生成器的四类一维运动, 不是通用时间序列分类器。
  * 碰撞对方速度 other_v2 在主体窗口物理不可见, 路由同样不编造 (置 0)。
  * 类型条件 conformal 保证的是各类型的边际 (样本×步×维) 覆盖率。
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch

from .dynamics import RAW_DIM, SCENE_PARAM_DIM, SCENE_PARAM_NAMES
from .scene_estimator import _validate

logger = logging.getLogger("udos.dynamics_router")

CLASS_NAMES: tuple = ("uniform", "accel", "spring", "collision")
_IDX = {n: i for i, n in enumerate(CLASS_NAMES)}

_SLOT_V0 = SCENE_PARAM_NAMES.index("v0")
_SLOT_A = SCENE_PARAM_NAMES.index("accel_a")
_SLOT_OMEGA = SCENE_PARAM_NAMES.index("spring_omega")
_SLOT_V2 = SCENE_PARAM_NAMES.index("other_v2")

# 工作点门限 (由 held-out seed=2026 探针标定, 见 tests 与 reports/v552*)
LINEAR_R2_MIN = 0.95     # 速度线性优度门限
OMEGA2_MIN = 0.09        # omega^2 下限, 排除近零曲率
HARM_RR_MAX = 0.10       # 简谐模型相对残差上限
ACCEL_MIN = 0.12         # |斜率| 超过它才算 accel, 否则归 uniform
JUMP_K = 3.0             # 碰撞跳变: max|dv| > k*median|dv| + floor
JUMP_FLOOR = 0.35
_EPS = 1e-12


@dataclass
class DynamicsRoute:
    """分类与路由信号。

    labels:        [B] long, 取值 0..3 对应 CLASS_NAMES
    uniform/accel/spring/collision: [B] bool 掩码
    c_hat/a_hat:   [B] 速度线性最小二乘截距/斜率
    lin_r2:        [B] 速度线性优度
    omega:         [B] 简谐反演角频率 (sqrt(omega^2), 已截断负值)
    harmonic_rr:   [B] 简谐模型相对残差 (越小越像简谐)
    const_rr:      [B] 匀加速 (常二阶差分) 模型相对残差
    jump_max/jump_med: [B] 帧间速度跳变的最大/中位绝对值
    """

    labels: torch.Tensor
    uniform: torch.Tensor
    accel: torch.Tensor
    spring: torch.Tensor
    collision: torch.Tensor
    c_hat: torch.Tensor
    a_hat: torch.Tensor
    lin_r2: torch.Tensor
    omega: torch.Tensor
    harmonic_rr: torch.Tensor
    const_rr: torch.Tensor
    jump_max: torch.Tensor
    jump_med: torch.Tensor

    def label_names(self) -> List[str]:
        return [CLASS_NAMES[int(i)] for i in self.labels.tolist()]


def _signals(window: torch.Tensor, dt: float):
    w = _validate(window, dt).float()
    B, W, _ = w.shape
    device = w.device
    t = torch.arange(W, dtype=torch.float32, device=device) * float(dt)
    tm = t.mean()
    t_c = t - tm
    stt = float((t_c ** 2).sum().clamp_min(_EPS))

    x = w[:, :, 0]
    vx = w[:, :, 3]
    vm = vx.mean(dim=1, keepdim=True)
    a_hat = (t_c.unsqueeze(0) * (vx - vm)).sum(dim=1) / stt
    c_hat = vm.squeeze(1) - a_hat * float(tm)
    v_pred = c_hat.unsqueeze(1) + a_hat.unsqueeze(1) * t.unsqueeze(0)
    ss_tot = ((vx - vm) ** 2).sum(dim=1)
    ss_res = ((vx - v_pred) ** 2).sum(dim=1)
    lin_r2 = torch.where(ss_tot > _EPS, 1.0 - ss_res / ss_tot.clamp_min(_EPS),
                         torch.ones_like(ss_tot))

    acc = (x[:, 2:] - 2.0 * x[:, 1:-1] + x[:, :-2]) / (float(dt) ** 2)
    xi = x[:, 1:-1]
    sxx = (xi ** 2).sum(dim=1).clamp_min(_EPS)
    sxa = (xi * acc).sum(dim=1)
    omega2 = -(sxa / sxx)
    pa = -omega2.unsqueeze(1) * xi
    e_acc = (acc ** 2).sum(dim=1).clamp_min(_EPS)
    harmonic_rr = ((acc - pa) ** 2).sum(dim=1) / e_acc
    const_rr = ((acc - acc.mean(dim=1, keepdim=True)) ** 2).sum(dim=1) / e_acc
    omega = torch.sqrt(omega2.clamp_min(0.0))

    dv = (vx[:, 1:] - vx[:, :-1]).abs()
    jump_med = dv.median(dim=1).values
    jump_max = dv.max(dim=1).values
    return dict(B=B, c_hat=c_hat, a_hat=a_hat, lin_r2=lin_r2, omega=omega,
                omega2=omega2, harmonic_rr=harmonic_rr, const_rr=const_rr,
                jump_med=jump_med, jump_max=jump_max)


@torch.no_grad()
def classify_dynamics(window: torch.Tensor, dt: float = 0.5, *,
                      linear_r2_min: float = LINEAR_R2_MIN,
                      omega2_min: float = OMEGA2_MIN,
                      harmonic_rr_max: float = HARM_RR_MAX,
                      accel_min: float = ACCEL_MIN,
                      jump_k: float = JUMP_K,
                      jump_floor: float = JUMP_FLOOR) -> DynamicsRoute:
    """把窗口分类为 uniform/accel/spring/collision (确定性, 可批量)。"""
    s = _signals(window, dt)
    collision = s["jump_max"] > jump_k * s["jump_med"] + jump_floor
    spring = ((~collision)
              & (s["omega2"] >= omega2_min)
              & (s["harmonic_rr"] <= harmonic_rr_max)
              & (s["harmonic_rr"] < s["const_rr"]))
    linear = (~collision) & (~spring) & (s["lin_r2"] >= linear_r2_min)
    accel = linear & (s["a_hat"].abs() >= accel_min)
    uniform = (~collision) & (~spring) & (~accel)

    labels = torch.full((s["B"],), _IDX["uniform"], dtype=torch.long,
                        device=s["c_hat"].device)
    labels[accel] = _IDX["accel"]
    labels[spring] = _IDX["spring"]
    labels[collision] = _IDX["collision"]

    return DynamicsRoute(
        labels=labels, uniform=uniform, accel=accel, spring=spring,
        collision=collision, c_hat=s["c_hat"], a_hat=s["a_hat"],
        lin_r2=s["lin_r2"], omega=s["omega"], harmonic_rr=s["harmonic_rr"],
        const_rr=s["const_rr"], jump_max=s["jump_max"], jump_med=s["jump_med"])


@torch.no_grad()
def routed_scene_params(window: torch.Tensor, dt: float = 0.5,
                        route: Optional[DynamicsRoute] = None) -> torch.Tensor:
    """按类型给出槽位干净的 4 维场景参数 [B,4] (有限值, 可直接喂主预测员)。

        uniform / collision -> [c, 0, 0, 0]
        accel               -> [c, a, 0, 0]
        spring              -> [0, 0, omega, 0]
    碰撞对方速度 other_v2 主体窗口不可见, 置 0 (不编造)。
    """
    w = _validate(window, dt).float()
    B = w.size(0)
    r = route if route is not None else classify_dynamics(w, dt)
    values = torch.zeros((B, SCENE_PARAM_DIM), dtype=torch.float32,
                         device=w.device)
    moving = r.uniform | r.accel | r.collision
    values[moving, _SLOT_V0] = r.c_hat[moving]
    values[r.accel, _SLOT_A] = r.a_hat[r.accel]
    values[r.spring, _SLOT_OMEGA] = r.omega[r.spring]
    return values


# ---------------------------------------------------------------------------
# 类型相关参数误差模型 + 类型条件 conformal
# ---------------------------------------------------------------------------
class _BoundKindErrorModel:
    """绑定某批样本的预测类型后的误差模型, 暴露 monte_carlo_rollout 需要的
    .sample(p_hat, n, generator) 接口。"""

    def __init__(self, parent: "KindSpecificParamErrorModel",
                 labels: torch.Tensor):
        self.parent = parent
        self.labels = labels.long()

    def sample(self, p_hat: torch.Tensor, n: int,
               generator: Optional[torch.Generator] = None) -> torch.Tensor:
        if p_hat.dim() != 2 or p_hat.size(-1) != self.parent.bias.size(-1):
            raise ValueError("p_hat 形状须为 [B, scene_dim]")
        if self.labels.shape[0] != p_hat.size(0):
            raise ValueError("绑定 labels 与 batch 大小不一致")
        bias = self.parent.bias[self.labels].unsqueeze(0)     # [1,B,4]
        std = self.parent.std[self.labels].unsqueeze(0)
        z = torch.randn(n, p_hat.size(0), p_hat.size(-1),
                        generator=generator, dtype=p_hat.dtype,
                        device=p_hat.device)
        return (p_hat.unsqueeze(0) - bias) + std * z


class KindSpecificParamErrorModel:
    """按真实类型拟合 (P_hat - P_true) 的逐槽位 bias/std, 推理按预测类型绑定。"""

    def __init__(self, bias: torch.Tensor, std: torch.Tensor):
        if bias.shape != std.shape or bias.dim() != 2:
            raise ValueError("bias/std 须为同形 [K,scene_dim]")
        if not bool(torch.isfinite(bias).all() and torch.isfinite(std).all()):
            raise ValueError("bias/std 含非有限值")
        if not bool((std > 0).all()):
            raise ValueError("std 必须严格为正")
        self.bias = bias
        self.std = std
        self.n_kinds = bias.size(0)

    @classmethod
    def fit(cls, p_hat: torch.Tensor, p_true: torch.Tensor,
            true_labels: torch.Tensor, n_kinds: int = len(CLASS_NAMES)
            ) -> "KindSpecificParamErrorModel":
        if p_hat.shape != p_true.shape:
            raise ValueError("p_hat/p_true 形状不一致")
        labs = true_labels.long()
        bias = torch.zeros(n_kinds, p_hat.size(-1), dtype=p_hat.dtype)
        std = torch.full((n_kinds, p_hat.size(-1)), 1e-4, dtype=p_hat.dtype)
        for k in range(n_kinds):
            m = labs == k
            if int(m.sum()) >= 2:
                err = p_hat[m] - p_true[m]
                bias[k] = err.mean(dim=0)
                std[k] = err.std(dim=0).clamp_min(1e-4)
        return cls(bias, std)

    def bind(self, labels: torch.Tensor) -> _BoundKindErrorModel:
        labs = labels.long()
        if labs.dim() != 1 or bool(((labs < 0) | (labs >= self.n_kinds)).any()):
            raise ValueError("labels 须为 [B] 且取值在类型范围内")
        return _BoundKindErrorModel(self, labs)


def apply_kind_inflation(low: torch.Tensor, median: torch.Tensor,
                         high: torch.Tensor, inflation: torch.Tensor,
                         labels: torch.Tensor):
    """按每行预测类型施加各自的 conformal 膨胀因子, 返回 (low, high)。"""
    if not (low.shape == median.shape == high.shape):
        raise ValueError("low/median/high 形状须一致")
    if not bool(torch.isfinite(inflation).all()) or bool((inflation <= 0).any()):
        raise ValueError("inflation 须为正有限值")
    labs = labels.long()
    if labs.shape[0] != median.size(0) or bool(
            ((labs < 0) | (labs >= inflation.numel())).any()):
        raise ValueError("labels 与样本数/类型数不匹配")
    half = (high - low) / 2.0
    r = inflation[labs].view(median.size(0), *([1] * (median.dim() - 1)))
    return median - r * half, median + r * half


@torch.no_grad()
def fit_kind_conformal_inflation(predictor, dataset, pred_labels: torch.Tensor,
                                 true_labels: Optional[torch.Tensor] = None,
                                 *, head=None, point_params=None,
                                 kind_error_model=None,
                                 nominal: float = 0.80, n_samples: int = 48,
                                 generator: Optional[torch.Generator] = None,
                                 n_kinds: int = len(CLASS_NAMES)
                                 ) -> torch.Tensor:
    """在独立校准集上为每个类型求 split-conformal 等比膨胀因子 [K]。

    pred_labels: 校准窗口的*预测*类型 (绑定误差模型用);
    分组求分位时用 true_labels (默认取 dataset.kinds 真实类型)。
    点估计参数由 head(dataset.X) 给出, 或直接传 point_params。
    """
    from .scene_fan import monte_carlo_rollout
    if not (0.0 < nominal < 1.0):
        raise ValueError("nominal 须在 (0,1)")
    if kind_error_model is None:
        raise ValueError("KindSpecificParamErrorModel 为必填")
    if point_params is None:
        if head is None:
            raise ValueError("head 与 point_params 至少提供一个")
        point_params = head(dataset.X)
    bound = kind_error_model.bind(pred_labels.long())
    fan = monte_carlo_rollout(predictor, dataset.X, point_params, bound,
                              n_samples=n_samples, horizon=dataset.Y.size(1),
                              quantiles=((1 - nominal) / 2, 0.5,
                                         (1 + nominal) / 2),
                              generator=generator)
    if true_labels is None:
        true_labels = torch.tensor([CLASS_NAMES.index(k) for k in dataset.kinds],
                                   dtype=torch.long)
    true_labels = true_labels.long()
    infl = torch.ones(n_kinds, dtype=torch.float32)
    half = ((fan.high - fan.low) / 2.0).clamp_min(1e-6)
    ratio = ((dataset.Y - fan.median).abs() / half)
    for k in range(n_kinds):
        m = true_labels == k
        if int(m.sum()) > 0:
            infl[k] = torch.quantile(ratio[m].flatten(), nominal,
                                     interpolation="higher")
    return infl.clamp_min(1e-6)
