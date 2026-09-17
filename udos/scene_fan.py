"""
场景参数不确定性 -> 蒙特卡洛轨迹扇形 (v5.5.1)
============================================
v5.5.0 的学习型场景头只给点估计 P_hat。本模块把"参数估计不确定性"传播到未来
轨迹:

  1. 在校准集上拟合参数误差模型  err = P_hat - P_true ~ N(bias, diag(std)^2);
  2. 推理时对 P_hat 去偏后采样 M 组参数, 各跑一次冻结主预测员 rollout;
  3. 对 M 条轨迹取分位数, 得到 p10/p50/p90 轨迹扇形;
  4. 用**独立校准集**做 split-conformal 膨胀 (以中位为中心、按扇形半宽等比放大),
     使 pooled 经验覆盖率达到名义水平 (默认 80% 带 = p10..p90)。

诚实边界:
  * 全局(类型无关)对角高斯无法表达分类型不确定性差异, pooled 覆盖达标但单类型
    可能欠/过覆盖 (实测加速欠、碰撞过); 类型相关收紧在 v5.5.2 路由后处理。
  * 这里只覆盖"场景参数估计不确定性", 不含模型结构/观测噪声不确定性。
  * conformal 保证的边际单位是 (样本×未来步×状态维) 的 pooled 覆盖率。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch

from .dynamics import SCENE_PARAM_NAMES, SCENE_PARAM_DIM

_STD_FLOOR = 1e-4


@dataclass
class ParamErrorModel:
    """P_hat - P_true 的逐槽位偏差与对角标准差 (全局、类型无关)。"""
    bias: torch.Tensor          # [scene_dim]
    std: torch.Tensor           # [scene_dim]

    @classmethod
    def fit(cls, head, dataset) -> "ParamErrorModel":
        with torch.no_grad():
            p_hat = head(dataset.X)
        err = p_hat - dataset.P
        bias = err.mean(dim=0)
        std = err.std(dim=0).clamp_min(_STD_FLOOR)
        return cls(bias=bias.detach(), std=std.detach())

    def sample(self, p_hat: torch.Tensor, n_samples: int,
               generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """返回 [n_samples, B, scene_dim] 的去偏参数样本。"""
        B = p_hat.size(0)
        z = torch.randn(n_samples, B, p_hat.size(1), generator=generator,
                        dtype=p_hat.dtype, device=p_hat.device)
        center = (p_hat - self.bias).unsqueeze(0)
        return center + self.std.view(1, 1, -1) * z


@dataclass
class TrajectoryFan:
    low: torch.Tensor       # [B,H,R]
    median: torch.Tensor
    high: torch.Tensor
    quantiles: Tuple[float, float, float]


def monte_carlo_rollout(predictor, window: torch.Tensor,
                        point_params: torch.Tensor,
                        error_model: ParamErrorModel,
                        n_samples: int, horizon: int,
                        quantiles: Sequence[float] = (0.10, 0.50, 0.90),
                        generator: Optional[torch.Generator] = None
                        ) -> TrajectoryFan:
    """对点估计参数采样 n_samples 次并 rollout, 返回分位数轨迹扇形。"""
    if len(quantiles) != 3:
        raise ValueError("quantiles 须为 (low, median, high) 三元组")
    qlo, qmid, qhi = float(quantiles[0]), float(quantiles[1]), float(quantiles[2])
    if not (qlo < qmid < qhi):
        raise ValueError("quantiles 须严格递增")
    params = error_model.sample(point_params, int(n_samples), generator)
    rolls = []
    with torch.no_grad():
        for i in range(int(n_samples)):
            rolls.append(predictor.rollout(
                window, int(horizon), scene_params=params[i]))
    samples = torch.stack(rolls, dim=0)          # [n,B,H,R]
    q = torch.quantile(samples, torch.tensor([qlo, qmid, qhi],
                      dtype=samples.dtype), dim=0)
    return TrajectoryFan(low=q[0], median=q[1], high=q[2],
                         quantiles=(qlo, qmid, qhi))


def coverage_fraction(low: torch.Tensor, high: torch.Tensor,
                      truth: torch.Tensor) -> float:
    """truth 落在 [low,high] 闭区间内的 pooled 比例。"""
    inside = (truth >= low) & (truth <= high)
    return float(inside.float().mean())


def per_step_coverage(low: torch.Tensor, high: torch.Tensor,
                      truth: torch.Tensor) -> list:
    inside = ((truth >= low) & (truth <= high)).float()
    return [float(inside[:, t].mean()) for t in range(truth.size(1))]


def fit_conformal_inflation(predictor, head, calib_dataset,
                            error_model: ParamErrorModel, *,
                            n_samples: int = 64, horizon: int = 4,
                            nominal: float = 0.80,
                            generator: Optional[torch.Generator] = None
                            ) -> float:
    """
    独立校准集上估计扇形半宽的等比膨胀因子 r, 使 pooled 覆盖达到 nominal。
    r = 分位( |Y - median| / (half_width + eps) ), half_width=(high-low)/2。
    """
    with torch.no_grad():
        p_hat = head(calib_dataset.X)
    fan = monte_carlo_rollout(predictor, calib_dataset.X, p_hat, error_model,
                              n_samples=n_samples, horizon=horizon,
                              quantiles=((1 - nominal) / 2, 0.5,
                                         (1 + nominal) / 2),
                              generator=generator)
    half = (fan.high - fan.low) / 2.0 + 1e-6
    ratio = ((calib_dataset.Y - fan.median).abs() / half).flatten()
    # 有限样本保守: higher 插值
    return float(torch.quantile(ratio, nominal, interpolation="higher"))


def calibrated_band(fan: TrajectoryFan, inflation: float
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """以中位为中心、按 conformal 因子等比放大扇形半宽, 返回 (lo, hi)。"""
    half = (fan.high - fan.low) / 2.0
    lo = fan.median - inflation * half
    hi = fan.median + inflation * half
    return lo, hi
