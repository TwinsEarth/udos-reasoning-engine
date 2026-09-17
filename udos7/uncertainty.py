"""v7 可信度：按 α 真分水平的 split conformal + 参数不确定性蒙特卡洛扇形。

根治旧版 conformal_by_alpha 的缺陷（名义 80/90/95 经验覆盖全为同一常数、α 被忽略）：
- 非一致性分数按**每个 α 单独取分位数**（有限样本修正），α 不同带宽必须不同；
- 同时给边际（标量池化）与逐步（每个 horizon 步）两套带宽；
- 校准集来自独立轨迹 seed（CALIB_SEED），不参与梯度，也不与 val/test 混；
- 校准概率（区间/扇形覆盖率）与任何 entropy“置信度”严格分离——确定性核不产概率。

蒙特卡洛参数扇形：对盲路径下的隐藏参数，用**校准集参数残差的经验自助采样**
（保留槽位相关性）扰动估计值，M 次 rollout 得 p10/p50/p90；不可观测槽位
（碰撞 v2、弹簧 ω）残差大 => 扇形自然变宽，覆盖率在 test 上实测。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from .dynamics import TrajectoryDataset
from .model import WorldModelCore


def _conformal_quantile(scores: torch.Tensor, alpha: float) -> float:
    """有限样本修正的 split-conformal 分位数（scores 为一维非负）。"""
    n = scores.numel()
    q_level = min(1.0, (1.0 - alpha) * (n + 1) / n)
    return float(torch.quantile(scores.float(), q_level))


@dataclass
class ConformalBand:
    """按 α 存带宽；marginal 标量 / per_step 每步。"""
    alpha: float
    q_marginal: float
    q_per_step: torch.Tensor          # [H]

    def interval(self, point: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """point [B,H,6] -> lower/upper（逐步带宽，按状态维广播）。"""
        q = self.q_per_step.view(1, -1, 1).to(point.device)
        return point - q, point + q


class ConformalCalibrator:
    """对一种条件模式（oracle / blind）校准多步预测区间。"""

    def __init__(self, horizon: int):
        self.horizon = horizon
        self.bands: Dict[float, ConformalBand] = {}

    @torch.no_grad()
    def fit(self, model: WorldModelCore, calib: TrajectoryDataset,
            use_explicit: bool, alphas: Tuple[float, ...] = (0.2, 0.1, 0.05),
            batch: int = 256) -> "ConformalCalibrator":
        model.eval()
        if self.horizon > calib.horizon:
            raise ValueError(
                f"校准视界 {self.horizon} 超过校准集视界 {calib.horizon}")
        resid = []
        for s in range(0, len(calib), batch):
            x, P, Y = calib.X[s:s + batch], calib.P[s:s + batch], calib.Y[s:s + batch]
            exp = P if use_explicit else None
            pred = model.rollout(x, self.horizon, explicit=exp)
            # 校准视界可短于数据集视界（服务支持任意 horizon）：对齐前 self.horizon 步
            resid.append((pred - Y[:, :self.horizon]).abs())
        R = torch.cat(resid, 0)                  # [N,H,6]
        scalar = R.reshape(-1)
        for a in alphas:
            qm = _conformal_quantile(scalar, a)
            qs = torch.tensor([_conformal_quantile(R[:, h, :].reshape(-1), a)
                               for h in range(self.horizon)])
            self.bands[a] = ConformalBand(a, qm, qs)
        return self

    @torch.no_grad()
    def predict_interval(self, model: WorldModelCore, window: torch.Tensor,
                         horizon: int, alpha: float,
                         explicit: Optional[torch.Tensor] = None
                         ) -> Dict[str, torch.Tensor]:
        if alpha not in self.bands:
            raise KeyError(f"未校准 alpha={alpha}，现有 {sorted(self.bands)}")
        point = model.rollout(window, horizon, explicit=explicit)
        lo, up = self.bands[alpha].interval(point)
        return {"point": point, "lower": lo, "upper": up}

    def widths_depend_on_alpha(self) -> bool:
        """探针：α 越小（名义覆盖越高），带宽必须越大。"""
        qs = sorted(((a, b.q_marginal) for a, b in self.bands.items()),
                    key=lambda t: -t[0])   # α 降序 0.2 -> 0.05
        widths = [q for _, q in qs]
        return widths[0] < widths[-1]


@torch.no_grad()
def empirical_coverage(lower: torch.Tensor, upper: torch.Tensor,
                       truth: torch.Tensor) -> Dict[str, object]:
    inside = ((truth >= lower) & (truth <= upper)).float()
    H = truth.size(1)
    return {
        "marginal": round(float(inside.mean()), 4),
        "per_step": [round(float(inside[:, h].mean()), 4) for h in range(H)],
    }


@dataclass
class ParamFan:
    p10: torch.Tensor
    p50: torch.Tensor
    p90: torch.Tensor
    draws: int


class MonteCarloParamFan:
    """盲路径参数不确定性扇形：校准集参数残差经验自助 + M 次 rollout。

    裸扇形只刻画**参数**不确定性，覆盖不了模型结构/自回归累积误差，因此在
    校准集上再做一层乘法式 split-conformal 膨胀（带绝对地板 eps），把结构
    误差吸收进带宽，使 [p10,p90] 中心带在 test 上达到名义覆盖。
    """

    def __init__(self, draws: int = 64, seed: int = 7, eps: float = 0.02):
        self.draws = draws
        self.eps = eps
        self.gen = torch.Generator().manual_seed(seed)
        self.residuals: Optional[torch.Tensor] = None   # [Ncal,4]
        self.inflate: Optional[torch.Tensor] = None     # [H]
        self.nominal: float = 0.8

    @torch.no_grad()
    def fit(self, model: WorldModelCore, calib: TrajectoryDataset,
            batch: int = 256) -> "MonteCarloParamFan":
        phats, trues = [], []
        for s in range(0, len(calib), batch):
            sc = model.scene(calib.X[s:s + batch], None)
            phats.append(sc.params_hat)
            trues.append(calib.P[s:s + batch])
        self.residuals = (torch.cat(trues, 0) - torch.cat(phats, 0))
        return self

    @torch.no_grad()
    def raw_fan(self, model: WorldModelCore, window: torch.Tensor,
                horizon: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.residuals is None:
            raise RuntimeError("ParamFan 未 fit")
        B = window.size(0)
        ncal = self.residuals.size(0)
        outs = []
        p_hat = model.scene(window, None).params_hat
        for _ in range(self.draws):
            idx = torch.randint(0, ncal, (B,), generator=self.gen)
            p_draw = p_hat + self.residuals[idx]
            outs.append(model.rollout(window, horizon, explicit=p_draw))
        A = torch.stack(outs, 0)                  # [M,B,H,6]
        q = torch.quantile(A, torch.tensor([0.1, 0.5, 0.9]), dim=0)
        return q[0], q[1], q[2]

    @torch.no_grad()
    def calibrate_inflation(self, model: WorldModelCore,
                            calib: TrajectoryDataset, horizon: int,
                            nominal: float = 0.8,
                            max_n: int = 512) -> "MonteCarloParamFan":
        """两步校准：
        1) 逐步乘法膨胀因子，让参数扇形贴合 calib 残差（标量池化）；
        2) 残差 conformal 地板 q_floor[H]——参数扇形只刻画参数不确定性，
           覆盖不了自回归结构误差；取与残差带的包络，保证覆盖率不低于
           纯残差 conformal（后者有边际覆盖保证）。
        """
        self.nominal = nominal
        alpha = 1.0 - nominal
        n = min(len(calib), max_n)
        X, Y = calib.X[:n], calib.Y[:n]
        lo, p50, up = self.raw_fan(model, X, horizon)
        half = ((up - lo) / 2).clamp_min(0.0)
        score = (Y - p50).abs() / (half + self.eps)       # [n,H,6]
        self.inflate = torch.tensor([
            _conformal_quantile(score[:, h, :].reshape(-1), alpha)
            for h in range(horizon)])
        point = model.rollout(X, horizon, explicit=None)  # 盲路径点预测
        rscore = (Y - point).abs()                        # [n,H,6]
        self.q_floor = torch.tensor([
            _conformal_quantile(rscore[:, h, :].reshape(-1), alpha)
            for h in range(horizon)])
        return self

    @torch.no_grad()
    def sample(self, model: WorldModelCore, window: torch.Tensor,
               horizon: int) -> ParamFan:
        if self.inflate is None or self.inflate.numel() != horizon:
            raise RuntimeError("sample 前需 calibrate_inflation(同 horizon)")
        lo0, p50, up0 = self.raw_fan(model, window, horizon)
        half = ((up0 - lo0) / 2).clamp_min(0.0)
        q = self.inflate.view(1, -1, 1)
        band = q * (half + self.eps)
        lower = p50 - band
        upper = p50 + band
        if getattr(self, "q_floor", None) is not None:
            # 与残差 conformal 带取包络：覆盖率不低于残差带
            qf = self.q_floor.view(1, -1, 1)
            lower = torch.minimum(lower, p50 - qf)
            upper = torch.maximum(upper, p50 + qf)
        return ParamFan(lower, p50, upper, self.draws)
