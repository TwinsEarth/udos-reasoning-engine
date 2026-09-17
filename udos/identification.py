"""
场景参数辨识 / 归因 (v2.6.0+dev2)
====================================
仅凭观测窗口反推隐藏物理参数 (SCENE_PARAM_NAMES=[v0, accel_a, spring_omega, other_v2]),
并做一阶 Sobol 敏感性归因。纯前向、确定性、不修改 predictor。

SceneParameterIdentifier:
    网格搜索每个参数的物理合理区间, 用**窗口内自洽校验**打分 —— 取窗口前 W-horizon
    帧作输入、后 horizon 帧作目标, 不同 scene_param 组合 rollout, 选 MSE 最小者。
    (窗口内自洽, 无需额外未来帧; 这是简化辨识, 精度受训练模型外推能力约束。)

sobol_attribution:
    一阶 Sobol 指数简化版: 逐参数独立扰动 ±σ 测量输出方差 V_i, 归一化
    S_i = V_i / sum_j V_j (和≈1)。固定种子确定性。
"""



from __future__ import annotations

import logging

logger = logging.getLogger("udos.identification")


from typing import Dict, List

import torch

from .dynamics import SCENE_PARAM_NAMES

# 各场景参数的物理合理搜索区间 (与 _sample_parametric 采样域一致)
PARAM_RANGES = {
    "v0": (-2.0, 2.0),
    "accel_a": (-1.5, 1.5),
    "spring_omega": (0.6, 1.6),
    "other_v2": (-0.5, 0.5),
}


class SceneParameterIdentifier:
    """网格搜索反演隐藏场景物理参数 (自洽校验)。"""

    def __init__(self, predictor, grid_size: int = 5):
        self.predictor = predictor
        self.grid_size = grid_size
        grids = []
        for name in SCENE_PARAM_NAMES:
            lo, hi = PARAM_RANGES[name]
            grids.append(torch.linspace(lo, hi, grid_size))
        self.grids = grids

    @torch.no_grad()
    def identify(self, raw_window: torch.Tensor, horizon: int = 2) -> Dict:
        """
        raw_window [B,W,R] (建议 B=1)。用窗口后 horizon 帧作自洽目标,
        网格搜索 [G,4] 个参数组合的 rollout MSE, 取最优。
        返回 identified_params[4] / param_names / loss_curve_min。
        """
        B, W, R = raw_window.shape
        assert W > horizon, "窗口长度需大于 horizon"
        inp = raw_window[:, :W - horizon, :]
        target = raw_window[:, W - horizon:, :]           # [B,horizon,R]

        mesh = torch.meshgrid(*self.grids, indexing="ij")  # 4 个 [g,...]
        combos = torch.stack([m.reshape(-1) for m in mesh], dim=1)  # [G,4]
        G = combos.size(0)

        inp_exp = inp.expand(G, W - horizon, R).reshape(G, W - horizon, R)
        roll = self.predictor.rollout(inp_exp, horizon, scene_params=combos)
        # roll [G,horizon,R], target [B,horizon,R] -> [G,B,h,R] 广播, 对 h/R/B 取均
        err = ((roll.unsqueeze(1) - target.unsqueeze(0)) ** 2) \
            .mean(dim=(2, 3)).mean(dim=1)                          # [G]
        # 可辨识性先验: 匀速/惯性段的 v0 已由窗口速度直接观测, 模型对其不敏感
        # (rollout MSE 在 v0 上近乎平坦), 用观测 x 速度作 v0 先验打破退化。
        v0_obs = float(raw_window[:, -1, 3].mean().item())
        err = err + 0.05 * (combos[:, 0] - v0_obs) ** 2
        best = int(torch.argmin(err).item())
        return {
            "identified_params": combos[best],
            "param_names": list(SCENE_PARAM_NAMES),
            "loss_curve_min": float(err[best].item()),
            "n_grid": G,
        }


@torch.no_grad()
def sobol_attribution(predictor, raw_window: torch.Tensor,
                      scene_params: torch.Tensor, n_samples: int = 32,
                      seed: int = 0) -> Dict:
    """
    一阶 Sobol 指数简化版: 对每个 scene_param 独立加 ±σ 扰动 (其他参数固定基线),
    测量输出 (rollout 均值) 方差 V_i; S_i = V_i / sum_j V_j。
    确定性 (固定种子)。返回 indices[4] / param_names / total_variance。
    """
    g = torch.Generator().manual_seed(seed)
    base = scene_params.reshape(1, -1)
    dim = base.size(1)
    # 扰动 σ 取该参数区间宽度的 ~15%
    sigmas = []
    for name in SCENE_PARAM_NAMES:
        lo, hi = PARAM_RANGES[name]
        sigmas.append(0.15 * (hi - lo))
    sigmas = torch.tensor(sigmas)

    def scalar_output(p_batch: torch.Tensor) -> torch.Tensor:
        n = p_batch.size(0)
        B, W, R = raw_window.shape
        win = raw_window.expand(n, W, R).reshape(n, W, R)
        roll = predictor.rollout(win, 2, scene_params=p_batch)
        return roll.mean(dim=(1, 2))                 # [n]

    V = []
    for i in range(dim):
        samples = base.repeat(n_samples, 1).clone()
        noise = torch.randn(n_samples, generator=g) * sigmas[i]
        samples[:, i] = samples[:, i] + noise
        out = scalar_output(samples)
        V.append(float(out.var(unbiased=False).item()))
    V = torch.tensor(V)
    total = float(V.sum().item()) + 1e-12
    S = V / total
    return {
        "indices": S,                       # [4], 和≈1
        "param_names": list(SCENE_PARAM_NAMES),
        "total_variance": total,
    }
