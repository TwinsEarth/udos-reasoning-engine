"""
深度集成不确定性 (v2.4.1, opt-in)
================================
多个**同架构、多种子**的 PhysicsPredictor 组成深度集成 (Deep Ensemble, Lakshminarayanan
et al. 2017): 用成员间预测分歧近似认知不确定性 (epistemic uncertainty)。

设计:
  - DeepEnsemble 仅包装已有 PhysicsPredictor 列表, 不重写前向逻辑;
  - predict_next/rollout 返回 dict {mean, variance, members[N,B,...]};
  - 方差为成员间**无偏样本方差** (ddof=1); N=1 时方差恒 0、均值即该成员输出,
    逐位等价于单模型 (向后兼容锚点);
  - 零额外可学参数, 纯前向; 训练由调用方对每个成员独立进行 (见测试)。

术语: 第二组件一律 GPM。
"""



from __future__ import annotations

import logging

logger = logging.getLogger("udos.ensemble")


import dataclasses
from typing import Callable, Dict, List, Optional

import torch

from .ctm_engine import CTMConfig
from .training import PhysicsPredictor, set_seed


class DeepEnsemble(torch.nn.Module):
    """N 个同架构 PhysicsPredictor 的集成, 提供均值/方差/逐成员预测。"""

    def __init__(self, members: List[PhysicsPredictor]):
        super().__init__()
        if not members:
            raise ValueError("集成至少需要 1 个成员")
        self.members = torch.nn.ModuleList(members)

    # ---------------- 构造 ---------------- #
    @classmethod
    def create(cls, ctm_config_factory: Callable[[], CTMConfig],
               n: int = 5, base_seed: int = 0,
               scene_param_dim: Optional[int] = None) -> "DeepEnsemble":
        """
        用工厂函数造 N 个**同架构、多种子**成员。factory 每次返回新的 CTMConfig
        (PhysicsPredictor 会就地改写 certainty_threshold, 故必须每次新 config)。
        """
        if n < 1:
            raise ValueError("n 必须 >= 1")
        members: List[PhysicsPredictor] = []
        for i in range(n):
            set_seed(base_seed + i)
            members.append(PhysicsPredictor(ctm_config_factory(),
                                            scene_param_dim=scene_param_dim))
        return cls(members)

    @property
    def n_members(self) -> int:
        return len(self.members)

    # ---------------- 前向聚合 ---------------- #
    @staticmethod
    def _aggregate(stacked: torch.Tensor) -> Dict[str, torch.Tensor]:
        """stacked: [N, ...] -> {mean, variance, members}。方差无偏 (ddof=1); N=1 方差=0。"""
        n = stacked.size(0)
        mean = stacked.mean(dim=0)
        if n == 1:
            var = torch.zeros_like(mean)
        else:
            var = stacked.var(dim=0, unbiased=True)
        return {"mean": mean, "variance": var, "members": stacked}

    @torch.no_grad()
    def predict_next(self, raw_seq: torch.Tensor,
                     scene_params: Optional[torch.Tensor] = None
                     ) -> Dict[str, torch.Tensor]:
        """单步下一状态集成预测: {mean, variance, members[N,B,R]}。"""
        self.eval()
        outs = [m.predict_next(raw_seq, scene_params=scene_params)
                for m in self.members]
        return self._aggregate(torch.stack(outs, dim=0))

    @torch.no_grad()
    def rollout(self, raw_window: torch.Tensor, horizon: int,
                scene_params: Optional[torch.Tensor] = None
                ) -> Dict[str, torch.Tensor]:
        """多步自由 rollout 集成: {mean, variance, members[N,B,H,R]}。"""
        self.eval()
        outs = [m.rollout(raw_window, horizon, scene_params=scene_params)
                for m in self.members]
        return self._aggregate(torch.stack(outs, dim=0))

    # ---------------- v2.4.7: 集成方差缩放 conformal 区间 ---------------- #
    @torch.no_grad()
    def predict_interval(self, raw_window: torch.Tensor, horizon: int,
                         scene_params: Optional[torch.Tensor] = None,
                         quantile_member: int = 0) -> Dict[str, torch.Tensor]:
        """
        集成 conformal 区间: 中值=集成 rollout 均值; 半宽 = sqrt(q^2 + v),
        其中 q 为某成员 (默认第 0 个) 的逐维残差 conformal 半宽, v 为集成成员间
        方差 (对 batch 取均值)。N=1 时 v=0 => 半宽=q, 逐位退化为普通区间。
        """
        base = self.members[quantile_member]
        if getattr(base, "residual_quantiles", None) is None:
            raise RuntimeError("集成成员未挂载残差半宽, 请先 attach_calibration")
        roll = self.rollout(raw_window, horizon, scene_params=scene_params)
        mean, var = roll["mean"], roll["variance"]     # [B,H,R], [B,H,R]
        B, H, R = mean.shape
        lowers, uppers, widths = [], [], []
        for h in range(H):
            q = base.residual_quantiles[min(h, len(base.residual_quantiles) - 1)]
            q = q.view(-1)                              # [R]
            v_bar = var[:, h, :].mean(dim=0)            # [R] 批内平均认知方差
            half = torch.sqrt(q.clamp_min(0.0) ** 2 + v_bar.clamp_min(0.0))
            lowers.append(mean[:, h, :] - half)
            uppers.append(mean[:, h, :] + half)
            widths.append(float(half.mean()))
        return {
            "median": mean,
            "lower": torch.stack(lowers, dim=1),
            "upper": torch.stack(uppers, dim=1),
            "half_width_by_step": [round(w, 6) for w in widths],
            "variance": var,
        }

    # ---------------- 轻量存/载 (本模块自包含; 正式 save_ensemble 在 persistence 后续版本) ------------- #
    def save(self, path: str) -> None:
        """存成员 state_dict + 架构配置 (可复载)。"""
        m0 = self.members[0]
        bundle = {
            "kind": "DeepEnsemble",
            "n": self.n_members,
            "raw_dim": m0.raw_dim,
            "scene_param_dim": m0.scene_param_dim,
            "ctm_config": dataclasses.asdict(m0.ctm.cfg),
            "members": [m.state_dict() for m in self.members],
        }
        torch.save(bundle, path)

    @classmethod
    def load(cls, path: str, map_location: str = "cpu") -> "DeepEnsemble":
        bundle = torch.load(path, map_location=map_location, weights_only=False)
        if bundle.get("kind") != "DeepEnsemble":
            raise ValueError("不是 DeepEnsemble checkpoint")
        members = []
        for sd in bundle["members"]:
            m = PhysicsPredictor(CTMConfig(**bundle["ctm_config"]),
                                 raw_dim=bundle["raw_dim"],
                                 scene_param_dim=bundle.get("scene_param_dim"))
            m.load_state_dict(sd)
            m.eval()
            members.append(m)
        return cls(members)
