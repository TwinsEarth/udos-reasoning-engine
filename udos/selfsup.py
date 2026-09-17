"""
自监督伪信号 (v4.1.0.dev3 / dev4)
=============================================
analogy, not reproduction —— 在冻结主预测器之上, 用既有外挂世界模型/物理守恒/多视角
模块**自产自监督伪标签**, 不引入任何需要标注的真实数据。

伪信号三件 (dev3/dev4 逐步落地):
    * PWM rollout 一致性伪标签 (dev3):
        用 LatentWorldModel.imagine_rollout 想象未来轨迹作伪标签;
        与主 predictor 真实自由 rollout 逐位比较得 step_mse,
        再映射为逐步"一致性置信" w_h = exp(-step_mse_h / median) ∈[0,1]。
        一致性高 => 伪标签可信; 低 => 降权 (供 dev5 伪标签增益 A/B)。
    * 物理守恒 + 多视角一致伪标签 (dev4):
        用 ConservationChecker 检验伪标签轨迹动量/能量守恒,
        用 spatial.multiview_consistency_error 检视角一致; 违反量大 => 降权。

设计纪律 (与全工程一致):
    * 纯前向、确定性、零梯度、不改主 predictor 52191 权重 (只读外挂)。
    * opt-in: 不构造本模块时, 旧推理路径逐位一致。
    * 空 / 非法输入显式 ValueError; 伪标签含 NaN/inf 不静默污染。
    * 日志走 logging_config (stderr), 绝不写 stdout / HTTP 体 / metrics。
    * 第二引擎统一称 GPM。
"""

from __future__ import annotations

import logging

logger = logging.getLogger("udos.selfsup")

from typing import Any, Dict, Optional

import numpy as np
import torch

from .world_model import LatentWorldModel


class PWMConsistencyPseudoLabeler:
    """PWM rollout 一致性伪标签 (dev3)。

    Parameters
    ----------
    predictor:
        已训练 PhysicsPredictor (只读)。
    wm:
        可选已拟合 LatentWorldModel; None 时首次 need 时用 fit_dataset 离线拟合。
    fit_dataset:
        未给 wm 时用于拟合外挂世界模型的小数据集。
    """

    def __init__(self, predictor, wm: Optional[LatentWorldModel] = None,
                 fit_dataset=None, fit_epochs: int = 20) -> None:
        self.predictor = predictor
        self.wm: Optional[LatentWorldModel] = wm
        self._fit_dataset = fit_dataset
        self._fit_epochs = int(fit_epochs)

    def _ensure_wm(self) -> LatentWorldModel:
        if self.wm is None:
            if self._fit_dataset is None:
                raise RuntimeError(
                    "未提供 wm 且未给 fit_dataset, 无法离线拟合 PWM")
            wm = LatentWorldModel(self.predictor)
            # pseudo_label 在 no_grad 下调用; WM 自拟合需短暂 enable_grad
            # (优化器只含 WM 自身参数, predictor 权重梯度恒 0)。
            with torch.enable_grad():
                wm.fit(self._fit_dataset, epochs=self._fit_epochs)
            self.wm = wm
        return self.wm

    @staticmethod
    def _as_window(window: torch.Tensor) -> torch.Tensor:
        w = torch.as_tensor(window, dtype=torch.float32)
        if w.dim() == 2:
            w = w.unsqueeze(0)
        if w.dim() != 3 or w.size(0) != 1:
            raise ValueError("window 需为 [W,6] 或 [1,W,6] (单任务)")
        return w

    @torch.no_grad()
    def pseudo_label(self, window: torch.Tensor, horizon: int = 3,
                     scene_params: Optional[torch.Tensor] = None
                     ) -> Dict[str, Any]:
        """产 PWM 一致性伪标签: 想象轨迹 + 逐步一致性置信。

        返回:
            pseudo_label_states [1,H,6]  想象伪标签轨迹 (第0步锚定真实单步);
            step_mse [H]                 想象 vs 真实 rollout 逐步 MSE;
            consistency [H]              exp(-step_mse/median) ∈[0,1] (伪标签可信权重);
            mean_consistency             一致性均值 (整任务伪标签质量)。
        horizon 须为正整数; 非法输入显式 ValueError。
        """
        w = self._as_window(window)
        if not isinstance(horizon, int) or horizon < 1:
            raise ValueError("horizon 需为 >=1 的整数")
        wm = self._ensure_wm()
        self.predictor.eval()
        roll = wm.imagine_rollout(w, horizon, scene_params=scene_params,
                                  compare_real=True)
        step_mse = roll["step_mse"]                       # [H]
        med = float(step_mse.median().item()) + 1e-9
        consistency = torch.exp(-step_mse / med).clamp(0.0, 1.0)
        states = roll["states"]
        if not bool(torch.isfinite(states).all()):
            raise ValueError("伪标签轨迹含 NaN/inf, 拒绝污染输出")
        return {
            "pseudo_label_states": states,
            "step_mse": step_mse,
            "consistency": consistency,
            "mean_consistency": round(float(consistency.mean().item()), 6),
            "wm_params": wm.n_params,
            "main_predictor_untouched": True,
            "zero_gradient": True,
            "analogy_not_reproduction": True,
        }


class PhysicsMultiviewGate:
    """物理守恒 + 多视角一致伪信号门 (dev4)。

    对一条 PWM 伪标签轨迹 states[1,H,6]=[pos3,vel3] 再施加两道自产物理校验:
        * 守恒门: ConservationChecker 检验动量/能量代理是否守恒;
        * 多视角门: 两个正交视角重投影同一批位置, multiview_consistency_error。
    两道违反量各映射为 [0,1] 门控分 (越守恒/越一致越接近 1), 与 PWM 一致性相乘,
    得最终伪标签可信权重。纯前向、零梯度、确定性。
    """

    def __init__(self, mass: float = 1.0, spring_k: Optional[float] = None,
                 conservation_scale: float = 0.5,
                 multiview_scale: float = 1.0) -> None:
        from .wm_conservation import ConservationChecker
        if conservation_scale <= 0 or multiview_scale <= 0:
            raise ValueError("scale 须为正")
        self.checker = ConservationChecker(mass=mass, spring_k=spring_k)
        self.conservation_scale = float(conservation_scale)
        self.multiview_scale = float(multiview_scale)

    @staticmethod
    def _as_states(states: torch.Tensor) -> torch.Tensor:
        s = torch.as_tensor(states, dtype=torch.float32)
        if s.dim() == 2:
            s = s.unsqueeze(0)
        if s.dim() != 3 or s.size(-1) != 6:
            raise ValueError("states 需为 [H,6] 或 [B,H,6]")
        if s.size(1) < 2:
            raise ValueError("守恒/多视角校验至少需 2 步")
        if not bool(torch.isfinite(s).all()):
            raise ValueError("states 含 NaN/inf")
        return s

    @torch.no_grad()
    def conservation_score(self, states: torch.Tensor) -> Dict[str, float]:
        """守恒门分: score = exp(-(动量违反+能量违反)/scale) ∈[0,1]。"""
        s = self._as_states(states)
        out = self.checker.check(s)
        batch = out["batch"][0]
        viol = batch["momentum_violation"] + batch["energy_violation"]
        score = float(np.exp(-viol / self.conservation_scale))
        return {"conservation_score": round(score, 6),
                "momentum_violation": batch["momentum_violation"],
                "energy_violation": batch["energy_violation"],
                "conserved": batch["conserved"]}

    @torch.no_grad()
    def multiview_score(self, states: torch.Tensor) -> Dict[str, float]:
        """多视角门分: 两正交视角重投影同一批位置的最大不一致 -> exp(-err/scale)。"""
        import numpy as np
        from .spatial import OrthographicView, multiview_consistency_error
        s = self._as_states(states)
        pos = s[0, :, 0:3].cpu().numpy().astype(np.float64)     # [H,3]
        va = OrthographicView(eye=(10.0, 0.0, 0.0), look_at=(0.0, 0.0, 0.0))
        vb = OrthographicView(eye=(0.0, 10.0, 0.0), look_at=(0.0, 0.0, 0.0))
        err = multiview_consistency_error(va, vb, pos)
        score = float(np.exp(-err / self.multiview_scale))
        return {"multiview_score": round(score, 6),
                "multiview_error": round(err, 8)}

    @torch.no_grad()
    def gate(self, states: torch.Tensor,
             pwm_consistency: float = 1.0) -> Dict[str, Any]:
        """综合门控: final_weight = pwm_consistency * conservation * multiview。"""
        cs = self.conservation_score(states)
        ms = self.multiview_score(states)
        pw = float(min(max(pwm_consistency, 0.0), 1.0))
        final = pw * cs["conservation_score"] * ms["multiview_score"]
        return {"pwm_consistency": round(pw, 6),
                "conservation_score": cs["conservation_score"],
                "multiview_score": ms["multiview_score"],
                "final_pseudo_weight": round(float(final), 6),
                **cs, **ms}
