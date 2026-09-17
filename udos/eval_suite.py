"""
UDOS 五维评测套件 (v3.0.0.dev5)
==================================
**UDOS 内部基准, 不是 PhysBrain 榜单分数。**

参照 PhysBrain EvalKit 的五维度思想, 为 UDOS 自身建多维评测 (analogy, not
reproduction): 所有维度都用 UDOS 自己的合成参数化动力学小模型可复算的代理指标,
与 PhysBrain 外部公开数字**严格分开**, 不得互相引用或比较。

五维度 (均为 UDOS 内部代理, 分数 0-100, 越高越好):
    ① 视觉空间感知代理  = 状态重建精度 (predictor 单步 MSE 越低分越高);
    ② 多视角空间理解代理 = scene_params 辨识精度 (identified vs 真实参数归一误差);
    ③ 具身认知与规划代理 = policy 动作优选质量 (已知最优候选被 MPC 选中的比例);
    ④ 空间指向与可供性代理 = affordance 打分准确率 (最近可达物体被选为 best_part);
    ⑤ 视觉轨迹推理代理  = rollout 轨迹精度 (多步 rollout MSE 越低分越高)。

设计纪律:
    * 纯前向、确定性、不修改 predictor;
    * 分数严格 [0,100], 可复现;
    * 每个维度独立方法, 空模型 (predictor=None) 守卫;
    * 不下载外部数据 / CPU-only。

第二引擎一律称 GPM。
"""



from __future__ import annotations

import logging

logger = logging.getLogger("udos.eval_suite")


from typing import Any, Dict, List, Optional

import torch

from .dynamics import build_parametric_dataset, SCENE_PARAM_NAMES
from .identification import SceneParameterIdentifier, PARAM_RANGES
from .policy import MPCActionSelector
from .affordance import AffordanceScorer


# 显式标注: 这是 UDOS 内部基准, 与 PhysBrain 外部数字严格分开
IS_INTERNAL_BENCHMARK = True
NOT_PHYSBRAIN_LEADERBOARD = True


def _clamp0100(x: float) -> float:
    return float(max(0.0, min(100.0, x)))


def _mse_to_score(mse: float) -> float:
    """MSE -> 0-100 分: 100/(1+mse) 型单调衰减, 确定性、有界。"""
    return _clamp0100(100.0 / (1.0 + max(float(mse), 0.0)))


class FiveDimensionEvaluator:
    """UDOS 内部五维评测器 (非 PhysBrain 榜单分数)。

    Parameters
    ----------
    predictor: 已训练 PhysicsPredictor。
    seed:      评测固定种子 (可复现)。
    n_per_kind: 评测合成数据规模 (小)。
    grid_size: scene_params 辨识网格粗细 (越小越快)。
    """

    DIMENSIONS: List[str] = [
        "visual_spatial_perception",
        "multiview_spatial_understanding",
        "embodied_cognition_planning",
        "spatial_affordance",
        "visual_trajectory_reasoning",
    ]

    def __init__(self, predictor, seed: int = 2025, n_per_kind: int = 16,
                 grid_size: int = 3) -> None:
        if predictor is None:
            raise ValueError("FiveDimensionEvaluator 需要已训练 predictor")
        self.predictor = predictor
        self.seed = int(seed)
        self.n_per_kind = int(n_per_kind)
        self.grid_size = int(grid_size)
        self._ds = None

    # ------------------------------------------------------------------ #
    def _dataset(self):
        if self._ds is None:
            torch.manual_seed(self.seed)
            self._ds = build_parametric_dataset(
                n_per_kind=self.n_per_kind, n_steps=14, window=6,
                horizon=4, dt=0.5, seed=self.seed)
        return self._ds

    # ① 视觉空间感知代理 = 状态重建精度
    def dim1_state_reconstruction(self) -> float:
        ds = self._dataset()
        pred = self.predictor.predict_next(ds.X, scene_params=ds.P)
        mse = float(((pred - ds.Y[:, 0, :]) ** 2).mean())
        return _mse_to_score(mse)

    # ② 多视角空间理解代理 = scene_params 辨识精度
    def dim2_scene_param_id(self) -> float:
        ds = self._dataset()
        ident = SceneParameterIdentifier(self.predictor,
                                         grid_size=self.grid_size)
        scores = []
        for i in range(min(8, len(ds))):
            res = ident.identify(ds.X[i:i+1], horizon=2)
            est = res["identified_params"]
            true = ds.P[i]
            errs = []
            for j, name in enumerate(SCENE_PARAM_NAMES):
                lo, hi = PARAM_RANGES[name]
                width = max(hi - lo, 1e-6)
                errs.append(abs(float(est[j]) - float(true[j])) / width)
            scores.append(1.0 - sum(errs) / len(errs))
        return _clamp0100(100.0 * max(0.0, sum(scores) / len(scores)))

    # ③ 具身认知与规划代理 = policy 动作优选质量
    def dim3_policy_planning(self) -> float:
        ds = self._dataset()
        n_correct = 0
        n_total = 0
        for i in range(min(8, len(ds))):
            w, p = ds.X[i:i+1], ds.P[i:i+1]
            # MPC horizon=2, 参考只取前 2 步以匹配 rollout 形状 [1,2,6]
            ref = ds.Y[i:i+1, :2, :]
            mpc = MPCActionSelector(self.predictor, horizon=2, reference=ref)
            # 候选 0: 零扰动 (贴合真值) 应最优; 候选 1: 大扰动
            out = mpc.select(w, scene_params=p,
                             candidate_actions=[{},
                                                {"state_perturbation": [5.0] * 6}])
            if not out["no_valid_action"] and out["best_index"] == 0:
                n_correct += 1
            n_total += 1
        return _clamp0100(100.0 * n_correct / max(n_total, 1))

    # ④ 空间指向与可供性代理 = affordance 打分准确率
    def dim4_affordance(self) -> float:
        # 合成: 机器人在原点, 一个物体近 (可达) 一个远 (不可达)
        scorer = AffordanceScorer(reach_radius=1.0)
        state = torch.zeros(1, 6)
        # 近物体在 0.3, 远物体在 5.0
        objects = torch.tensor([[[0.3, 0.0, 0.0, 0, 0, 0],
                                 [5.0, 0.0, 0.0, 0, 0, 0]]], dtype=torch.float32)
        res = scorer.score(state, objects)
        best = int(res["best_part"][0])
        return _clamp0100(100.0 if best == 0 else 0.0)

    # ⑤ 视觉轨迹推理代理 = rollout 轨迹精度
    def dim5_rollout_trajectory(self) -> float:
        ds = self._dataset()
        H = ds.Y.size(1)
        roll = self.predictor.rollout(ds.X, H, scene_params=ds.P)
        mse = float(((roll - ds.Y) ** 2).mean())
        return _mse_to_score(mse)

    # ------------------------------------------------------------------ #
    def evaluate(self) -> Dict[str, float]:
        """返回五维分数 dict (均 0-100)。"""
        return {
            "visual_spatial_perception": round(self.dim1_state_reconstruction(), 2),
            "multiview_spatial_understanding": round(self.dim2_scene_param_id(), 2),
            "embodied_cognition_planning": round(self.dim3_policy_planning(), 2),
            "spatial_affordance": round(self.dim4_affordance(), 2),
            "visual_trajectory_reasoning": round(self.dim5_rollout_trajectory(), 2),
        }

    # ------------------------------------------------------------------ #
    def composite_score(self, scores: Optional[Dict[str, float]] = None,
                       weights: Optional[Dict[str, float]] = None) -> float:
        """五维加权平均 (默认等权)。权重自动归一化 (和不必为 1)。

        Parameters
        ----------
        scores:   可选; 不传则现场 evaluate()。
        weights:  可选; {dimension: weight}; 默认五维等权。权重和<=0 抛错。
        """
        if scores is None:
            scores = self.evaluate()
        if weights is None:
            weights = {d: 1.0 for d in self.DIMENSIONS}
        missing_w = [d for d in self.DIMENSIONS if d not in weights]
        if missing_w:
            raise ValueError(f"权重缺少维度: {missing_w}")
        missing_s = [d for d in self.DIMENSIONS if d not in scores]
        if missing_s:
            raise ValueError(f"分数缺少维度: {missing_s}")
        wsum = float(sum(weights[d] for d in self.DIMENSIONS))
        if wsum <= 0:
            raise ValueError("权重和必须 > 0")
        total = 0.0
        for d in self.DIMENSIONS:
            total += (weights[d] / wsum) * float(scores[d])
        return _clamp0100(total)

    def metadata(self) -> Dict[str, Any]:
        return {
            "is_internal_benchmark": IS_INTERNAL_BENCHMARK,
            "not_physbrain_leaderboard": NOT_PHYSBRAIN_LEADERBOARD,
            "dimensions": list(self.DIMENSIONS),
            "note": "UDOS 内部基准, 与 PhysBrain 外部数字严格分开",
        }
