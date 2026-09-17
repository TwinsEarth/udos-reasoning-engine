"""
任务/课程自动生成器 + 可解性自验证器 (v4.1.0 自规划自监督线起点)
=====================================================================
analogy, not reproduction —— 受"环境自造 + 自产验证器"思想启发的合成低维类比实现,
不宣称复现任何真实课程学习 / 自动环境生成算法; 任务为参数化物理窗口->目标滚问题,
课程为按 stage 组织的一组 lesson。

设计纪律 (与全工程 opt-in / 零外挂优先一致):
    * 纯前向、确定性; 只读冻结主 predictor, 不改主权重 (恒 52191)。
    * opt-in: 不构造本模块时, 旧推理路径逐位一致。
    * 空 / 非法输入显式 ValueError; 预测产生 NaN/inf 不静默污染。
    * 日志走 logging_config (logger "udos.curriculum", 默认 stderr),
      绝不写 stdout / HTTP 体 / metrics。
    * 第二引擎统一称 GPM。

本模块三件套:
    * LessonSpec          —— 单节课程的不可变规格 (stage / 难度旋钮 / seed)。
    * CurriculumGenerator —— 按 stage 自动生成一串 lesson 的数据集 (复用 build_parametric_dataset)。
    * SolvabilityVerifier  —— 用冻结 predictor 自身前向对生成任务做可解性自验证
                             (有限性 + 物理界内 + 首步预测健康度), 筛出可解任务。
"""

from __future__ import annotations

import logging

logger = logging.getLogger("udos.curriculum")

import dataclasses
from typing import Any, Dict, List, Optional, Sequence

import torch

from .dynamics import (RAW_DIM, SCENE_PARAM_DIM, ParametricDynamicsDataset,
                       build_parametric_dataset)


# 物理界代理 (合成低维空间): 位置/速度幅值健康上界。越界 => 任务对当前模型不可解。
_POS_BOUND = 20.0
_VEL_BOUND = 20.0


@dataclasses.dataclass(frozen=True)
class LessonSpec:
    """单节课程规格 (不可变; 纯数据, 不持权重)。"""

    stage: int
    n_per_kind: int
    horizon: int
    spread: float          # 场景参数采样幅度扩张系数 (>=1; 越大环境越难)
    seed: int
    n_steps: int = 14
    window: int = 6

    def describe(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


class CurriculumGenerator:
    """按 stage 自动生成参数化物理课程 (任务=窗口->未来 H 步预测问题)。

    复用 build_parametric_dataset 的四类合成运动 (uniform/accel/spring/collision);
    stage 只通过 (n_per_kind, horizon, spread) 三个旋钮调节课程量与难度,
    不引入新运动学 (analogy not reproduction)。

    Parameters
    ----------
    base_seed:
        课程生成主种子; 不同 stage 派生确定性子种子 (base_seed + 1009*stage)。
    window:
        历史窗口长度 (对齐 predictor)。
    """

    def __init__(self, base_seed: int = 0, window: int = 6) -> None:
        if window < 1:
            raise ValueError("window 需 >= 1")
        self.base_seed = int(base_seed)
        self.window = int(window)

    # ------------------------------------------------------------------ #
    # stage -> 难度旋钮 (4.1.0 骨架: 线性插值; dev1 扩展为环境复杂度自动扩张)
    # ------------------------------------------------------------------ #
    def stage_profile(self, stage: int) -> LessonSpec:
        """stage -> LessonSpec。4.1.0 骨架: n_per_kind/horizon/spread 随 stage 线性增长。

        stage<0 显式 ValueError。n_per_kind 夹在 [2, 96], horizon 夹在 [1, 6],
        spread 夹在 [1.0, 2.5]。
        """
        if not isinstance(stage, int) or stage < 0:
            raise ValueError("stage 需为 >=0 的整数")
        n_per_kind = int(min(96, max(2, 8 + stage * 4)))
        horizon = int(min(6, max(1, 2 + stage)))
        spread = float(min(2.5, max(1.0, 1.0 + 0.25 * stage)))
        seed = self.base_seed + 1009 * stage
        return LessonSpec(stage=stage, n_per_kind=n_per_kind,
                          horizon=horizon, spread=spread, seed=seed)

    # ------------------------------------------------------------------ #
    # 生成单节 / 一串课程
    # ------------------------------------------------------------------ #
    def make_lesson(self, stage: int,
                    spec: Optional[LessonSpec] = None
                    ) -> Dict[str, Any]:
        """生成一节课程: {spec, dataset, n_samples, kinds}。

        spec 缺省用 stage_profile(stage)。dataset 为 ParametricDynamicsDataset。
        """
        spec = spec if spec is not None else self.stage_profile(stage)
        ds = build_parametric_dataset(
            n_per_kind=spec.n_per_kind, n_steps=spec.n_steps,
            window=self.window, horizon=spec.horizon, dt=0.5, seed=spec.seed)
        kinds: Dict[str, int] = {}
        for k in ds.kinds:
            kinds[k] = kinds.get(k, 0) + 1
        return {
            "spec": spec, "dataset": ds,
            "n_samples": len(ds), "kinds": kinds,
            "spread": spec.spread,
        }

    def generate(self, stages: Sequence[int]) -> List[Dict[str, Any]]:
        """按 stage 列表依次生成课程; 空 stages 显式 ValueError。"""
        if not stages:
            raise ValueError("stages 不能为空 (至少一节课程)")
        return [self.make_lesson(s) for s in stages]

    # ------------------------------------------------------------------ #
    # v4.1.0.dev1: 课程难度递进 (环境复杂度自动扩展)
    # ------------------------------------------------------------------ #
    def _probe(self, verifier: "SolvabilityVerifier", horizon: int,
               spread: float, seed: int, n_probe: int) -> Dict[str, Any]:
        """以给定 horizon/spread 采一批探针并自验证 (内部用)。"""
        spec = LessonSpec(stage=-1, n_per_kind=max(2, n_probe // 4),
                          horizon=horizon, spread=spread, seed=seed)
        ds = build_parametric_dataset(n_per_kind=spec.n_per_kind,
                                      n_steps=spec.n_steps, window=self.window,
                                      horizon=horizon, dt=0.5, seed=seed)
        return verifier.verify_dataset(ds, max_samples=n_probe, horizon=horizon)

    def auto_progression(self, verifier: "SolvabilityVerifier",
                         stages: Sequence[int], n_probe: int = 16
                         ) -> List[Dict[str, Any]]:
        """课程难度递进: 逐 stage 记录自验证可解率与难度代理。

        环境复杂度随 stage 自动扩张 (horizon/spread 来自 stage_profile)。
        返回每 stage 的 {stage, horizon, spread, solvable_ratio,
        mean_pos_norm, mean_vel_norm, n_checked}。空 stages 显式 ValueError。
        """
        if not stages:
            raise ValueError("stages 不能为空")
        table: List[Dict[str, Any]] = []
        for s in stages:
            prof = self.stage_profile(s)
            r = self._probe(verifier, prof.horizon, prof.spread,
                            self.base_seed + 1009 * s, n_probe)
            table.append({
                "stage": s, "horizon": prof.horizon, "spread": prof.spread,
                "solvable_ratio": r["solvable_ratio"],
                "mean_pos_norm": r["mean_pos_norm_solvable"],
                "mean_vel_norm": r["mean_vel_norm_solvable"],
                "n_checked": r["n_checked"],
            })
        return table

    def auto_expand_horizon(self, verifier: "SolvabilityVerifier",
                            start: int = 2, step: int = 1, max_h: int = 8,
                            solvable_floor: float = 0.8,
                            n_probe: int = 16) -> Dict[str, Any]:
        """环境复杂度自动扩展: 从 start 起按 step 递增 rollout horizon, 用自验证器
        探测每个难度, 直到可解率跌破 solvable_floor 或达到 max_h。

        返回自动找到的难度前沿: {frontier_horizon, floor, schedule[], stopped_by}。
        难度越高 horizon 越长, rollout 越易发散 => 可解率单调 (趋势)下降。
        start<1 / step<1 / floor∉(0,1] 显式 ValueError。
        """
        if start < 1 or step < 1:
            raise ValueError("start 需 >=1, step 需 >=1")
        if not (0.0 < solvable_floor <= 1.0):
            raise ValueError("solvable_floor 需在 (0,1]")
        if max_h < start:
            raise ValueError("max_h 需 >= start")
        spread = 1.0
        schedule: List[Dict[str, Any]] = []
        frontier = start
        h = start
        while h <= max_h:
            r = self._probe(verifier, h, spread,
                            self.base_seed + 777 + h, n_probe)
            schedule.append({"horizon": h,
                             "solvable_ratio": r["solvable_ratio"]})
            if r["solvable_ratio"] < solvable_floor:
                return {"frontier_horizon": frontier, "floor": solvable_floor,
                        "schedule": schedule, "stopped_by": "floor"}
            frontier = h
            h += step
        return {"frontier_horizon": frontier, "floor": solvable_floor,
                "schedule": schedule, "stopped_by": "max_h"}

    def describe(self) -> Dict[str, Any]:
        return {"base_seed": self.base_seed, "window": self.window,
                "analogy_not_reproduction": True, "zero_gradient": True}


class SolvabilityVerifier:
    """用冻结主 predictor 自身前向对生成任务做可解性自验证 (自产验证器)。

    一个任务 (窗口) 判为"可解", 当且仅当:
        1) predictor.predict_next 输出全有限 (无 NaN/inf);
        2) 首步预测位置/速度幅值在物理界内 (pos|.|<=_POS_BOUND, vel|.|<=_VEL_BOUND);
        3) H 步自由 rollout (若 horizon>1) 全程有限且不发散 (步间 L2 不爆炸)。

    纯前向、零梯度; 只读 predictor, 不回传改权重。空/非法输入显式 ValueError。
    """

    def __init__(self, predictor,
                 pos_bound: float = _POS_BOUND,
                 vel_bound: float = _VEL_BOUND,
                 divergence_growth: float = 8.0) -> None:
        if pos_bound <= 0 or vel_bound <= 0:
            raise ValueError("pos_bound/vel_bound 须为正")
        if divergence_growth <= 1.0:
            raise ValueError("divergence_growth 须 >1")
        self.predictor = predictor
        self.pos_bound = float(pos_bound)
        self.vel_bound = float(vel_bound)
        self.divergence_growth = float(divergence_growth)

    # ------------------------------------------------------------------ #
    # 输入守卫
    # ------------------------------------------------------------------ #
    @staticmethod
    def _as_window(window: torch.Tensor) -> torch.Tensor:
        w = torch.as_tensor(window, dtype=torch.float32)
        if w.dim() == 2:
            w = w.unsqueeze(0)
        if w.dim() != 3:
            raise ValueError("window 需为 [W,RAW] 或 [B,W,RAW]")
        if w.size(0) != 1:
            raise ValueError("自验证器按单任务验证, window 批维须为 1")
        return w

    # ------------------------------------------------------------------ #
    # 单任务可解性
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def verify(self, window: torch.Tensor,
               scene_params: Optional[torch.Tensor] = None,
               horizon: int = 1) -> Dict[str, Any]:
        """验证单任务窗口是否可解。返回 {solvable, finite, within_bounds, ...}。"""
        w = self._as_window(window)
        if not bool(torch.isfinite(w).all()):
            raise ValueError("window 含 NaN/inf")
        if not isinstance(horizon, int) or horizon < 1:
            raise ValueError("horizon 需为 >=1 的整数")
        sp = scene_params
        if sp is not None:
            sp = torch.as_tensor(sp, dtype=torch.float32)
            if sp.dim() == 1:
                sp = sp.unsqueeze(0)
        self.predictor.eval()

        nxt = self.predictor.predict_next(w, scene_params=sp)
        finite = bool(torch.isfinite(nxt).all())
        if not finite:
            return {"solvable": False, "finite": False, "within_bounds": False,
                    "reason": "predict_next 产生非有限输出 (NaN/inf)"}

        pos_n = nxt[0, 0:3]
        vel_n = nxt[0, 3:6]
        pos_norm = float(pos_n.norm().item())
        vel_norm = float(vel_n.norm().item())
        within_bounds = (pos_norm <= self.pos_bound
                         and vel_norm <= self.vel_bound)

        divergent = False
        if horizon > 1:
            traj = self.predictor.rollout(w, horizon, scene_params=sp)  # [1,H,RAW]
            if not bool(torch.isfinite(traj).all()):
                divergent = True
                reason = f"rollout horizon={horizon} 含非有限步"
            else:
                step_l2 = traj[0].norm(dim=-1)                     # [H]
                # 步间幅值相对首步的最大倍数 > growth => 判发散
                base = float(step_l2[0].item()) + 1e-9
                worst = float((step_l2 / base).max().item())
                divergent = worst > self.divergence_growth
                reason = (f"rollout 幅值发散 (worst_growth={worst:.2f})"
                          if divergent else "")
        else:
            reason = ""

        solvable = finite and within_bounds and not divergent
        return {
            "solvable": solvable,
            "finite": finite,
            "within_bounds": within_bounds,
            "rollout_divergent": divergent,
            "pos_norm": round(pos_norm, 6),
            "vel_norm": round(vel_norm, 6),
            "reason": reason if not solvable else "",
        }

    # ------------------------------------------------------------------ #
    # 批量可解性自验证 (课程筛选)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def verify_dataset(self, dataset: ParametricDynamicsDataset,
                       max_samples: int = 64, horizon: int = 1
                       ) -> Dict[str, Any]:
        """对课程数据集逐样本自验证, 返回可解率与筛选后的索引。

        max_samples<=0 显式 ValueError; 实际验证数 = min(max_samples, len(dataset))。
        """
        if max_samples <= 0:
            raise ValueError("max_samples 需 >0")
        n = min(max_samples, len(dataset))
        if n == 0:
            raise ValueError("dataset 为空, 无法自验证")
        P = getattr(dataset, "P", None)
        solvable_idx: List[int] = []
        unsolvable_idx: List[int] = []
        pos_b, vel_b, fin = 0.0, 0.0, 0
        for i in range(n):
            w = dataset.X[i:i + 1]
            sp = P[i:i + 1] if P is not None else None
            r = self.verify(w, scene_params=sp, horizon=horizon)
            if r["solvable"]:
                solvable_idx.append(i)
                pos_b += r["pos_norm"]
                vel_b += r["vel_norm"]
                fin += 1
            else:
                unsolvable_idx.append(i)
        m = max(fin, 1)
        return {
            "n_checked": n,
            "n_solvable": fin,
            "solvable_ratio": round(fin / n, 6),
            "mean_pos_norm_solvable": round(pos_b / m, 6),
            "mean_vel_norm_solvable": round(vel_b / m, 6),
            "solvable_indices": solvable_idx,
            "unsolvable_indices": unsolvable_idx,
            "main_predictor_untouched": True,
            "zero_gradient": True,
        }
