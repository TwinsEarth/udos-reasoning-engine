"""世界模型调度器 —— WMScheduler (v3.8.0.dev1, 全域调度线)
======================================================================
为多体场景**分配世界模型想象预算**: 谁需要想象、想象几步、何时回退真实预测。
**复用既有 LatentWorldModel** (3.6 线 PWM), 本模块只做确定性预算分配, 不加任何权重。

分工 (纯推理外挂, 零梯度, 确定性, opt-in):
    * allocate(scene): 按优先级贪婪分配想象步数 horizon。
        - 每个智能体至少 base_horizon(=1) 步; horizon=1 即**回退真实单步预测**
          (LatentWorldModel.imagine(horizon=1) 已逐位锚定 predictor.predict_next)。
        - 总预算 total_imagination_budget 为全部智能体 horizon 之和的上限;
        - 高优先级 (priority 小者) 优先获得额外想象步, 直至触及 max_horizon;
        - 同级按 agent_id 字典序 (确定性)。
    * imagine(wm, scene, windows, scene_params): 按分配结果对每体调用 wm.imagine,
        汇总多步想象轨迹与"回退真实"标记; 不改 wm/predictor 权重。

设计纪律 (与全工程一致):
    * 纯前向、**确定性**、零梯度; 预算分配为确定性规则, 无可训参数。
    * opt-in: 不构造 WMScheduler 时, 旧路径逐位一致。
    * 空 / 非法输入显式 ValueError。
    * analogy, not reproduction —— 预算分配为合成可测, 非真机总线调度。

第二引擎一律称 GPM。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch

from .multi_agent import MultiAgentScene

logger = logging.getLogger("udos.wm_scheduler")


class WMScheduler:
    """多体世界模型想象预算调度器 (确定性, 非学习)。"""

    def __init__(self, total_imagination_budget: int, base_horizon: int = 1,
                 max_horizon: int = 8) -> None:
        if not isinstance(total_imagination_budget, int) or total_imagination_budget < 1:
            raise ValueError("total_imagination_budget 需为 >=1 的整数")
        if base_horizon < 1:
            raise ValueError("base_horizon 需 >=1")
        if max_horizon < base_horizon:
            raise ValueError("max_horizon 需 >= base_horizon")
        self.total_budget = int(total_imagination_budget)
        self.base_horizon = int(base_horizon)
        self.max_horizon = int(max_horizon)
        self.last_allocation: Dict[str, int] = {}

    def allocate(self, scene: MultiAgentScene) -> Dict[str, int]:
        """按优先级贪婪分配每体想象步数 horizon (>=1)。

        每个智能体先给 base_horizon; 剩余预算按 (priority, agent_id) 升序逐体 +1,
        直到预算用尽或全部达到 max_horizon。
        """
        scene.require_agents()
        ids = scene.ids
        n = len(ids)
        if self.total_budget < n * self.base_horizon:
            raise ValueError(
                f"预算 {self.total_budget} 不足以给 {n} 体各 {self.base_horizon} 步 "
                f"(需 >= {n * self.base_horizon})")
        alloc = {i: self.base_horizon for i in ids}
        remaining = self.total_budget - n * self.base_horizon
        # 优先级升序 (小者优先); 同级 id 字典序。
        # 集中式贪婪: 先把最高优先者加到 max_horizon, 再给次高, 直到预算用尽。
        order = sorted(ids, key=lambda i: (scene.get(i).priority, i))
        for i in order:
            if remaining <= 0:
                break
            can = self.max_horizon - alloc[i]
            give = min(can, remaining)
            alloc[i] += give
            remaining -= give
        self.last_allocation = alloc
        return alloc

    def imagine(self, wm, scene: MultiAgentScene,
                windows: Dict[str, torch.Tensor],
                scene_params: Optional[Dict[str, torch.Tensor]] = None
                ) -> Dict[str, Any]:
        """按 allocate() 结果对每体调用 wm.imagine。

        windows: {agent_id: window [W,6] or [1,W,6]}。
        返回每体想象轨迹 [H,6]、horizon、是否回退真实 (horizon==1)、总消耗预算。
        """
        alloc = self.last_allocation or self.allocate(scene)
        results: Dict[str, Any] = {}
        used = 0
        fell_back: List[str] = []
        imagined: List[str] = []
        for i in scene.ids:
            if i not in windows:
                raise ValueError(f"缺少 agent[{i}] 的 window")
            h = int(alloc[i])
            sp = None if scene_params is None else scene_params.get(i)
            traj = wm.imagine(windows[i], h, scene_params=sp)  # [1,H,6]
            traj = traj.reshape(h, -1)
            used += h
            if h <= 1:
                fell_back.append(i)
            else:
                imagined.append(i)
            results[i] = {
                "horizon": h,
                "fell_back_real": h <= 1,
                "trajectory": [round(float(x), 6) for x in traj.flatten().tolist()],
                "traj_shape": list(traj.shape),
            }
        return {
            "allocation": alloc,
            "agents": results,
            "used_budget": used,
            "budget_cap": self.total_budget,
            "fell_back_real": fell_back,
            "imagined": imagined,
            "deterministic": True,
            "learned": False,
        }
