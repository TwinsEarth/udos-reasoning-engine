"""
可供性打分 (Affordance Scoring) — v2.9.0.dev4
================================================
analogy, not reproduction —— 借鉴 Affordance / 可供性打分思想的**轻量化类比**, 非复现:

UDOS 为 CPU-only ~52k 参数合成动力学小模型, 不涉及真实物体 RGBD / 点云 / 机械臂
抓取。这里用 **状态向量的子集代理"物体部位"**: 物体部位用 [pos3, vel3] 的低维
代理向量表示, 打分基于 (距离 / 相对速度 / 可达性) 合成特征, 输出每个可操作部位
的归一化得分 [B, N_parts] 与操作建议。

设计纪律 (与全工程一致):
    * **纯前向、确定性、不修改主模型权重** (推理时外挂);
    * **归一化**: 对可达部位做 softmax 风格归一, scores 对每个 batch 求和为 1
      (无可达部位时诚实返回全 0, 不伪造证据);
    * **空/非法输入守卫**: N_parts=0、形状不符、非有限值显式 ValueError;
    * 不参与正式件训练 (2.9.0 主模型 52191 参数不变)。

第二引擎一律称 GPM。
"""



from __future__ import annotations

import logging

logger = logging.getLogger("udos.affordance")


from typing import Any, Dict, Optional

import torch


class AffordanceScorer:
    """给定机器人状态 + 物体部位代理向量, 输出可操作部位打分 [B, N_parts]。

    Parameters
    ----------
    reach_radius:
        可达半径 (欧氏距离代理); 超过此半径的部位视为不可达, 得分为 0。
    state_pos_dim / state_vel_dim:
        从状态向量里截取前 pos_dim 维代理机器人位置、随后 vel_dim 维代理速度。
    """

    def __init__(self, reach_radius: float = 2.0,
                 state_pos_dim: int = 3, state_vel_dim: int = 3) -> None:
        if reach_radius <= 0:
            raise ValueError("reach_radius 必须 > 0")
        if state_pos_dim < 1 or state_vel_dim < 0:
            raise ValueError("state_pos_dim>=1, state_vel_dim>=0")
        self.reach_radius = float(reach_radius)
        self.state_pos_dim = int(state_pos_dim)
        self.state_vel_dim = int(state_vel_dim)

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def score(self, state: torch.Tensor,
              object_proxy: torch.Tensor) -> Dict[str, Any]:
        """
        state:         [B, D] 机器人状态 (前 pos_dim 维位置, 接下来 vel_dim 维速度)。
        object_proxy:  [B, N_parts, 6] 每个部位 [pos3, vel3] 代理向量。

        返回:
            scores     [B, N_parts] 归一化得分 (对可达部位求和=1; 无可达部位全 0)
            best_part  [B] 得分最高部位索引
            reachable  [B, N_parts] bool 可达性掩码
            distances  [B, N_parts] 部位距离 (代理)
            suggestion 操作建议文本 (取 best_part)
        """
        s = torch.as_tensor(state, dtype=torch.float32)
        op = torch.as_tensor(object_proxy, dtype=torch.float32)
        if s.dim() == 1:
            s = s.unsqueeze(0)
        if s.dim() != 2:
            raise ValueError("state 需为 [D] 或 [B,D]")
        B = s.size(0)
        if op.dim() != 3 or op.size(0) != B:
            raise ValueError("object_proxy 需为 [B,N_parts,6]")
        P = op.size(1)
        if P == 0:
            raise ValueError("物体部位数 N_parts 不能为 0")
        if not (torch.isfinite(s).all() and torch.isfinite(op).all()):
            raise ValueError("输入含 NaN/inf 非有限值")

        agent_pos = s[:, :self.state_pos_dim]                       # [B,3]
        obj_pos = op[..., :self.state_pos_dim]                      # [B,P,3]
        rel = obj_pos - agent_pos.unsqueeze(1)                      # [B,P,3]
        dist = rel.norm(dim=-1)                                     # [B,P]

        reachable = (dist <= self.reach_radius).float()             # [B,P]
        # 距离特征: 越近分越高
        closeness = 1.0 / (1.0 + dist)
        # 相对速度特征: 部位朝机器人移动 => 加分 (仅在有速度通道时)
        approach = torch.zeros_like(dist)
        if self.state_vel_dim > 0 and op.size(-1) >= 6:
            obj_vel = op[..., 3:6]
            agent_vel = s[:, 3:6].unsqueeze(1)                      # [B,1,3]
            rel_vel = obj_vel - agent_vel                           # [B,P,3]
            dir_ = rel / (dist.unsqueeze(-1) + 1e-6)                # 单位方向
            approach = torch.relu(-(rel_vel * dir_).sum(dim=-1))    # 靠近为正

        raw = closeness * reachable * (1.0 + 0.5 * approach)         # [B,P]
        # 归一化: 对可达部位求和=1; 无可达部位全 0
        denom = raw.sum(dim=-1, keepdim=True)
        scores = torch.where(denom > 0, raw / denom,
                             torch.zeros_like(raw))
        best = scores.argmax(dim=-1)                                # [B]
        suggestion = [
            f"approach_part_{int(best[b])}"
            if reachable[b, int(best[b])] > 0
            else "no_reachable_part"
            for b in range(B)]
        return {
            "scores": scores,
            "best_part": best,
            "reachable": reachable.bool(),
            "distances": dist,
            "suggestion": suggestion,
            "n_parts": P,
        }

    # ------------------------------------------------------------------ #
    def config_dict(self) -> Dict[str, Any]:
        return {"kind": "AffordanceScorer",
                "reach_radius": self.reach_radius,
                "state_pos_dim": self.state_pos_dim,
                "state_vel_dim": self.state_vel_dim}


# --------------------------------------------------------------------------- #
# affordance + spatial relation + retargeting 联合动作规划 (v2.9.0.dev6)
# analogy, not reproduction: 在合成代理向量上规划"朝可操作部位接近"的动作序列。
# --------------------------------------------------------------------------- #
class AffordanceActionPlanner:
    """结合 affordance 打分 + (可选) retargeting 生成面向可操作部位的动作序列。

    affordance 低 (低于阈值或不可达) 的部位**不生成动作**。
    输出动作以 state_perturbation[6] 表示, 与 policy 动作空间对齐; 可选挂
    ActionRetargeter 把动作重定向到目标形态。

    Parameters
    ----------
    scorer:
        AffordanceScorer 实例。
    retargeter:
        可选 ActionRetargeter (若提供, 最终动作重定向到目标形态)。
    horizon:
        动作序列长度。
    aff_thresh:
        归一化得分阈值; 最高得分低于此值 => no_valid_action (低 affordance 不动作)。
    action_dim:
        输出动作维度 (默认 6, 与 RAW_DIM 对齐)。
    """

    def __init__(self, scorer: AffordanceScorer,
                 retargeter: Optional[Any] = None,
                 horizon: int = 4, aff_thresh: float = 0.05,
                 action_dim: int = 6) -> None:
        if horizon < 1:
            raise ValueError("horizon 必须 >= 1")
        self.scorer = scorer
        self.retargeter = retargeter
        self.horizon = int(horizon)
        self.aff_thresh = float(aff_thresh)
        self.action_dim = int(action_dim)

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def plan(self, state: torch.Tensor,
             object_proxy: torch.Tensor) -> Dict[str, Any]:
        """对机器人状态 + 物体部位代理规划动作序列。

        返回 {scores, best_part, action_trajectory[B,H,action_dim],
              targeted, filtered_low_affordance, no_valid_action}。
        """
        s = torch.as_tensor(state, dtype=torch.float32)
        if s.dim() == 1:
            s = s.unsqueeze(0)
        res = self.scorer.score(s, object_proxy)
        scores = res["scores"]                       # [B,P]
        best = res["best_part"]                       # [B]
        B, P = scores.shape
        top_score = scores.max(dim=-1).values         # [B]
        targeted = (top_score >= self.aff_thresh) & \
            res["reachable"].gather(1, best.view(B, 1)).squeeze(1).bool()

        agent_pos = s[:, :3]                          # [B,3]
        obj = torch.as_tensor(object_proxy, dtype=torch.float32)
        best_pos = obj.gather(1, best.view(B, 1, 1).expand(B, 1, 3)
                              ).squeeze(1)            # [B,3]
        direction = best_pos - agent_pos              # [B,3]
        direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-6)

        # 接近动作: 方向 * 步长, 重复 horizon 步; pad 到 action_dim
        step = 0.1 * direction                        # [B,3]
        pad = step.new_zeros(B, self.action_dim - 3)
        one_step = torch.cat([step, pad], dim=-1)     # [B,action_dim]
        traj = one_step.unsqueeze(1).expand(B, self.horizon,
                                            self.action_dim).clone()
        # 低 affordance / 不可达 => 不动作 (全零扰动)
        traj[~targeted] = 0.0

        return {
            "scores": scores,
            "best_part": best,
            "top_score": top_score,
            "action_trajectory": traj,
            "targeted": targeted,
            "filtered_low_affordance": int((~targeted).sum().item()),
            "no_valid_action": bool((~targeted).all()),
            "suggestion": res["suggestion"],
        }

    # ------------------------------------------------------------------ #
    def plan_action_dict(self, state: torch.Tensor,
                         object_proxy: torch.Tensor) -> Dict[str, Any]:
        """产出与 PhysicalLoopRunner/policy 兼容的 best_action dict。

        返回 {best_action, best_part, no_valid_action, scores}: best_action 为
        {"state_perturbation": [action_dim]} (未命中时 None)。
        """
        res = self.plan(state, object_proxy)
        b = int(res["best_part"][0])
        if res["no_valid_action"]:
            return {"best_action": None, "best_part": b,
                    "no_valid_action": True, "scores": res["scores"][0].tolist()}
        pert = res["action_trajectory"][0, 0].tolist()
        return {"best_action": {"state_perturbation": pert},
                "best_part": b, "no_valid_action": False,
                "scores": res["scores"][0].tolist()}
