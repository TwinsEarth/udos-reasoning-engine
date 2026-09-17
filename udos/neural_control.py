"""
分层神经控制 —— 大脑 / 小脑 / 脊髓 (v3.7.0, 全域 WM 统一线)
=================================================================
analogy, not reproduction —— 受生物运动皮层/小脑/脊髓三层架构启发的**合成轻量化类比**,
不宣称复现真机 whole-body control (UDOS 为 CPU-only ~52k 参数合成动力学小模型)。

三层分工 (纯推理外挂, 零梯度, 确定性, opt-in):
    * 大脑 CortexPlanner   —— 慢频率规划 (复用 policy.MPCActionSelector 多步候选评估),
                              输出目标轨迹点; 结果缓存供小脑消费。
    * 小脑 CerebellumTracker —— 快频率边缘轨迹平滑/跟踪协调 (PID 代理 + 前馈 + 低通),
                              把大脑目标点转为平滑控制信号。
    * 脊髓 SpinalReflex    —— 本地反射弧 (碰撞->制动 / 越界->截断 / 超速->减速),
                              **先制动/保护再上报**, 反射延迟 <1 步, 不等大脑/小脑。

优先级 (安全状态机): 脊髓 > 小脑 > 大脑。安全违例时脊髓指令覆盖上层, 可证。

设计纪律 (与全工程一致):
    * 纯前向、**确定性**、**不修改主 predictor 52191 参数** (只读外挂; 调用前 eval())。
    * PID / 反射弧为**确定性算法**, 无可训参数 (零梯度外挂)。
    * opt-in: 不构造 HierarchicalController 时, 旧推理路径逐位一致; 本模块不挂任何默认钩子。
    * 空 / 非法输入显式 ValueError, 不静默 inf 传播。
    * 日志沿用 logging_config (默认 WARNING, 仅 stderr)。
    * 延迟预算为**合成可测** (perf_counter 实测), 不伪造真机总线延迟。

第二引擎一律称 GPM。
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Sequence

import torch

from .decision import safety_boundary
from .dynamics import RAW_DIM
from .policy import MPCActionSelector

logger = logging.getLogger("udos.neural_control")

_POS_DIM = 3   # [pos(3), vel(3)] 布局


class ControlLayer:
    """控制层基类: 频率 / 延迟预算 / 优先级 (确定性, 无可训参数)。

    Parameters
    ----------
    name:          层名 (cortex / cerebellum / spinal)。
    frequency_hz:  该层标称频率 (合成参考; 调度由 HierarchicalController 按步分配)。
    latency_budget_ms: 该层单次调用的合成延迟预算 (ms, 仅作合同上限, 不强制阻塞)。
    priority_rank: 优先级数字越小越优先 (脊髓=0 > 小脑=1 > 大脑=2)。
    """

    #: 优先级常量: 数值越小越优先 (安全反射最高)
    PRIORITY = {"spinal": 0, "cerebellum": 1, "cortex": 2}

    def __init__(self, name: str, frequency_hz: float,
                 latency_budget_ms: float, priority_rank: int) -> None:
        if frequency_hz <= 0:
            raise ValueError("frequency_hz 必须 > 0")
        if latency_budget_ms <= 0:
            raise ValueError("latency_budget_ms 必须 > 0")
        self.name = str(name)
        self.frequency_hz = float(frequency_hz)
        self.latency_budget_ms = float(latency_budget_ms)
        self.priority_rank = int(priority_rank)
        self.last_elapsed_ms: float = 0.0

    # 子类实现: 返回该层当步输出 dict
    def act(self, state: torch.Tensor, ctx: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def profile(self, state: torch.Tensor, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """包裹 act() 并实测延迟 (perf_counter), 结果写入 last_elapsed_ms。"""
        t0 = time.perf_counter()
        out = self.act(state, ctx)
        self.last_elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 6)
        out["elapsed_ms"] = self.last_elapsed_ms
        out["frequency_hz"] = self.frequency_hz
        out["latency_budget_ms"] = self.latency_budget_ms
        return out


class CortexPlanner(ControlLayer):
    """大脑: 慢频率目标规划。

    v3.7.0.dev1: 封装 policy.MPCActionSelector 做多步候选 rollout 评估。
        * 慢频率 (1-5Hz): 由 HierarchicalController 按 cortex_every 调度, 非规划步
          复用上一步缓存目标 (供小脑平滑消费)。
        * 候选动作由调用方经 ctx["candidate_actions"] 提供; 对每个候选做自由 rollout
          + 风险/安全打分 (奖励 - lambda*风险 - 安全违例罚), 取最优者的 candidate_state
          (或 rollout 首步预测) 作为目标轨迹点。
        * 被否决候选: 安全违例 (safe=False) 的候选索引, 显式记录 (诚实不静默过滤)。
        * 未提供候选集时诚实退化: 目标 = predictor.predict_next (与 v3.7.0 逐位一致)。
    """

    def __init__(self, predictor, frequency_hz: float = 2.0,
                 latency_budget_ms: float = 40.0, horizon: int = 4,
                 lambda_risk: float = 1.0) -> None:
        super().__init__("cortex", frequency_hz, latency_budget_ms,
                         ControlLayer.PRIORITY["cortex"])
        self.predictor = predictor
        self.horizon = int(horizon)
        self.lambda_risk = float(lambda_risk)
        self._mpc: Optional[MPCActionSelector] = None
        self.last_rejected: List[int] = []
        self.last_n_candidates: int = 0

    def act(self, state: torch.Tensor, ctx: Dict[str, Any]) -> Dict[str, Any]:
        window = ctx["window"]
        sp = ctx.get("scene_params")
        candidates = ctx.get("candidate_actions")
        if candidates is None:
            # 未提供候选 => 诚实退化: 目标 = 单步预测 (与 v3.7.0 逐位一致)
            with torch.no_grad():
                target = self.predictor.predict_next(window, scene_params=sp)[0]
            return {
                "layer": "cortex", "target": target, "ran": True,
                "rejected_candidates": [], "n_candidates": 0,
                "best_score": None, "planned": False,
            }
        # 多步候选 rollout 评估 (MPC 只读外挂, 纯前向)
        if self._mpc is None:
            self._mpc = MPCActionSelector(self.predictor, horizon=self.horizon,
                                          lambda_risk=self.lambda_risk)
        sel = self._mpc.select(window, scene_params=sp,
                               candidate_actions=candidates)
        if sel["no_valid_action"]:
            raise ValueError("大脑慢规划: 候选集无有效动作 (全空/越界)")
        best = sel["best_action"]
        if best.get("candidate_state") is not None:
            target = torch.as_tensor(best["candidate_state"],
                                      dtype=torch.float32).reshape(-1)
        else:
            with torch.no_grad():
                target = self.predictor.predict_next(window,
                                                     scene_params=sp)[0]
        # 被否决候选 = 安全违例 (safe=False) 的候选索引
        rejected = [int(r["index"]) for r in sel["ranked_actions"]
                    if not r["safe"]]
        # v3.7.0.dev5: 复用 decision.safety_boundary 校验最优目标是否落在区间下界之上
        sb = safety_boundary(self.predictor, window, [target],
                             scene_params=sp)[0]
        target_safe = bool(sb["safe"])
        self.last_rejected = rejected
        self.last_n_candidates = len(candidates)
        return {
            "layer": "cortex", "target": target, "ran": True,
            "rejected_candidates": rejected,
            "n_candidates": len(candidates),
            "best_score": sel["best_score"],
            "best_index": sel["best_index"],
            "target_safe": target_safe,
            "boundary_reason": sb.get("reason"),
            "planned": True,
        }


class CerebellumTracker(ControlLayer):
    """小脑: 快频率轨迹跟踪/平滑 (确定性 PID 代理 + 前馈 + 低通, 无可训参数)。

    v3.7.0.dev2:
        * PID 代理:   effort = kp*e + ki*∫e + kd*Δe,  e = target - state (6D);
        * 前馈补偿:   ff = kff*(target_t - target_{t-1})  (预判目标速度);
        * 低通平滑:   smoothed_t = alpha*smoothed_{t-1} + (1-alpha)*(effort+ff);
        * 输出:       command = state + smoothed (增量跟踪, 不瞬移)。
    纯确定性算法, 内部积分/微分项在多次 act 间累积, reset() 清空。
    """

    def __init__(self, frequency_hz: float = 20.0,
                 latency_budget_ms: float = 5.0,
                 kp: float = 0.3, ki: float = 0.0, kd: float = 0.0,
                 kff: float = 0.0, alpha: float = 0.5) -> None:
        super().__init__("cerebellum", frequency_hz, latency_budget_ms,
                         ControlLayer.PRIORITY["cerebellum"])
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.kff = float(kff)
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha (低通系数) 需在 [0,1]")
        self.alpha = float(alpha)
        self.reset()

    def reset(self) -> None:
        self._integral: Optional[torch.Tensor] = None
        self._prev_error: Optional[torch.Tensor] = None
        self._prev_target: Optional[torch.Tensor] = None
        self._smoothed: Optional[torch.Tensor] = None

    def act(self, state: torch.Tensor, ctx: Dict[str, Any]) -> Dict[str, Any]:
        target = ctx.get("target")
        if target is None:
            raise ValueError("小脑需要大脑提供 target (ctx['target'])")
        s = state[0]
        target = torch.as_tensor(target, dtype=torch.float32).reshape(-1)
        err = target - s
        # 积分项
        if self._integral is None:
            self._integral = torch.zeros_like(err)
        self._integral = self._integral + err
        # 微分项
        if self._prev_error is None:
            derr = torch.zeros_like(err)
        else:
            derr = err - self._prev_error
        self._prev_error = err
        pid = self.kp * err + self.ki * self._integral + self.kd * derr
        # 前馈补偿 (目标速度预判)
        if self._prev_target is None:
            ff = torch.zeros_like(err)
        else:
            ff = self.kff * (target - self._prev_target)
        self._prev_target = target
        raw = pid + ff
        # 低通平滑
        if self._smoothed is None:
            self._smoothed = raw.clone()
        else:
            self._smoothed = self.alpha * self._smoothed + (1.0 - self.alpha) * raw
        command = s + self._smoothed
        return {
            "layer": "cerebellum",
            "command": command,
            "error": err,
            "smoothed": self._smoothed,
            "feedforward": ff,
            "tracked": True,
        }


class SpinalReflex(ControlLayer):
    """脊髓: 本地反射弧 (先制动/保护再上报, 反射延迟 <1 步, 不等大脑/小脑)。

    v3.7.0.dev3:
        * 碰撞 -> 制动: 位置距任一障碍物 < collision_radius => 速度归零 (position 保留);
        * 越界 -> 截断: 位置维越出 ±position_bounds => 截断到界内;
        * 超速 -> 减速: |速度| > speed_limit => 速度按比例缩到限内;
        * 反射**同步**在当步 act() 内完成 (无等待/无异步), 故反射延迟 <1 步;
        * "先制动再上报": 当步即返回安全指令覆盖上层, 事件仅作事后日志 (events)。
    纯确定性算法, 无可训参数。
    """

    def __init__(self, frequency_hz: float = 50.0,
                 latency_budget_ms: float = 0.5,
                 position_bounds: Optional[float] = None,
                 collision_obstacles: Optional[Sequence[Sequence[float]]] = None,
                 collision_radius: float = 0.5,
                 speed_limit: Optional[float] = None) -> None:
        super().__init__("spinal", frequency_hz, latency_budget_ms,
                         ControlLayer.PRIORITY["spinal"])
        self.position_bounds = (None if position_bounds is None
                                else float(position_bounds))
        self.collision_obstacles = (None if collision_obstacles is None
                                    else [torch.as_tensor(o, dtype=torch.float32)
                                          for o in collision_obstacles])
        self.collision_radius = float(collision_radius)
        self.speed_limit = (None if speed_limit is None
                            else float(speed_limit))
        self.events: List[Dict[str, Any]] = []

    def act(self, state: torch.Tensor, ctx: Dict[str, Any]) -> Dict[str, Any]:
        s = state[0].clone()
        triggered = False
        events: List[Dict[str, Any]] = []

        # 1) 碰撞 -> 制动 (优先级最高: 立即零速)
        if self.collision_obstacles is not None:
            pos = s[:_POS_DIM]
            for obs in self.collision_obstacles:
                dist = float((pos - obs).norm().item())
                if dist < self.collision_radius:
                    before_vel = s[_POS_DIM:].tolist()
                    s[_POS_DIM:] = 0.0
                    triggered = True
                    events.append({"kind": "collision_brake",
                                   "dist": round(dist, 6),
                                   "obstacle": [round(float(x), 6)
                                                for x in obs.tolist()],
                                   "velocity_before": before_vel})
                    break   # 一次制动即可, 不重复叠加

        # 2) 越界 -> 截断位置
        if self.position_bounds is not None:
            lo, hi = -self.position_bounds, self.position_bounds
            clamped = s[:_POS_DIM].clamp(lo, hi)
            if bool(torch.any(clamped != s[:_POS_DIM])):
                events.append({"kind": "bound_clamp",
                               "before": [round(float(x), 6)
                                          for x in s[:_POS_DIM].tolist()],
                               "after": [round(float(x), 6)
                                         for x in clamped.tolist()]})
                s[:_POS_DIM] = clamped
                triggered = True

        # 3) 超速 -> 减速 (速度向量按比例缩到限内)
        if self.speed_limit is not None:
            v = s[_POS_DIM:]
            speed = float(v.norm().item())
            if speed > self.speed_limit:
                scale = self.speed_limit / max(speed, 1e-12)
                before = v.tolist()
                s[_POS_DIM:] = v * scale
                triggered = True
                events.append({"kind": "speed_slowdown",
                               "speed_before": round(speed, 6),
                               "speed_limit": self.speed_limit,
                               "velocity_before": [round(float(x), 6)
                                                   for x in before]})

        return {
            "layer": "spinal",
            "reflex_triggered": triggered,
            "command": s,                 # 反射制动/截断后的安全指令
            "events": events,
        }


class HierarchicalController:
    """三层容器: 统一 step() 接口, 多频率分层调度, 优先级解析。

    Parameters
    ----------
    predictor:            训练好的 PhysicsPredictor (只读外挂, 不改权重)。
    cortex_hz / cerebellum_hz / spinal_hz: 三层标称频率; 脊髓最高频=每步。
    cortex_every:        大脑每 N 步规划一次 (None => 由频率比自动推算)。
    position_bounds:     脊髓越界截断阈值 (None => 不启用越界反射)。
    """

    def __init__(self, predictor, cortex_hz: float = 2.0,
                 cerebellum_hz: float = 20.0, spinal_hz: float = 50.0,
                 cortex_every: Optional[int] = None,
                 horizon: int = 4,
                 position_bounds: Optional[float] = None,
                 collision_obstacles: Optional[Sequence[Sequence[float]]] = None,
                 collision_radius: float = 0.5,
                 speed_limit: Optional[float] = None) -> None:
        if cortex_every is not None and cortex_every < 1:
            raise ValueError("cortex_every 必须 >= 1")
        self.predictor = predictor
        self.cortex = CortexPlanner(predictor, frequency_hz=cortex_hz,
                                    horizon=horizon)
        self.cerebellum = CerebellumTracker(frequency_hz=cerebellum_hz)
        self.spinal = SpinalReflex(frequency_hz=spinal_hz,
                                   position_bounds=position_bounds,
                                   collision_obstacles=collision_obstacles,
                                   collision_radius=collision_radius,
                                   speed_limit=speed_limit)
        # 频率比: 大脑每 N 步规划一次 (小脑/脊髓每步)
        if cortex_every is None:
            ratio = max(1, round(cerebellum_hz / max(cortex_hz, 1e-9)))
            self.cortex_every = int(ratio)
        else:
            self.cortex_every = int(cortex_every)
        self.step_index: int = 0
        self._cached_target: Optional[torch.Tensor] = None
        self.reflex_log: List[Dict[str, Any]] = []
        # v3.7.0.dev4: 各层实际调度次数计数 (验证频率比正确性)
        self.run_counts = {"cortex": 0, "cerebellum": 0, "spinal": 0}
        # v3.7.0.dev5: 安全状态机 (nominal/caution/reflex)
        self.safety_state: str = "nominal"

    def scheduler_summary(self) -> Dict[str, Any]:
        """返回多频率调度统计: 各层累计实际调用次数与频率比。"""
        total = max(self.step_index, 1)
        return {
            "total_steps": self.step_index,
            "cortex_every": self.cortex_every,
            "cortex_hz": self.cortex.frequency_hz,
            "cerebellum_hz": self.cerebellum.frequency_hz,
            "spinal_hz": self.spinal.frequency_hz,
            "ran_cortex": self.run_counts["cortex"],
            "ran_cerebellum": self.run_counts["cerebellum"],
            "ran_spinal": self.run_counts["spinal"],
            "cortex_ratio": round(self.run_counts["cortex"] / total, 4),
        }

    # ------------------------------------------------------------------ #
    @staticmethod
    def _validate_window(window: torch.Tensor) -> torch.Tensor:
        w = torch.as_tensor(window, dtype=torch.float32)
        if w.dim() == 2:
            w = w.unsqueeze(0)
        if w.dim() != 3:
            raise ValueError("window 需为 [W,RAW] 或 [B,W,RAW]")
        if w.size(1) == 0 or w.size(0) == 0:
            raise ValueError("window 不能为空 (需要 [B>=1, W>=1, RAW])")
        if w.size(-1) != RAW_DIM:
            raise ValueError(f"window 最后一维应为 RAW_DIM={RAW_DIM}")
        if not bool(torch.isfinite(w).all()):
            raise ValueError("window 含 NaN/inf 非有限值")
        return w

    @torch.no_grad()
    def step(self, window: torch.Tensor,
             scene_params: Optional[torch.Tensor] = None,
             candidate_actions: Optional[List[Dict[str, Any]]] = None
             ) -> Dict[str, Any]:
        """
        执行一步分层控制。返回 command + 各层诊断 + 优先级解析结果。
        纯前向、确定性、不改 predictor 权重。
        """
        w = self._validate_window(window)
        if scene_params is not None:
            sp = torch.as_tensor(scene_params, dtype=torch.float32)
            if sp.dim() == 1:
                sp = sp.unsqueeze(0)
            if not bool(torch.isfinite(sp).all()):
                raise ValueError("scene_params 含 NaN/inf 非有限值")
        else:
            sp = None

        state = w[:, -1, :]                 # [B,6] 末帧
        do_cortex = (self.step_index % self.cortex_every == 0)
        ctx: Dict[str, Any] = {"window": w, "scene_params": sp,
                               "candidate_actions": candidate_actions}

        # 1) 大脑 (慢频率; 不规划则复用上一步缓存目标)
        cortex_out: Dict[str, Any] = {"ran": False, "planned": False}
        if do_cortex:
            self.predictor.eval()
            cortex_out = self.cortex.profile(state, ctx)
            self._cached_target = cortex_out.get("target")
            self.run_counts["cortex"] += 1
        elif self._cached_target is not None:
            cortex_out["target"] = self._cached_target
            cortex_out["rejected_candidates"] = self.cortex.last_rejected
            cortex_out["n_candidates"] = self.cortex.last_n_candidates
        ctx["target"] = self._cached_target
        if self._cached_target is None:
            raise ValueError("大脑尚未给出目标 (首步应触发规划)")

        # 2) 小脑 (每步跟踪)
        cerebellum_out = self.cerebellum.profile(state, ctx)
        self.run_counts["cerebellum"] += 1

        # 3) 脊髓 (每步反射; 先制动再上报)
        spinal_out = self.spinal.profile(state, ctx)
        self.run_counts["spinal"] += 1
        if spinal_out["reflex_triggered"]:
            for ev in spinal_out["events"]:
                rec = {"step": self.step_index, **ev}
                self.reflex_log.append(rec)

        # 4) 优先级解析: 脊髓 > 小脑 > 大脑
        if spinal_out["reflex_triggered"]:
            final_command = spinal_out["command"]
            winner = "spinal"
        else:
            final_command = cerebellum_out["command"]
            winner = "cerebellum"

        # 5) 安全状态机: reflex (反射触发) > caution (大脑目标越安全边界) > nominal
        if spinal_out["reflex_triggered"]:
            self.safety_state = "reflex"
        elif cortex_out.get("target_safe") is False:
            self.safety_state = "caution"
        else:
            self.safety_state = "nominal"

        self.step_index += 1
        return {
            "command": final_command,
            "step_index": self.step_index - 1,
            "priority_winner": winner,
            "cortex": cortex_out,
            "cerebellum": {k: v for k, v in cerebellum_out.items()
                           if k != "command"},
            "spinal": {k: v for k, v in spinal_out.items()
                       if k != "command"},
            "reflex_triggered": bool(spinal_out["reflex_triggered"]),
            "reflex_events": spinal_out["events"],
            "safety_state": self.safety_state,
            "priority_matrix": dict(ControlLayer.PRIORITY),
        }

    def reset(self) -> None:
        """清空调度计数器、目标缓存与小脑/PID 累积状态 (跨 episode)。"""
        self.step_index = 0
        self._cached_target = None
        self.cerebellum.reset()
        self.spinal.events = []
        self.reflex_log = []
        self.safety_state = "nominal"
        self.run_counts = {"cortex": 0, "cerebellum": 0, "spinal": 0}


__all__ = [
    "ControlLayer",
    "CortexPlanner",
    "CerebellumTracker",
    "SpinalReflex",
    "HierarchicalController",
]
