"""MotionPrior（动作先验 TopK 候选）+ SemanticCritic（语义挑选/接管）混合控制。

三模式对应公开报告的对照实验：
- direct  语义层直接出末端修正（无动作先验候选）：语义强、接触弱、Token 消耗高；
- motion  仅动作先验（固定 PD + 避障候选，按最近目标贪心）：接触稳、任务顺序弱；
- hybrid  动作先验每段提 K 个候选，语义层二选一（采纳先验 / 出短修正接管）。

所有“Token 消耗”均为**显式声明的 CPU 代理量**（token proxy），不是真实 LLM
计费；接 LLM 裁判需 API key（资源闸门），届时以供应商返回 usage 替换。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch

from .env import EnvTask, PointMassEnv

# ---- Token 代理计费（每次决策；cpu-proxy，非真实计费）----
TOKEN_PROXY = {
    "direct": {"in": 2200, "out": 650},   # 直读多路视觉+历史+长推理
    # 候选轨迹是紧凑结构化量（per_cand 小）；hybrid 每 6 步决策一次、direct 每 3 步，
    # 故总量上 hybrid 显著低于 direct（与外部报告 direct 更费 Token 的方向一致）。
    "hybrid": {"base_in": 900, "per_cand": 40, "out": 260},
    "motion": {"in": 0, "out": 0},        # 本地策略，无语言模型调用
}
OVERRIDE_LEN = 3   # 语义接管时出 1–5 步短修正
SEG_LEN = 6        # 采纳动作先验时执行的段长（类比 1–15 步）


@dataclass
class Candidate:
    kind: str                    # "prior" | "override"
    accs: List[torch.Tensor]
    score: float = -math.inf
    rollout: dict = field(default_factory=dict)


def _pd_acc(env: PointMassEnv, target: torch.Tensor, kp: float, kd: float,
            bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    a = kp * (target - env.pos) - kd * env.vel
    if bias is not None:
        a = a + bias
    return torch.clamp(a, -env.task.amax, env.task.amax)


class MotionPrior:
    """π0.5 类比：给当前目标提 K 条短动作段（PD 机动 + 横向避让采样）。"""

    def __init__(self, K: int = 32, seg_len: int = SEG_LEN, seed: int = 0):
        self.K = K
        self.seg_len = seg_len
        self.g = torch.Generator().manual_seed(seed)

    def _program(self, env: PointMassEnv, target: torch.Tensor,
                 kp: float, kd: float, bias_mag: float, bias_steps: int
                 ) -> List[torch.Tensor]:
        e = env.clone()
        approach = (target - e.pos)
        # 在 xy 平面内取与接近方向垂直的横向单位向量（避让障碍用）
        perp = torch.tensor([-approach[1].item(), approach[0].item(), 0.0])
        if perp.norm() > 1e-6:
            perp = perp / perp.norm()
        side = 1.0 if torch.rand(1, generator=self.g).item() < 0.5 else -1.0
        bias = side * bias_mag * perp
        accs: List[torch.Tensor] = []
        for i in range(self.seg_len):
            b = bias if i < bias_steps else None
            a = _pd_acc(e, target, kp, kd, b)
            accs.append(a)
            e.pos, e.vel, _ = e._integrate(e.pos, e.vel, a)
        return accs

    def propose(self, env: PointMassEnv, target: torch.Tensor) -> List[Candidate]:
        cands: List[Candidate] = []
        for k in range(self.K):
            if k < self.K // 4:           # 1/4 直线 PD 变体
                kp = 1.6 + 1.6 * torch.rand(1, generator=self.g).item()
                kd = 0.7 + 0.7 * torch.rand(1, generator=self.g).item()
                bias_mag, bias_steps = 0.0, 0
            else:                         # 3/4 横向机动机动（含避障轨迹）
                kp = 1.8 + 1.4 * torch.rand(1, generator=self.g).item()
                kd = 0.8 + 0.6 * torch.rand(1, generator=self.g).item()
                bias_mag = 0.4 + 2.4 * torch.rand(1, generator=self.g).item()
                bias_steps = 2 + int(torch.randint(0, self.seg_len - 2, (1,),
                                                   generator=self.g).item())
            accs = self._program(env, target, kp, kd, bias_mag, bias_steps)
            cands.append(Candidate("prior", accs))
        return cands


class SemanticCritic:
    """Astra 类比：读同一画面/历史 + 候选轨迹，按任务语义打分并决定采纳或接管。

    cpu-proto：语义判断用确定性规则（顺序、进展、碰撞、努力度），不是 LLM。
    """

    def __init__(self, override_len: int = OVERRIDE_LEN):
        self.override_len = override_len

    def _active_target(self, env: PointMassEnv) -> torch.Tensor:
        wp = env.task.waypoints
        return torch.as_tensor(wp[min(env.active, len(wp) - 1)], dtype=torch.float32)

    @staticmethod
    def nearest_target(env: PointMassEnv) -> torch.Tensor:
        """motion-only 策略：贪心选最近目标（不理解顺序语义）。"""
        wps = [torch.as_tensor(w, dtype=torch.float32) for w in env.task.waypoints]
        return min(wps, key=lambda w: (w - env.pos).norm()).clone()

    def override_program(self, env: PointMassEnv,
                         target: torch.Tensor) -> List[torch.Tensor]:
        e = env.clone()
        accs = []
        for _ in range(self.override_len):
            a = _pd_acc(e, target, kp=2.4, kd=1.1)   # 直奔语义目标，无避让先验
            accs.append(a)
            e.pos, e.vel, _ = e._integrate(e.pos, e.vel, a)
        return accs

    def score(self, env: PointMassEnv, target: torch.Tensor,
              accs: Sequence[torch.Tensor]) -> dict:
        d0 = (env.pos - target).norm().item()
        r = env.rollout_program(accs)
        # 段末应对准的目标：若段内已推进，则对新 active 目标量距离
        wp = env.task.waypoints
        end_target = torch.as_tensor(
            wp[min(r["end_active"], len(wp) - 1)], dtype=torch.float32)
        d1 = (r["end_pos"] - end_target).norm().item()
        margin = r["min_margin"] if math.isfinite(r["min_margin"]) else 9.9
        score = (6.0 * (d0 - d1) + 40.0 * r["reached"] - 0.02 * r["effort"]
                 - 2.0 * max(0.0, 0.35 - margin))
        if r["collide"]:
            score -= 1e6
        r["score"] = score
        return r

    def decide(self, env: PointMassEnv, cands: List[Candidate]) -> Candidate:
        target = self._active_target(env)
        for c in cands:
            c.rollout = self.score(env, target, c.accs)
            c.score = c.rollout["score"]
        return max(cands, key=lambda c: c.score)


def _route_remaining(env: PointMassEnv) -> float:
    """从当前位置到终点、按顺序经过剩余目标的路程。"""
    wp = env.task.waypoints
    if env.active >= len(wp):
        return 0.0
    total = (env.pos - torch.as_tensor(wp[env.active], dtype=torch.float32)).norm().item()
    for i in range(env.active, len(wp) - 1):
        total += (torch.as_tensor(wp[i + 1]) - torch.as_tensor(wp[i])).norm().item()
    return total


def run_episode(task: EnvTask, mode: str, seed: int = 0, K: int = 32,
                tracer=None) -> Dict:
    """闭环：观察 → (动作先验提候选) → 语义裁决 → 执行一段 → 重新观察。"""
    assert mode in ("direct", "motion", "hybrid")
    env = PointMassEnv(task)
    prior = MotionPrior(K=K, seed=seed)
    critic = SemanticCritic()
    D0 = _route_remaining(env) or 1.0
    decisions = interventions = 0
    tok_in = tok_out = 0
    chosen_kinds: List[str] = []

    root_cm = tracer.span("embodied.episode", task=task.name, mode=mode, seed=seed) \
        if tracer else _Null()
    with root_cm as root:
        while not env.done:
            if mode == "motion":
                tgt = critic.nearest_target(env)
            else:
                tgt = critic._active_target(env)
            decisions += 1
            if mode == "direct":
                chosen = Candidate("override", critic.override_program(env, tgt))
                tok_in += TOKEN_PROXY["direct"]["in"]
                tok_out += TOKEN_PROXY["direct"]["out"]
                kind = "override"
            else:
                with _maybe_span(tracer, "motion.propose", K=K) as ps:
                    cands = prior.propose(env, tgt)
                if mode == "hybrid":
                    cands.append(Candidate("override",
                                           critic.override_program(env, tgt)))
                with _maybe_span(tracer, "semantic.critic",
                                 candidates=len(cands)) as cs:
                    chosen = critic.decide(env, cands)
                kind = chosen.kind
                if mode == "hybrid":
                    tok_in += (TOKEN_PROXY["hybrid"]["base_in"]
                               + TOKEN_PROXY["hybrid"]["per_cand"] * K)
                    tok_out += TOKEN_PROXY["hybrid"]["out"]
                    if kind == "override":
                        interventions += 1
                        if cs:
                            cs.add_event("intervention", active=env.active)
                if ps:
                    ps.set_attribute("chosen", kind)
            chosen_kinds.append(kind)
            for acc in chosen.accs:          # 逐步真实执行（扰动在此刻生效）
                _, info = env.step(acc)
                if env.done:
                    break
        if root is not None:
            root.record_tokens(tok_in, tok_out)

    n_wp = len(task.waypoints)
    reached_frac = env.active / n_wp
    progress_frac = max(0.0, min(1.0, 1.0 - _route_remaining(env) / D0))
    clean = 1.0 if env.collisions == 0 else 0.0
    score = 100.0 * (0.6 * reached_frac + 0.2 * progress_frac) + 20.0 * clean
    intervention_rate = (interventions / decisions) if mode == "hybrid" else \
        (1.0 if mode == "direct" else 0.0)
    return {
        "task": task.name, "mode": mode, "seed": seed,
        "success": bool(env.success), "score": round(score, 2),
        "steps": env.steps, "collisions": env.collisions,
        "waypoints_reached": env.active, "waypoints_total": n_wp,
        "decisions": decisions, "interventions": interventions,
        "intervention_rate": round(intervention_rate, 3),
        "tokens_in_proxy": tok_in, "tokens_out_proxy": tok_out,
        "chosen_kinds": chosen_kinds,
    }


class _Null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def _maybe_span(tracer, name, **attrs):
    return tracer.span(name, **attrs) if tracer else _Null()
