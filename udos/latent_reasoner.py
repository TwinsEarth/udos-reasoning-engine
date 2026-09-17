"""
UDOS v4.5.0 隐式思考器 (Latent Reasoner) — CPU 合成数据机制类比
=================================================================
性质: 外挂零梯度, 不修改主权重, 不入主 state_dict。

机制 (analogy, not reproduction; 不复刻 Coconut / 大模型潜空间推理):
    在 CTM 连续隐藏状态上做"内部 tick 上的分支探索": 给定同一物理场景,
    用确定性私有随机数发生器对**编码后的物理 token 序列**加 K 条受控扰动,
    每条分支独立跑一遍 CTM 前向 (K 条潜路径, latent best-of-K, 非 token 层 beam),
    按各路径终态 certainty 做潜空间聚合/选路。
    隐式阶段**不产可读中间步**; 只输出"隐式摘要"(路径数/收敛分/路径间分歧/置信)。

none 档逐位等价: effort=none 时直接委托 engine.reason() 原路径,
    不加任何扰动/额外分支, 数值与 v4.4.1 默认完全一致 (加锚点测试)。
low/high/max 与自适应路由均 opt-in, 默认关。

设计纪律: 纯前向、确定性、不修改主权重; 第二引擎统一 GPM; 日志走 logging_config stderr。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from .ctm_engine import normalized_entropy

logger = logging.getLogger("udos.latent_reasoner")


EFFORT_LEVELS = ("none", "low", "high", "max")


@dataclass(frozen=True)
class EffortSpec:
    """一档 effort 的预算配置 (确定性, 不可变)。"""
    k: int                 # 潜路径条数 K (latent best-of-K)
    sigma: float           # 相对扰动强度 (乘编码序列 RMS)
    externalize: bool      # 是否投影为可读显式链
    chain_level: str       # "none" | "partial" | "full"
    description: str = ""


# 四档 effort 预算表 (dev2 将逐档量化并落 JSON)
EFFORT_TABLE: Dict[str, EffortSpec] = {
    # none: 单路径零扰动, 直接走引擎原路径, 与 v4.4.1 默认逐位等价
    "none": EffortSpec(k=1, sigma=0.0, externalize=False, chain_level="none",
                       description="零额外思考: 单次原向前向, 等价默认"),
    # low:   少量隐式 tick 内单/少路径探索, 不显式化
    "low":  EffortSpec(k=2, sigma=0.02, externalize=False, chain_level="none",
                       description="2 条潜路径小扰动探索, 不产可读步"),
    # high:  多路径隐式探索 + 聚合, 部分显式化
    "high": EffortSpec(k=4, sigma=0.05, externalize=True, chain_level="partial",
                       description="4 条潜路径中扰动探索, 部分可读链"),
    # max:   多路径隐式探索 + 完整显式推理链, 最深最慢
    "max":  EffortSpec(k=8, sigma=0.08, externalize=True, chain_level="full",
                       description="8 条潜路径较大扰动探索, 完整可读链"),
}


def normalize_effort(effort: Any) -> str:
    """校验 effort 档位; 空/非法 -> ValueError(对应 HTTP 400)。"""
    if effort is None:
        raise ValueError("缺少字段 'effort'")
    e = str(effort).strip().lower()
    if e not in EFFORT_TABLE:
        raise ValueError(
            f"非法 effort={effort!r}, 可选 {list(EFFORT_TABLE)}")
    return e


@dataclass
class PathOutcome:
    """单条潜路径的结果。"""
    path_index: int
    convergence: float          # 该路径终态 certainty (1-归一化熵)
    ticks_used: int             # 该路径内部 tick 数 (CTM iterations)

    def to_dict(self) -> Dict[str, Any]:
        return {"path": self.path_index,
                "convergence": round(self.convergence, 6),
                "ticks_used": self.ticks_used}


@dataclass
class LatentResult:
    """隐式思考一次调用的结果。"""
    effort: str
    n_paths: int
    total_internal_ticks: int
    externalized: bool
    explicit_chain: List[Dict[str, Any]] = field(default_factory=list)
    latent_summary: Dict[str, Any] = field(default_factory=dict)
    # 与 /reason 对齐的可解释输出 (由 engine.reason 产出, 此处透传/补全)
    prediction: Optional[List[float]] = None
    predicted_state: Optional[Dict[str, Any]] = None
    future_states: Optional[List[Dict[str, Any]]] = None
    ticks_used: int = 0
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "effort": self.effort,
            "n_paths": self.n_paths,
            "internal_ticks": self.total_internal_ticks,
            "externalized": self.externalized,
            "latent_summary": self.latent_summary,
            "latency_ms": round(self.latency_ms, 4),
        }
        if self.explicit_chain:
            out["explicit_chain"] = self.explicit_chain
            out["explicit_chain_len"] = len(self.explicit_chain)
        return out


class LatentReasoner:
    """隐式思考器 (外挂, 零梯度, 不入主 state_dict)。

    持有 engine 引用, 不持有可学习参数。effort=none 时委托 engine.reason 逐位等价。
    """

    # 私有种子, 不污染全局 torch RNG
    _SEED_BASE = 20260916

    def __init__(self, engine: Any):
        self.engine = engine
        self._rng = torch.Generator().manual_seed(self._SEED_BASE)

    # ------------------------------------------------------------------ #
    # 单条 CTM 前向 (复刻 engine.reason 的 CTM 段, 加受控扰动)
    # ------------------------------------------------------------------ #
    def _ctm_forward_once(self, scene, scene_ctx, sigma: float):
        """跑一条潜路径: 编码 token -> (可选扰动) -> CTM 前向。

        返回 (preds[out,ticks], certs[2,ticks], sync_out[n_synch], ticks_used)。
        sigma=0 时与 engine.reason 的 CTM 数值逐位一致。
        """
        encoder = self.engine.ctm_encoder
        ctm = self.engine.ctm
        seq = encoder(scene.tokens).unsqueeze(0)  # [1, S, d_input]
        if sigma > 0.0:
            # 受控分支: 对编码序列加高斯扰动 (相对其 RMS), 私有种子
            rms = seq.detach().pow(2).mean().sqrt().clamp_min(1e-8)
            with torch.no_grad():
                noise = torch.randn(seq.shape, generator=self._rng,
                                    dtype=seq.dtype)
                seq = seq + sigma * rms * noise
        preds, certs, sync_out, info = ctm(
            seq, track=False, scene_context=scene_ctx)
        preds = preds[0]    # [out, T]
        certs = certs[0]    # [2, T]
        return preds, certs, sync_out[0], int(info["ticks_used"])

    # ------------------------------------------------------------------ #
    # 主入口
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def reason(self, scene, effort: str = "none", horizon: int = 1,
               query: str = "") -> Dict[str, Any]:
        """按 effort 跑隐式思考。effort=none 逐位委托 engine.reason。"""
        effort = normalize_effort(effort)
        spec = EFFORT_TABLE[effort]
        # 每次调用复位私有种子: 同一 effort 多次调用逐位一致 (不受调用历史影响)
        self._rng.manual_seed(self._SEED_BASE)
        t0 = time.perf_counter()

        if effort == "none":
            # 逐位等价: 完全走引擎原路径, 不做任何额外操作
            r = self.engine.reason(scene, query=query, horizon=horizon)
            dt = (time.perf_counter() - t0) * 1e3
            return self._none_result(r, dt)

        # low/high/max: K 条潜路径并行探索 (CPU 串行跑 K 次前向)
        return self._explore(scene, spec, horizon, t0)

    # ------------------------------------------------------------------ #
    # none 档: 委托 + 包装 (逐位)
    # ------------------------------------------------------------------ #
    def _none_result(self, r, dt: float) -> Dict[str, Any]:
        return {
            "status": "ok",
            "effort": "none",
            "n_paths": 1,
            "internal_ticks": int(r.ticks_used),
            "externalized": False,
            "latent_summary": {
                "mode": "none-bitwise-identical",
                "n_paths": 1,
                "convergence_selected": round(float(r.convergence()), 6),
            },
            "explicit_chain": [],
            "prediction_vector": [round(x, 6) for x in r.prediction.tolist()],
            "latency_ms": round(dt, 4),
        }

    # ------------------------------------------------------------------ #
    # low/high/max: K 路径探索
    # ------------------------------------------------------------------ #
    def _explore(self, scene, spec: EffortSpec, horizon: int,
                 t0: float) -> Dict[str, Any]:
        ctm = self.engine.ctm
        # 与 engine.reason 一致的场景上下文
        scene_ctx = None
        if self.engine.scene_conditioning:
            scene_ctx = self.engine.gpm.scene_embedding(scene).unsqueeze(0)

        B, S, D = 1, 1, ctm.cfg.out_dims  # 占位, 实际形状由首次前向决定
        path_preds: List[torch.Tensor] = []
        path_cert: List[float] = []
        path_ticks: List[int] = []

        for k in range(spec.k):
            preds, certs, sync_out, ticks_used = self._ctm_forward_once(
                scene, scene_ctx, spec.sigma)
            # 终态预测 [out_dims], 终态 certainty = 1 - 归一化熵
            final_pred = preds[:, -1]
            conf = float(certs[1, -1].item())
            path_preds.append(final_pred.detach())
            path_cert.append(conf)
            path_ticks.append(ticks_used)

        # 潜空间聚合: best-of-K 选最高 certainty 路径 + 加权均值
        weights = torch.softmax(
            torch.tensor(path_cert), dim=0)          # [K]
        sel_idx = int(int(torch.tensor(path_cert).argmax()))

        # 路径间分歧: 各路径终态预测的标准差 (归一化)
        stacked = torch.stack(path_preds, dim=0)      # [K, out]
        disagreement = float(stacked.std(dim=0).mean().item())

        # 聚合答案: 置信加权平均 (best-of-K 思想: 高置信路径权重更大)
        agg_pred = (stacked * weights.unsqueeze(-1)).sum(dim=0)

        # 与主预测器对齐的可解释物理输出 (用 predictor.rollout, 与隐式探索解耦)
        predicted_state = None
        future_states = None
        if self.engine.predictor is not None and len(scene.tokens) >= 1:
            raw = self.engine._tokens_raw(scene)
            H = max(1, int(horizon))
            roll = self.engine.predictor.rollout(raw, H)[0]
            predicted_state = {"position": [round(v, 6) for v in roll[0][:3].tolist()],
                               "velocity": [round(v, 6) for v in roll[0][3:6].tolist()]}
            if H > 1:
                future_states = [
                    {"position": [round(v, 6) for v in vec[:3].tolist()],
                     "velocity": [round(v, 6) for v in vec[3:6].tolist()]}
                    for vec in roll]

        # 隐式摘要
        latent_summary = {
            "mode": "latent-best-of-k",
            "n_paths": spec.k,
            "explored_paths": spec.k,           # dev5 Trace: 探索路径数
            "per_path_ticks": path_ticks,
            "convergence_per_path": [round(c, 6) for c in path_cert],
            "convergence_mean": round(float(torch.tensor(path_cert).mean()), 6),
            "convergence_selected": round(path_cert[sel_idx], 6),
            "selected_path": sel_idx,
            "selection_reason": (              # dev5 Trace: 选择理由
                f"按终态 certainty 选路径 {sel_idx} (softmax 加权聚合)"),
            "path_disagreement": round(disagreement, 6),
            "confidence": round(path_cert[sel_idx], 6),
            "switch_point": {                  # dev5 Trace: 隐式->显式切换点
                "externalized": spec.externalize,
                "chain_level": spec.chain_level,
            },
            "closer": "latent_orchestrator",   # dev5 Trace: 收口人
        }

        # 显式化投影 (high/max): 把潜空间探索"投影"为可读步骤 (机制可读,
        # 不等于忠实复现内部计算; 详见 LATENT_REASONING_RESEARCH §1.5)
        explicit_chain = self._project_explicit_chain(
            spec, sel_idx, path_cert, disagreement)

        dt = (time.perf_counter() - t0) * 1e3
        total_ticks = sum(path_ticks)

        return {
            "status": "ok",
            "effort": effort_name(spec),
            "n_paths": spec.k,
            "internal_ticks": total_ticks,
            "externalized": spec.externalize,
            "latent_summary": latent_summary,
            "explicit_chain": explicit_chain,
            "latency_ms": round(dt, 4),
        }

    # ------------------------------------------------------------------ #
    def _project_explicit_chain(self, spec: EffortSpec, selected: int,
                                path_cert: List[float],
                                disagreement: float) -> List[Dict[str, Any]]:
        """把隐式探索投影为可读显式链 (high=partial / max=full)。"""
        chain: List[Dict[str, Any]] = []
        if not spec.externalize:
            return chain
        # 头部: 说明这是机制可读投影, 非大模型意义上的忠实 CoT
        chain.append({
            "step": 0, "kind": "note",
            "text": "显式链为隐式探索的可读投影 (机制类比), 不等于模型内部计算忠实转录"})
        if spec.chain_level == "partial":
            chain.append({
                "step": 1, "kind": "summary",
                "text": (f"隐式探索: 并行 K={spec.k} 条潜路径, "
                         f"各路径终态收敛分={[round(c,3) for c in path_cert]}, "
                         f"路径间分歧={round(disagreement,4)}, "
                         f"选择路径 {selected} (最高置信)。")})
        else:  # full
            chain.append({
                "step": 1, "kind": "summary",
                "text": f"隐式探索: 并行 K={spec.k} 条潜路径 (latent best-of-K)"})
            for i, c in enumerate(path_cert):
                chain.append({
                    "step": 2 + i, "kind": "path", "path": i,
                    "text": (f"路径 {i}: 内部 tick 收敛终态, "
                             f"certainty={round(c, 4)}"
                             + ("  [被选为最高置信]" if i == selected else ""))})
            chain.append({
                "step": 2 + len(path_cert), "kind": "selection",
                "text": (f"聚合: 按路径 certainty 做软选路, 选定路径 {selected}; "
                         f"路径间分歧={round(disagreement, 4)} "
                         f"(分歧大 -> 应升级 max/显式, 分歧小 -> 可隐式直出)。")})
        return chain


def effort_name(spec: EffortSpec) -> str:
    """反向查表 effort 名 (供内部)。"""
    for name, s in EFFORT_TABLE.items():
        if s is spec:
            return name
    return "unknown"
