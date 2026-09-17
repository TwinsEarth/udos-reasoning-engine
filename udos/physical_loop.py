"""
Physical Loop 统一闭环编排器 (v2.8.0)
========================================
analogy, not reproduction —— 受 PhysBrain 1.5 "Physical Loop" 统一闭环架构启发的
轻量化类比实现, 非复现 (UDOS 为 CPU-only ~52k 参数合成动力学小模型, 不涉及真机/VLM)。

在已有 predictor / reasoning / policy / online / adaptive 模块之上, 做一层**显式五步
编排**(不重复造轮子, 每步委托现有模块):

    observe        -> 接收历史窗口 + 场景参数, 调用 predictor 特征提取
    understand     -> 结构化理解向量 (目标状态 / 风险标记)
    predict_action -> 调用 policy.MPCActionSelector 选最优动作
    future_state   -> 对选中动作做 predictor.rollout 多步预测 + 不确定性区间
    feedback       -> 用偏差 + online.StreamingDriftDetector 决定是否需要修正

设计纪律 (与全工程一致):
    * **纯前向、确定性、不修改 predictor 权重** (只读外挂; 调用前 eval())。
    * **每步可插拔**: 构造时可传入 observe_fn/understand_fn/... 覆盖内置默认;
      不挂载任何自定义 hook 时, 编排器产出的下一步预测与直接调用
      ``predictor.predict_next`` **逐位一致** (loop_state 仅记录元数据, 无副作用)。
    * loop_state 记录每步 name / 输入形状 / 输出形状 / 耗时 / 关键标量; 动作历史与
      correction 信号在多次 run 间累积, 纯元数据。
    * 空输入守卫: 空 window / 非法 horizon 显式 ValueError, 不静默 inf 传播。

第二引擎一律称 GPM。
"""



from __future__ import annotations

import logging

logger = logging.getLogger("udos.physical_loop")


import time
from typing import Any, Callable, Dict, List, Optional

import torch

from .dynamics import RAW_DIM
from .policy import MPCActionSelector
from .online import OnlineAdapter

# 五步固定顺序 (显式编排契约, 测试与文档均依赖此顺序)
LOOP_STEPS = ("observe", "understand", "predict_action", "future_state", "feedback")


def _as_window(window: torch.Tensor) -> torch.Tensor:
    w = torch.as_tensor(window, dtype=torch.float32)
    if w.dim() == 2:
        w = w.unsqueeze(0)
    if w.dim() != 3:
        raise ValueError("window 需为 [W,RAW] 或 [B,W,RAW]")
    return w


def _as_sp(scene_params: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if scene_params is None:
        return None
    s = torch.as_tensor(scene_params, dtype=torch.float32)
    if s.dim() == 1:
        s = s.unsqueeze(0)
    return s


def _shape(x: Any) -> Any:
    """把张量/嵌套结构转成 JSON 友好的形状描述 (递归, 不拷贝数据)。"""
    if isinstance(x, torch.Tensor):
        return list(x.shape)
    if isinstance(x, dict):
        return {k: _shape(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_shape(v) for v in x]
    return None


class PhysicalLoopRunner:
    """observe -> understand -> predict_action -> future_state -> feedback 五步编排。

    Parameters
    ----------
    predictor:
        训练好的 PhysicsPredictor (只读外挂, 不修改其权重)。
    horizon:
        future_state / policy 默认 rollout 步数 (>=1)。
    observe_fn / understand_fn / predict_action_fn / future_state_fn / feedback_fn:
        可选, 覆盖对应步骤的内置默认实现。签名见各内置 ``_step_*`` 方法文档。
    """

    def __init__(self, predictor, horizon: int = 4,
                 observe_fn: Optional[Callable[..., Any]] = None,
                 understand_fn: Optional[Callable[..., Any]] = None,
                 predict_action_fn: Optional[Callable[..., Any]] = None,
                 future_state_fn: Optional[Callable[..., Any]] = None,
                 feedback_fn: Optional[Callable[..., Any]] = None,
                 use_tokenized: bool = False,
                 tokenizer: Optional[Any] = None,
                 token_predictor: Optional[Any] = None,
                 action_examples: Optional[list] = None,
                 use_memory: bool = False,
                 memory: Optional[Any] = None,
                 icl_examples: Optional[list] = None,
                 use_icm: bool = False,
                 icm_memory: Optional[Any] = None,
                 icm_k: int = 3) -> None:
        if horizon < 1:
            raise ValueError("horizon 必须 >= 1")
        self.predictor = predictor
        self.horizon = int(horizon)
        # 可插拔 hook (None => 用内置默认 _step_*)
        self.observe_fn = observe_fn
        self.understand_fn = understand_fn
        self.predict_action_fn = predict_action_fn
        self.future_state_fn = future_state_fn
        self.feedback_fn = feedback_fn
        # v3.1.1: opt-in tokenized predict_action 路径 (默认关 => 与 MPC 逐位一致)
        self.use_tokenized = bool(use_tokenized)
        self.tokenizer = tokenizer
        self.token_predictor = token_predictor
        self.action_examples = action_examples
        # v3.2.1: opt-in TemporalMemory / ICL 上下文注入 (默认关 => 与旧路径逐位一致)
        self.use_memory = bool(use_memory)
        self.memory = memory            # TemporalMemory 实例 (use_memory=True 时用)
        self.icl_examples = icl_examples   # ICL few-shot 示例窗口列表
        self._ecw = None               # 懒构造 ExtendedContextWindow
        self._icl = None
        # v3.4.1: opt-in ICM 上下文记忆 (默认关 => 与旧路径逐位一致)
        self.use_icm = bool(use_icm)
        self.icm_memory = icm_memory    # DemonstrationMemory (非空才走 ICM)
        self.icm_k = int(icm_k)
        self._icm_agg = None
        # v3.6.1: opt-in PWM 潜在世界模型作 future_state 想象模块 (默认关 => 逐位走 rollout)
        self.use_world_model: bool = False
        self.world_model = None         # LatentWorldModel (use_world_model=True 时用)
        # v2.8.0.dev2: 懒构造的 MPC 动作选择器 (predict_action 步骤使用, 只读外挂)
        self.mpc = None
        # v2.8.0.dev4: feedback 阶段的在线适配器 / 修正门控
        self.online: Optional[OnlineAdapter] = None
        self.dev_threshold: float = 1e9     # 偏差触发阈 (默认极大 => 偏差单独不触发)
        self.enable_recalibration: bool = False   # opt-in: 触发时跑 PAVA 再校准
        self.recalibration_data = None               # 再校准用的小校准集 (ParametricDataset)
        # 跨 run 累积的纯元数据
        self.loop_state: Dict[str, Any] = {
            "steps": [],            # 本次 run 每步记录
            "action_history": [],   # predict_action 历史 (跨 run 累积)
            "correction": None,     # feedback 回写的修正信号
            "run_count": 0,
        }

    # ------------------------------------------------------------------ #
    # 每步的可插拔分发: hook 优先, 否则内置默认
    # ------------------------------------------------------------------ #
    def _dispatch(self, name: str, fn: Optional[Callable[..., Any]],
                  ctx: Dict[str, Any]) -> Any:
        if fn is not None:
            return fn(ctx)
        return getattr(self, f"_step_{name}")(ctx)

    # ------------------------------------------------------------------ #
    # v3.2.1 opt-in: 集成预测 (ICL 上下文注入 / TemporalMemory 前插)
    # ------------------------------------------------------------------ #
    def _integrated_predict(self, window: torch.Tensor,
                            sp: Optional[torch.Tensor]) -> torch.Tensor:
        """统一下一步预测入口。默认 (无 opt-in) 逐位委托 predictor.predict_next;
        挂载 ICL 示例或 memory 时走对应外挂路径 (B=1)。"""
        if not self.use_memory and not self.icl_examples \
                and not self.use_icm:
            return self.predictor.predict_next(window, scene_params=sp)
        if window.size(0) != 1:
            raise ValueError("memory/ICL/ICM 集成路径当前仅支持 batch=1")
        # v3.4.1 ICM 检索聚合优先 (零梯度残差空间聚合, 优于 naive ICL)
        if self.use_icm and self.icm_memory is not None \
                and self.icm_memory.size > 0:
            from .icm import ICMAggregator
            if self._icm_agg is None:
                self._icm_agg = ICMAggregator(self.predictor)
            out = self._icm_agg.predict(window[0], memory=self.icm_memory,
                                         k=self.icm_k, scene_params=sp)
            return out.unsqueeze(0)
        # ICL 上下文注入优先
        if self.icl_examples:
            from .incontext import InContextLearner
            if self._icl is None:
                self._icl = InContextLearner(self.predictor)
            out = self._icl.predict(window[0], examples=self.icl_examples,
                                    scene_params=sp)
            return out.unsqueeze(0)
        # TemporalMemory 前插长上下文
        from .extended_context import ExtendedContextWindow
        if self._ecw is None:
            self._ecw = ExtendedContextWindow(self.predictor, max_len=24)
        ext = self.memory.build_extended_window(window[0], k=4).unsqueeze(0)
        return self._ecw.predict_next(ext, scene_params=sp, use_pe=True)

    # ------------------------------------------------------------------ #
    # 内置默认步骤 (node 1 骨架: 最小安全实现; dev1..dev4 逐步充实)
    # ------------------------------------------------------------------ #
    def _step_observe(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """observe (dev1): 接收历史 window + scene_params, 调用 predictor 特征提取。

        输出:
            features       [B, W, d_input] 观测编码器输出 (只读前向)
            scene_context  [B, scene_dim] 或 None (场景参数经 scene_encoder 编码)
            feature_shape / scene_shape
        """
        window = ctx["window"]
        sp = ctx["scene_params"]
        with torch.no_grad():
            features = self.predictor.obs_encoder(window)
            scene_ctx = self.predictor._resolve_context(None, sp)
        return {
            "features": features,
            "scene_context": scene_ctx,
            "feature_shape": list(features.shape),
            "scene_shape": (list(scene_ctx.shape)
                            if scene_ctx is not None else None),
        }

    def _step_understand(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """understand (dev1): 结构化理解向量 = 目标状态 + 风险标记。

        目标状态 = predictor.predict_next 的下一步预测 (任务目标代理)。
        风险标记 = 若挂载 OOD 检测器则给出 ood_flag/o o d_score, 否则诚实置 None;
        understanding_vector = concat([B,6] 目标状态, [B,1] 风险标量, [B,1] 不确定标量)
        固定 [B, 8], 未挂载外挂时风险/不确定通道为 0 (不伪造证据)。
        """
        window = ctx["window"]
        sp = ctx["scene_params"]
        with torch.no_grad():
            target_state = self._integrated_predict(window, sp)

        # 风险标记: OOD 检测器 (若挂载)
        ood_flag, ood_score = None, 0.0
        if getattr(self.predictor, "has_ood_detector", False):
            scores = self.predictor.ood_score(window)
            thr = float(self.predictor.ood_detector.threshold_)
            ood_flag = bool((scores > thr).any())
            ood_score = float(scores.mean())

        # 不确定标量: conformal 半宽 (若挂载)
        uncertain, interval_w = False, 0.0
        q = getattr(self.predictor, "residual_quantiles", None)
        if q is not None and len(q) > 0:
            allw = torch.cat([t.reshape(-1).float() for t in q], dim=0)
            interval_w = float(allw.mean())
            uncertain = interval_w > 0.0   # 有半宽即存在不确定区间

        risk_flags = {
            "ood_flag": ood_flag,
            "ood_score": round(ood_score, 6),
            "uncertain": bool(uncertain),
            "interval_width": round(interval_w, 6),
        }
        # 固定维理解向量 [B, 8] = [目标状态 6 | 风险 1 | 不确定 1]
        B = target_state.size(0)
        risk_chan = target_state.new_full((B, 1), ood_score)
        unc_chan = target_state.new_full((B, 1), interval_w)
        understanding_vector = torch.cat(
            [target_state, risk_chan, unc_chan], dim=-1)
        return {
            "target_state": target_state,
            "risk_flags": risk_flags,
            "understanding_vector": understanding_vector,
            "understanding_shape": list(understanding_vector.shape),
        }

    def _step_predict_action(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """predict_action (dev2): 调用 policy.MPCActionSelector 选最优动作。

        候选动作可由调用方经 ``run(..., candidate_actions=[...])`` 提供; 未提供时
        默认生成单个无操作动作 ``{}`` (诚实不伪造奖励, 仅走风险/安全项)。
        输出 best_action / best_score / best_index / ranked_actions / no_valid_action
        / best_risk; best_action 追加进 loop_state["action_history"] (跨 run 累积)。
        与直接调用 MPCActionSelector.select 逐位一致 (同一 selector)。
        """
        window = ctx["window"]
        sp = ctx["scene_params"]
        # ---- v3.1.1 opt-in: tokenized predict_action 路径 (默认关) ---- #
        if self.use_tokenized:
            if self.tokenizer is None or self.token_predictor is None:
                raise ValueError("use_tokenized=True 需提供 tokenizer 与 token_predictor")
            hist = self.loop_state.setdefault("token_history", [])
            # 种子: 用窗口末行 (6D 状态, 与 action_dim 同口径) 编码为起始 token
            if not hist:
                seed_vec = window[0, -1, :].reshape(1, -1)
                with torch.no_grad():
                    seed_tok = int(self.tokenizer.encode(seed_vec)[0])
                hist.append(seed_tok)
            nxt = int(self.token_predictor.predict_next(
                torch.tensor(hist, dtype=torch.long)))
            hist.append(nxt)
            with torch.no_grad():
                action_vec = self.tokenizer.decode(torch.tensor([nxt]))[0]
            best = {"state_perturbation": action_vec.tolist()}
            self.loop_state["action_history"].append(
                {"tokenized": True, "token": nxt})
            return {
                "best_action": best, "best_score": None, "best_index": nxt,
                "ranked_actions": [{"risk": 0.0}], "no_valid_action": False,
                "best_risk": 0.0, "n_candidates": 1, "tokenized": True,
            }
        # ---- 默认 MPC 路径 (与历史逐位一致) ---- #
        if self.mpc is None:
            self.mpc = MPCActionSelector(self.predictor, horizon=self.horizon)
        candidates = ctx.get("candidate_actions")
        if candidates is None:
            candidates = [{}]     # 未提供 => 默认单个无操作 (诚实占位)
        # 显式空列表 [] 保持空 => select 返回 no_valid_action=True
        result = self.mpc.select(window, scene_params=sp,
                                  candidate_actions=candidates)
        # 动作历史累积 (纯元数据)
        if result["best_action"] is not None:
            self.loop_state["action_history"].append(
                {"best_index": result["best_index"],
                 "best_score": result["best_score"],
                 "no_valid_action": result["no_valid_action"]})
        return {
            "best_action": result["best_action"],
            "best_score": result["best_score"],
            "best_index": result["best_index"],
            "ranked_actions": result["ranked_actions"],
            "no_valid_action": result["no_valid_action"],
            "best_risk": (result["ranked_actions"][0]["risk"]
                          if result["ranked_actions"] else None),
            "n_candidates": len(candidates),
        }

    @staticmethod
    def _apply_action(window: torch.Tensor, sp: Optional[torch.Tensor],
                      action: Optional[Dict[str, Any]]):
        """把选中动作的 state_perturbation / scene_param 覆写到窗口 (只读副本)。"""
        w = window.clone()
        if action and action.get("state_perturbation") is not None:
            pert = torch.as_tensor(action["state_perturbation"],
                                   dtype=torch.float32).reshape(-1)
            w[0, -1, :] = w[0, -1, :] + pert
        eff_sp = sp
        if action and action.get("scene_param") is not None:
            eff_sp = torch.as_tensor(action["scene_param"],
                                     dtype=torch.float32).reshape(1, -1)
        return w, eff_sp

    def _step_future_state(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """future_state (dev3): 对选中动作做 predictor.rollout 多步预测。

        输出:
            trajectory     [B, H, 6] 未来 H 步状态轨迹
            lower/upper    [B, H, 6] 不确定性区间 (挂载 conformal 半宽时; 否则 None)
            deviation      [B, 6] 首步轨迹 vs understand.target_state 的偏差信号
                           (无动作扰动时二者均为 predict_next => 偏差逐位为 0)
        """
        window = ctx["window"]
        sp = ctx["scene_params"]
        pa = ctx.get("predict_action") or {}
        best_action = pa.get("best_action")
        eff_window, eff_sp = self._apply_action(window, sp, best_action)

        with torch.no_grad():
            # v3.6.1: opt-in 用潜在世界模型想象 future_state (默认关 => 走 rollout 逐位一致)
            if self.use_world_model and self.world_model is not None:
                trajectory = self.world_model.imagine(
                    eff_window, self.horizon, scene_params=eff_sp)
            else:
                trajectory = self.predictor.rollout(
                    eff_window, self.horizon, scene_params=eff_sp)   # [B,H,6]

        # 不确定性区间 (挂载 conformal 半宽时给出, 否则诚实置 None)
        lower = upper = None
        q = getattr(self.predictor, "residual_quantiles", None)
        if q is not None and len(q) > 0:
            iv = self.predictor.predict_interval(
                eff_window, self.horizon, scene_params=eff_sp)
            lower, upper = iv["lower"], iv["upper"]

        # 偏差信号: 首步轨迹 vs understand 目标状态
        und = ctx.get("understand") or {}
        target = und.get("target_state")
        deviation = None
        if target is not None:
            deviation = trajectory[:, 0, :] - target

        return {
            "trajectory": trajectory,
            "lower": lower, "upper": upper,
            "deviation": deviation,
            "trajectory_shape": list(trajectory.shape),
            "has_interval": lower is not None,
        }

    def _step_feedback(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """feedback (dev4): 用 future_state 偏差 + StreamingDriftDetector 决定修正。

        * 偏差信号: future_state.deviation 的 L2 范数; 超过 self.dev_threshold => 触发;
        * 漂移: 若挂载了 self.online, 推入当前窗口并用其 is_drifted() 判定;
        * 默认 (enable_recalibration=False): **仅记录修正信号, 不触发任何权重/校准变更**;
        * opt-in (enable_recalibration=True 且挂载 online 与 recalibration_data):
          触发时调用 online.check_and_adapt 重跑 PAVA 校准 (不改主权重)。
        correction 信号回写 loop_state["correction"] 供下一步使用。
        """
        window = ctx["window"]
        fs = ctx.get("future_state") or {}

        # 偏差范数
        deviation = fs.get("deviation")
        deviation_norm = 0.0
        if deviation is not None:
            deviation_norm = float(deviation.norm(dim=-1).mean())
        dev_triggered = deviation_norm > self.dev_threshold

        # 漂移检测 (挂载 online 时)
        drifted = False
        drift_score = None
        if self.online is not None:
            self.online.observe(window)
            drifted = self.online.is_drifted()
            s = self.online.drift_score()
            drift_score = None if s != s else round(float(s), 6)

        triggered = bool(dev_triggered or drifted)
        reason = ("deviation" if dev_triggered else "") + \
                 ("+drift" if (dev_triggered and drifted) else
                  ("drift" if (drifted and not dev_triggered) else ""))
        reason = reason or ("none" if not triggered else "unknown")

        correction = {
            "triggered": triggered,
            "reason": reason,
            "deviation_norm": round(deviation_norm, 6),
            "dev_threshold": self.dev_threshold,
            "drifted": bool(drifted),
            "drift_score": drift_score,
            "weights_modified": False,
            "recalibrated": False,
        }

        # opt-in 再校准 (默认关 => 即使触发也不改任何状态)
        if triggered and self.enable_recalibration and self.online is not None \
                and self.recalibration_data is not None:
            res = self.online.check_and_adapt(self.predictor,
                                              self.recalibration_data)
            correction["recalibrated"] = bool(res.get("adapted", False))
            correction["weights_modified"] = bool(res.get("weights_modified",
                                                         False))

        self.loop_state["correction"] = correction
        return correction

    # ------------------------------------------------------------------ #
    # 空循环守卫
    # ------------------------------------------------------------------ #
    @staticmethod
    def _validate_window(window: torch.Tensor) -> torch.Tensor:
        w = _as_window(window)
        if w.size(1) == 0 or w.size(0) == 0:
            raise ValueError("window 不能为空 (需要 [B>=1, W>=1, RAW])")
        if not bool(torch.isfinite(w).all()):
            raise ValueError("window 含 NaN/inf 非有限值")
        return w

    # ------------------------------------------------------------------ #
    # 主入口: 跑一遍五步闭环
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def run(self, window: torch.Tensor,
            scene_params: Optional[torch.Tensor] = None,
            candidate_actions: Optional[List[Dict[str, Any]]] = None
            ) -> Dict[str, Any]:
        """
        对一条 (或一批) 观测窗口跑完整五步闭环。

        candidate_actions: 可选, predict_action 步骤的候选动作 dict 列表; 未提供时
            默认单个无操作动作。
        返回 {prediction, loop_state}; prediction 与直接调用
        ``predictor.predict_next(window, scene_params=...)`` 逐位一致
        (默认未挂载任何自定义 hook 时)。
        """
        w = self._validate_window(window)
        sp = _as_sp(scene_params)
        if sp is not None and not bool(torch.isfinite(sp).all()):
            raise ValueError("scene_params 含 NaN/inf 非有限值")

        ctx: Dict[str, Any] = {"window": w, "scene_params": sp,
                               "candidate_actions": candidate_actions}
        step_records: List[Dict[str, Any]] = []
        hooks = {
            "observe": self.observe_fn,
            "understand": self.understand_fn,
            "predict_action": self.predict_action_fn,
            "future_state": self.future_state_fn,
            "feedback": self.feedback_fn,
        }
        outputs: Dict[str, Any] = {}
        for name in LOOP_STEPS:
            t0 = time.perf_counter()
            out = self._dispatch(name, hooks[name], ctx)
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 4)
            outputs[name] = out
            ctx[name] = out           # 后续步骤可读前一步输出
            step_records.append({
                "name": name,
                "elapsed_ms": elapsed_ms,
                "output_shape": _shape(out),
            })
            logger.debug("loop step=%s elapsed_ms=%.4f shape=%s",
                         name, elapsed_ms, _shape(out))

        # 闭环最终下一步预测: 默认 understand 路径下, understand.target_state 已是
        # _integrated_predict(w, sp) 的结果 (同输入 w/sp、同 no_grad、确定性前向,
        # 探针验证逐位相等 max_abs_diff=0), 直接复用以消除一次冗余前向 (省 ~6%)。
        # 自定义 understand_fn 时无法保证其 target_state 语义, 回退原调用路径。
        und_out = outputs.get("understand") or {}
        if self.understand_fn is None and isinstance(und_out, dict) \
                and "target_state" in und_out:
            prediction = und_out["target_state"]
        else:
            prediction = self._integrated_predict(w, sp)

        # v3.2.1: 把当前窗口末帧推入记忆 (供下一次 run 累积长时域上下文)
        if self.use_memory and self.memory is not None:
            self.memory.update(w[0, -1, :])

        self.loop_state["steps"] = step_records
        self.loop_state["outputs"] = outputs
        self.loop_state["prediction_shape"] = list(prediction.shape)
        self.loop_state["run_count"] += 1
        return {"prediction": prediction, "loop_state": self.loop_state}

    # ------------------------------------------------------------------ #
    # 序列化兼容 (loop_state 纯元数据; 权重仍走 save_predictor)
    # ------------------------------------------------------------------ #
    def state_summary(self) -> Dict[str, Any]:
        """返回 JSON 友好的 loop_state 摘要 (不含原始张量)。"""
        return {
            "run_count": self.loop_state["run_count"],
            "action_history_len": len(self.loop_state["action_history"]),
            "correction": self.loop_state["correction"],
            "last_steps": [
                {k: v for k, v in s.items() if k != "output_shape"}
                for s in self.loop_state.get("steps", [])],
        }
