"""
UDOS 推理 HTTP 服务 (零额外依赖, Python 标准库)
==============================================
把 CTM+GPM 双引擎封装为 REST 服务, 供部署/远程调用。

路由:
    GET  /health              健康检查 + 当前已内化场景 + 训练状态
    POST /internalize         入参 {scene: <PCE dict>, chunks?: [PCE...]}
    POST /reason              入参 {scene: <PCE dict>, query?: str}
    POST /reset               入参 {scene_id: str}
    POST /train               v2: 在合成动力学上短训物理预测器并挂载, 返回 loss 曲线
    POST /evaluate            v2.2: 在新鲜测试集评估当前预测器 (未训练 -> 409)
    POST /save                v2.2: 把当前预测器连同指标落 checkpoint (未训练 -> 409)
    POST /calibrate           v2.3: 拟合保序校准+残差分位并挂载 (未训练 -> 409)
    GET  /checkpoints         v2.3: 列出 checkpoints/ 下可用件
    POST /load                v2.3: 按白名单名加载 checkpoint 并挂载
    POST /predict             v2.4.12: 在线单步下一状态预测, 可选 guard=true 过退化守卫
    GET  /metrics            v2.5.1: Prometheus 文本格式服务指标 (计数/延迟分位/缓存/OOD)
    POST /rollback           v2.5.1: 回滚到上一已加载 checkpoint (无历史 409)
    POST /export-snapshot    v2.5.1: 导出无状态推理快照 (不含权重)
    POST /import-snapshot    v2.5.1: 从无状态快照恢复推理后处理
    POST /counterfactual     v2.6.0+dev6: 干预式反事实 rollout (baseline/cf/ATE)
    POST /identify           v2.6.0+dev6: 仅凭观测窗口反推隐藏场景物理参数
    POST /risk               v2.6.0+dev6: 区间宽+OOD+置信 -> 风险分/等级
    POST /diff-checkpoints   v2.6.0+dev6: 白名单内两 checkpoint 数值对比
    POST /policy/select      v2.7.0.dev6: MPC 候选动作 rollout 优选 + 全排序 (未训练 409)
    POST /online/adapt       v2.7.0.dev6: 推入观测, 漂移触发再校准 (未训练 409)
    POST /active/sample      v2.7.0.dev6: 样本池 top-K 不确定性采样 (未训练 409, 空池 400)
    GET  /experiments        v2.7.0.dev6: 实验注册表列表 (JSON, 不存在则空)
    GET  /demo                用内置工厂场景跑一遍 internalize->reason (便于联调)

启动:
    python -m udos.server --host 0.0.0.0 --port 8000 --preset small
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional

import torch

from . import __version__
from .logging_config import configure_logging
from .ctm_engine import CTMConfig
from .gpm_engine import GPMConfig, TinyBaseModel
from .pce_format import PCEParser
from .reasoning import UDOSReasoningEngine
from .dynamics import (build_dynamics_dataset, build_parametric_dataset,
                       naive_baseline_mse)
from .training import PhysicsPredictor, CTMTrainer, TrainConfig
from .evaluation import evaluate_predictor
from .calibration import fit_predictor_calibration
from .persistence import (load_predictor, save_predictor,
                          export_snapshot, import_snapshot, compare_checkpoints)
from .counterfactual import CounterfactualEngine
from .identification import SceneParameterIdentifier
from .decision import RiskGrader
from .policy import MPCActionSelector
from .online import OnlineAdapter
from .active_learning import UncertaintySampler
from .experiment import ExperimentRegistry
from .physical_loop import PhysicalLoopRunner
from .multitask import (MultiTaskHead, SpatialCoordHead, ActionTrajectoryHead,
                        FutureStateHead, SpatialRelationHead)
from .retargeting import MorphologyConfig, ActionRetargeter, MorphologyLibrary
from .affordance import AffordanceScorer, AffordanceActionPlanner
from .spatial import SpatialObject, SpatialScene
from .spatial_query import SpatialQueryEngine
from .collision import CollisionDetector
from .future_multimodal import FutureMultimodalHead
from .eval_suite import FiveDimensionEvaluator
from .world_model import LatentWorldModel
from .wm_conservation import ConservationChecker
from .neural_control import HierarchicalController
from .multi_agent import AgentCoordinator, MultiAgentScene
from .digital_twin import DigitalTwinScene
from .closed_loop import ClosedLoopOrchestrator
from .latent_reasoner import LatentReasoner, normalize_effort
from .reasoning_router import ReasoningRouter, DifficultySignals
# v4.5.4: 开源资源注册表 (L0-L3 四层, 惰性装配, 不破坏核心 import)
from .connectors import build_default_registry, ResourceUnavailable
from .resource_registry import registry_from_env
from .auth import AuthError
from .intelligence.api import NoData as NoIntelData

logger = logging.getLogger("udos.server")

# DIAG-009: checkpoint 目录固定为工程根下的绝对路径, 不依赖进程 cwd。
# server.py 位于 <root>/udos/server.py, 上级即工程根。
CHECKPOINTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")


class ServiceNotReady(Exception):
    """请求依赖尚未训练/挂载的预测器, 对应 HTTP 409。"""


class MetricsCollector:
    """v2.5.1 线程安全服务指标采集器 (纯标准库, 无 prometheus_client)。

    采集: 每端点请求计数、延迟样本 (用于 p50/p95/p99)、缓存命中率、OOD 触发率。
    """

    def __init__(self, max_latency_samples: int = 1024):
        self._lock = threading.Lock()
        self._counters: Dict[str, int] = {}
        self._latency: Dict[str, list] = {}
        self.max_latency_samples = max_latency_samples
        self._cache_hits = 0
        self._cache_misses = 0
        self._ood_triggers = 0
        self._ood_total = 0

    def record(self, endpoint: str, latency_s: float) -> None:
        with self._lock:
            self._counters[endpoint] = self._counters.get(endpoint, 0) + 1
            lst = self._latency.setdefault(endpoint, [])
            lst.append(latency_s)
            if len(lst) > self.max_latency_samples:
                lst.pop(0)

    def record_cache(self, hit: bool) -> None:
        with self._lock:
            if hit:
                self._cache_hits += 1
            else:
                self._cache_misses += 1

    def record_ood(self, triggered: bool) -> None:
        with self._lock:
            self._ood_total += 1
            if triggered:
                self._ood_triggers += 1

    @staticmethod
    def _percentile(sorted_vals: list, pct: float) -> float:
        if not sorted_vals:
            return 0.0
        k = (len(sorted_vals) - 1) * pct
        f = int(k)
        c = min(f + 1, len(sorted_vals) - 1)
        if f == c:
            return float(sorted_vals[f])
        return float(sorted_vals[f] + (k - f) * (sorted_vals[c] - sorted_vals[f]))

    def snapshot(self) -> Dict[str, Any]:
        """返回当前指标快照 (JSON 友好)。"""
        with self._lock:
            endpoints = {}
            for ep, count in self._counters.items():
                lat = sorted(self._latency.get(ep, []))
                endpoints[ep] = {
                    "count": count,
                    "latency_p50_s": round(self._percentile(lat, 0.50), 6),
                    "latency_p95_s": round(self._percentile(lat, 0.95), 6),
                    "latency_p99_s": round(self._percentile(lat, 0.99), 6),
                }
            cache_total = self._cache_hits + self._cache_misses
            ood_total = self._ood_total
            return {
                "endpoints": endpoints,
                "cache_hits": self._cache_hits,
                "cache_misses": self._cache_misses,
                "cache_hit_rate": round(
                    self._cache_hits / cache_total, 4) if cache_total else 0.0,
                "ood_triggers": self._ood_triggers,
                "ood_total": ood_total,
                "ood_trigger_rate": round(
                    self._ood_triggers / ood_total, 4) if ood_total else 0.0,
            }

    def prometheus_text(self) -> str:
        """输出 Prometheus 文本格式 exposition (纯标准库字符串拼接)。"""
        s = self.snapshot()
        lines = [
            "# HELP udos_requests_total Total requests by endpoint",
            "# TYPE udos_requests_total counter",
        ]
        for ep, data in s["endpoints"].items():
            safe_ep = ep.replace('"', "")
            lines.append(
                f'udos_requests_total{{endpoint="{safe_ep}"}} {data["count"]}')
        lines += [
            "# HELP udos_latency_seconds Request latency in seconds",
            "# TYPE udos_latency_seconds gauge",
        ]
        for ep, data in s["endpoints"].items():
            safe_ep = ep.replace('"', "")
            lines.append(
                f'udos_latency_seconds{{endpoint="{safe_ep}",quantile="0.5"}} {data["latency_p50_s"]}')
            lines.append(
                f'udos_latency_seconds{{endpoint="{safe_ep}",quantile="0.95"}} {data["latency_p95_s"]}')
            lines.append(
                f'udos_latency_seconds{{endpoint="{safe_ep}",quantile="0.99"}} {data["latency_p99_s"]}')
        lines += [
            "# HELP udos_cache_hits_total Inference cache hits",
            "# TYPE udos_cache_hits_total counter",
            f"udos_cache_hits_total {s['cache_hits']}",
            "# HELP udos_cache_misses_total Inference cache misses",
            "# TYPE udos_cache_misses_total counter",
            f"udos_cache_misses_total {s['cache_misses']}",
            "# HELP udos_ood_triggers_total OOD triggers",
            "# TYPE udos_ood_triggers_total counter",
            f"udos_ood_triggers_total {s['ood_triggers']}",
        ]
        return "\n".join(lines) + "\n"

_PRESETS = {
    # 名称: (hidden/feature/latent, d_model, iterations, n_layers, lora_rank)
    "small": dict(width=64, d_model=128, iterations=16, n_layers=2, rank=4),
    "medium": dict(width=128, d_model=256, iterations=32, n_layers=4, rank=8),
}


class UDOSService:
    """引擎生命周期与请求处理 (与 HTTP 层解耦, 便于单测)。"""

    def __init__(self, preset: str = "small", lora_scaling: float = 0.1,
                 device: str = "cpu", checkpoint: Optional[str] = None,
                 checkpoints_dir: Optional[str] = None):
        torch.manual_seed(0)
        p = _PRESETS[preset]
        base = TinyBaseModel(hidden=p["width"], n_layers=p["n_layers"])
        gpm = GPMConfig(
            feature_dim=p["width"], latent_size=p["width"], n_latents=16,
            lora_rank=p["rank"],
            target_modules=("down_proj", "gate_proj", "up_proj"),
            layer_indices=tuple(range(p["n_layers"])),
            init_scaler_b_zero=False)
        ctm = CTMConfig(
            iterations=p["iterations"], d_model=p["d_model"],
            d_input=p["width"], heads=4, n_synch_out=32, n_synch_action=32,
            memory_length=16, out_dims=p["width"], certainty_threshold=0.0)
        self.engine = UDOSReasoningEngine(ctm, gpm, base_model=base,
                                          lora_scaling=lora_scaling)
        self.engine.eval()
        self._lock = threading.Lock()
        self.device = device
        self.trained = False
        self.last_train: Dict[str, Any] = {}
        # DIAG-009: checkpoint 目录锚定工程根, 不随 cwd 漂移;
        # checkpoints_dir 仅供测试注入隔离目录, 生产默认工程根。
        self.checkpoints_dir = os.path.abspath(
            checkpoints_dir if checkpoints_dir else CHECKPOINTS_DIR)
        # v2.5.1: 服务指标采集 + checkpoint 加载栈 (用于回滚)
        self.metrics = MetricsCollector()
        self._load_stack: list = []   # 已加载 checkpoint 路径栈 (后进先出)
        # v2.7.0.dev6: 懒初始化的在线适配器 (首次 /online/adapt 时按参考分布构造)
        self._online: Optional[OnlineAdapter] = None
        # v2.8.2: 懒初始化 PhysicalLoopRunner 与 MultiTaskHead
        self._loop: Optional[PhysicalLoopRunner] = None
        self._mth: Optional[MultiTaskHead] = None
        # v3.0.1: 懒初始化 future_multimodal 头与五维评测器
        self._fmh: Optional[FutureMultimodalHead] = None
        self._eval: Optional[FiveDimensionEvaluator] = None
        # v3.1.2: 懒初始化 ActionPiece tokenizer (tokenize/detokenize 端点用)
        self._tokenizer = None
        # v3.4.1: ICM 上下文记忆库 (服务级单例; 演示条件化预测 opt-in)
        self._icm_memory = None
        self._icm_agg = None
        # v3.5.1: SFM 空间场景 (opt-in; 也可由请求体 objects 临时构造)
        self._spatial_scene: Optional[SpatialScene] = None
        self._collision = CollisionDetector()
        # v3.6.1: PWM 潜在世界模型 (懒初始化外挂; 首次 /wm/* 时构造+离线拟合)
        self._wm = None
        # v3.7.1: 分层神经控制器 (大脑/小脑/脊髓; 首次 /neural/* 时懒构造)
        self._neural: Optional[HierarchicalController] = None
        # v3.8.2: 数字孪生场景 + 全域闭环编排器 (懒构造; 首次 /twin/* 时创建)
        self._twin: Optional[DigitalTwinScene] = None
        self._twin_loop: Optional[ClosedLoopOrchestrator] = None
        # v4.1.2: 自规划自监督线懒单例 (课程生成器/自验证器/伪标签器/分解器)
        self._cur_gen = None
        self._cur_ver = None
        self._selfsup = None
        self._decomposer = None
        # v4.2.2: 完全自训练线懒单例 (三元组生成器/自博弈探索器)
        self._selftrain_gen = None
        self._selfplay_expl = None
        # v4.3.1: 完全自进化线懒单例 (配置自优化评估器)
        self._self_evo_evaluator = None
        # v4.4.0: 多智能体协作线懒单例 (能力注册表 + trace 链查询存储)
        self._collab_registry = None
        self._collab_traces: Dict[str, Any] = {}
        # v4.5.1: 隐式思考线懒单例 (隐式思考器 + 难度路由)
        self._latent_reasoner: Optional[LatentReasoner] = None
        self._reasoning_router = ReasoningRouter()
        # v4.5.4: 开源资源注册表懒单例 (profile 由 UDOS_RESOURCE_PROFILE 决定,
        #         默认 performance; 装配 79 条元数据, 真探测带缓存)
        self._registry = None
        self._resource_profile = registry_from_env("performance")
        # v5.0.1: 安全与受控自治懒单例 (默认关/透明)
        self._auth = None
        self._authz = None
        self._backup = None
        self._autonomy = None
        self._intel = None
        self._last_kvcache_sim = None
        from .authz import auth_enabled
        if auth_enabled():
            self._get_auth()
        # v2.1: 可选预加载训练好的预测器, 启动即带物理多步推演
        if checkpoint:
            predictor, meta = load_predictor(checkpoint)
            self.engine.attach_predictor(predictor)
            self.trained = True
            self.last_train = meta.get("metrics", {})
            self._load_stack.append(checkpoint)

    @staticmethod
    def _coerce_int(value: Any, default: int, field: str) -> int:
        """FIX-301: 安全 int 转换, None/非数值 -> ValueError(400) 而非 TypeError(500)。"""
        if value is None:
            return default
        try:
            return int(value)
        except (ValueError, TypeError) as e:
            raise ValueError(f"字段 '{field}' 无法解析为整数: {e}")

    @staticmethod
    def _coerce_float(value: Any, default: float, field: str) -> float:
        """FIX-301: 安全 float 转换, None/非数值 -> ValueError(400) 而非 TypeError(500)。"""
        if value is None:
            return default
        try:
            return float(value)
        except (ValueError, TypeError) as e:
            raise ValueError(f"字段 '{field}' 无法解析为浮点数: {e}")

    @staticmethod
    def _scene(body: Dict[str, Any]):
        if "scene" not in body:
            raise ValueError("缺少字段 'scene'")
        scene = PCEParser._from_dict(body["scene"])
        # DIAG-004: 结构残缺的 scene 解析后无有效 tokens, 后续 torch.stack([]) 会
        # 抛 RuntimeError 被吞成 500; 在入口层做输入校验 -> 400。
        # 不改动 pce_format.py 的合法 PCE 序列化路径。
        if not getattr(scene, "tokens", None):
            raise ValueError("PCE scene 结构不完整或无有效 tokens")
        return scene

    def internalize(self, body: Dict[str, Any]) -> Dict[str, Any]:
        scene = self._scene(body)
        chunks = None
        if body.get("chunks"):
            chunks = [PCEParser._from_dict(c) for c in body["chunks"]]
        with self._lock:
            msg = self.engine.internalize_scene(scene, chunks=chunks)
            lora = self.engine.scene_memory[scene.scene_id]
        logger.info("internalize scene_id=%s lora_params=%s",
                    scene.scene_id, lora.num_params())
        return {"status": "ok", "message": msg, "scene_id": scene.scene_id,
                "lora_params": lora.num_params(),
                "lora_kb_fp32": round(lora.num_bytes_fp32() / 1024, 2),
                "target_modules": lora.module_names()}

    def reason(self, body: Dict[str, Any]) -> Dict[str, Any]:
        scene = self._scene(body)
        query = str(body.get("query", ""))
        horizon = int(body.get("horizon", 1))
        if not (1 <= horizon <= 16):
            raise ValueError("horizon 需在 [1,16]")
        with self._lock:
            r = self.engine.reason(scene, query=query, horizon=horizon)
        out = {
            "status": "ok",
            "summary": r.summary(),
            "ticks_used": r.ticks_used,
            "scene_conditioned": r.scene_conditioned,
            "predictor_conditioned": r.predictor_conditioned,
            "horizon": horizon,
            "final_certainty": r.convergence(),
            "certainty_trajectory": [round(x, 6) for x in
                                     r.certainty_trajectory[1].tolist()],
            "prediction_vector": [round(x, 6) for x in r.prediction.tolist()],
            "causal_chain": r.causal_chain,
            "internalized": r.scene_id in self.engine.scene_memory,
        }
        if r.predicted_state is not None:
            out["predicted_next_state"] = r.predicted_state
        if r.future_states is not None:
            out["future_states"] = r.future_states
        return out

    def train(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """
        v2.1: 参数化场景条件 + 多步物理推演训练。隐藏物理参数经 scene_encoder
        进入 CTM, 训练后挂载; 返回单步/消融/多步rollout/物理一致性全套指标。
        """
        epochs = int(body.get("epochs", 45))
        n_per_kind = int(body.get("n_per_kind", 32))
        horizon = int(body.get("horizon", 4))
        phys_weight = float(body.get("phys_weight", 0.0))
        ss_max = float(body.get("ss_max", 0.0))       # v2.2 默认 0=纯 teacher-forcing
        ss_start = int(body.get("ss_start", 15))
        ss_warmup = int(body.get("ss_warmup", 25))
        # v2.3 多时域损失权重, 默认 front (与 2.2.1 等价; uniform/back 为 opt-in)
        step_scheme = str(body.get("step_weight_scheme", "front"))
        if not (1 <= epochs <= 300):
            raise ValueError("epochs 需在 [1,300]")
        if not (1 <= n_per_kind <= 512):
            raise ValueError("n_per_kind 需在 [1,512]")
        if not (1 <= horizon <= 8):
            raise ValueError("horizon 需在 [1,8]")
        if not (0.0 <= ss_max <= 1.0):
            raise ValueError("ss_max 需在 [0,1]")
        if step_scheme not in TrainConfig.STEP_WEIGHT_SCHEMES:
            raise ValueError(
                f"step_weight_scheme 需为 {TrainConfig.STEP_WEIGHT_SCHEMES}")
        ds = build_parametric_dataset(n_per_kind=n_per_kind, n_steps=14,
                                      window=6, horizon=horizon, dt=0.5)
        tr, te = ds.split(0.8)
        cfg = CTMConfig(iterations=8, d_model=64, d_input=32, heads=4,
                        n_synch_out=16, n_synch_action=8, memory_length=8,
                        nlm_hidden=16, out_dims=32, certainty_threshold=0.0)
        predictor = PhysicsPredictor(cfg, scene_param_dim=ds.P.size(1))
        # 训练前对照 (条件单步 MSE) 与"状态不变"朴素基线
        before = evaluate_predictor(predictor, te)
        untrained_mse = before["single_step_mse"]
        naive_mse = ((te.X[:, -1, :] - te.Y[:, 0, :]) ** 2).mean().item()
        trainer = CTMTrainer(predictor, TrainConfig(
            epochs=epochs, lr=3e-3, batch_size=64, phys_weight=phys_weight,
            ss_max=ss_max, ss_start=ss_start, ss_warmup=ss_warmup,
            step_weight_scheme=step_scheme))
        hist = trainer.train(tr, te)
        report = evaluate_predictor(predictor, te)
        trained_mse = report["single_step_mse"]
        n_params = sum(p.numel() for p in predictor.parameters())
        with self._lock:
            self.engine.attach_predictor(predictor)
            self.trained = True
            self.last_train = {
                "epochs": epochs, "params": n_params, "horizon": horizon,
                "untrained_mse": untrained_mse, "naive_mse": naive_mse,
                "trained_mse": trained_mse, "evaluation": report}
        return {
            "status": "ok", "trained": True, "epochs": epochs,
            "horizon": horizon, "samples": len(ds),
            "predictor_params": n_params,
            "untrained_mse": round(untrained_mse, 6),
            "naive_mse": round(naive_mse, 6),
            "trained_mse": round(trained_mse, 6),
            "reduction_vs_untrained_x": round(
                untrained_mse / max(trained_mse, 1e-12), 2),
            "reduction_vs_naive_x": round(
                naive_mse / max(trained_mse, 1e-12), 2),
            "evaluation": report,
            "scheduled_sampling": {"ss_max": ss_max, "ss_start": ss_start,
                                   "ss_warmup": ss_warmup,
                                   "stopped_early": hist.stopped_early},
            "step_weight_scheme": step_scheme,
            "train_loss_curve": [round(x, 6) for x in hist.train_loss],
            "eval_mse_curve": [round(x, 6) for x in hist.eval_mse],
            "certainty_curve": [round(x, 6) for x in hist.cert],
            "phys_violation_curve": [round(x, 6) for x in hist.phys_violation],
            "ss_prob_curve": [round(x, 4) for x in hist.ss_prob],
        }

    # ---------- v2.2 模型管理: 在线评估 / 落盘 ----------
    def _require_predictor(self) -> PhysicsPredictor:
        pred = getattr(self.engine, "predictor", None)
        if pred is None or not getattr(self, "trained", False):
            raise ServiceNotReady("尚未训练或挂载物理预测器, 请先 POST /train")
        return pred

    def evaluate(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """在与训练分布一致的新鲜参数化测试集上评估当前预测器。"""
        predictor = self._require_predictor()
        n_per_kind = int(body.get("n_per_kind", 32))
        horizon = int(body.get("horizon", 4))
        seed = int(body.get("seed", 2024))   # 默认用不同于训练的种子 => 新鲜样本
        include_ood = bool(body.get("include_ood", False))
        if not (1 <= n_per_kind <= 512):
            raise ValueError("n_per_kind 需在 [1,512]")
        if not (1 <= horizon <= 8):
            raise ValueError("horizon 需在 [1,8]")
        with self._lock:
            ds = build_parametric_dataset(
                n_per_kind=n_per_kind, n_steps=6 + horizon + 4,
                window=6, horizon=horizon, dt=0.5, seed=seed)
            _, te = ds.split(0.8)
            rep = evaluate_predictor(predictor, te)
            out = {"status": "evaluated", "version": __version__,
                   "test_seed": seed, "n_eval": len(te), "metrics": rep}
            # v2.4.8: 可选 OOD 段 (仅当预测器挂载了检测器)
            if include_ood and getattr(predictor, "has_ood_detector", False):
                out["ood"] = predictor.ood_detector.ks_drift(te.X)
            # v2.4.12: 可选 guard 段 (触发/回退次数)。guard=true 时在测试集上跑一遍
            # 守卫透传预测并统计 NaN/inf 回退与越界截断; 健康模型应为 0/0。
            if bool(body.get("guard", False)):
                from .guard import PredictionGuard
                predictor.attach_guard(PredictionGuard())
                scene = te.P if predictor.scene_encoder is not None \
                    else torch.zeros_like(te.P)
                predictor.predict_next(te.X, scene_params=scene, guard=True)
                out["guard"] = {"enabled": True,
                                **predictor.guard.stats()}
            # v2.5.1: /evaluate 含 service_metrics 段 (当前服务指标快照, 不覆盖旧 metrics)
            out["service_metrics"] = self.metrics.snapshot()
        return out

    # ---------- v2.4.12 在线单步预测 (可选退化守卫) ----------
    def predict(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """
        在线单步下一状态预测。入参 window: [W,RAW] 或 [N,W,RAW] (历史窗口运动学),
        可选 scene_params: [N,P], guard: bool (默认 false 透传, true 过 PredictionGuard)。
        未训练/挂载 -> 409; 非法形状 -> 400。guard=true 时返回守卫触发/回退统计。
        """
        predictor = self._require_predictor()
        seq = body.get("window")
        if seq is None:
            raise ValueError("缺少字段 'window' (形状 [W,RAW] 或 [N,W,RAW])")
        try:
            window = torch.tensor(seq, dtype=torch.float32)
        except Exception as e:
            raise ValueError(f"window 无法解析为张量: {e}")
        if window.dim() == 2:
            window = window.unsqueeze(0)
        if window.dim() != 3:
            raise ValueError("window 需为 [W,RAW] 或 [N,W,RAW]")
        if window.size(-1) != predictor.raw_dim:
            raise ValueError(
                f"window 最后一维 {window.size(-1)} 应为 {predictor.raw_dim}")
        sp = body.get("scene_params")
        if sp is not None:
            try:
                sp = torch.tensor(sp, dtype=torch.float32)
            except Exception as e:
                raise ValueError(f"scene_params 无法解析为张量: {e}")
            if sp.dim() == 1:
                sp = sp.unsqueeze(0)
        guard_on = bool(body.get("guard", False))
        with self._lock:
            if guard_on:
                from .guard import PredictionGuard
                predictor.attach_guard(PredictionGuard())   # 每请求全新计数
            pred = predictor.predict_next(window, scene_params=sp,
                                          guard=guard_on)
            guard_stats = ({"enabled": True, **predictor.guard.stats()}
                           if guard_on else {"enabled": False})
        return {"status": "ok", "version": __version__,
                "prediction": pred.tolist(),
                "shape": list(pred.shape), "guard": guard_stats}

    _NAME_SAFE = re.compile(r"[^A-Za-z0-9._-]")

    def save(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """把当前预测器连同最近一次指标落 checkpoints/<name>.pt。"""
        predictor = self._require_predictor()
        raw_name = body.get("name") or f"predictor_{int(time.time())}"
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise ValueError("name 必须是非空字符串")
        name = self._NAME_SAFE.sub("_", raw_name).strip("._")
        if not name or name.startswith("/") or ".." in name or os.sep in name:
            raise ValueError("非法 checkpoint 名称")
        if not name.endswith(".pt"):
            name += ".pt"
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        base = os.path.abspath(self.checkpoints_dir)
        path = os.path.abspath(os.path.join(base, name))
        # 根本防线: 无论名称如何清洗, 最终路径不得逃出 checkpoints 目录
        if os.path.commonpath([base, path]) != base:
            raise ValueError("非法 checkpoint 路径")
        last_train = getattr(self, "last_train", None)
        saved_metrics = dict(last_train or {})
        saved_metrics["saved_by"] = "UDOSService.save"
        with self._lock:
            save_predictor(predictor, path, metrics=saved_metrics)
        nbytes = os.path.getsize(path)
        logger.info("save checkpoint name=%s bytes=%d", name, nbytes)
        return {"status": "saved", "version": __version__, "path": path,
                "bytes": nbytes, "metrics": last_train}

    def reset(self, body: Dict[str, Any]) -> Dict[str, Any]:
        sid = body.get("scene_id")
        if not sid:
            raise ValueError("缺少字段 'scene_id'")
        with self._lock:
            msg = self.engine.reset_scene(sid)
        return {"status": "ok", "message": msg, "scene_id": sid}

    # ---------- v2.3 校准与多 checkpoint 管理 ----------
    def _safe_checkpoint_path(self, raw_name: Any, need_exist: bool = True) -> str:
        """复用 save 的白名单 + commonpath 不可逃逸校验, 返回绝对路径。"""
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise ValueError("name 必须是非空字符串")
        name = UDOSService._NAME_SAFE.sub("_", raw_name).strip("._")
        if not name or name.startswith("/") or ".." in name or os.sep in name:
            raise ValueError("非法 checkpoint 名称")
        if not name.endswith(".pt"):
            name += ".pt"
        base = os.path.abspath(self.checkpoints_dir)
        path = os.path.abspath(os.path.join(base, name))
        if os.path.commonpath([base, path]) != base:
            raise ValueError("非法 checkpoint 路径")
        if need_exist and not os.path.isfile(path):
            raise ValueError(f"checkpoint 不存在: {name}")
        return path

    def calibrate(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """在独立校准集上为当前预测器拟合保序校准 + 残差分位, 并挂载 (随 /save 持久化)。"""
        predictor = self._require_predictor()
        n_per_kind = int(body.get("n_per_kind", 32))
        horizon = int(body.get("horizon", 4))
        cal_seed = int(body.get("calibration_seed", 2031))
        test_seed = int(body.get("test_seed", 2042))
        if not (1 <= n_per_kind <= 512):
            raise ValueError("n_per_kind 需在 [1,512]")
        if not (1 <= horizon <= 8):
            raise ValueError("horizon 需在 [1,8]")
        with self._lock:
            # 独立校准/测试集整份使用 (独立种子生成, 无需再切; 保证逐维半宽样本量)
            cal_te = build_parametric_dataset(
                n_per_kind=n_per_kind, n_steps=6 + horizon + 4,
                window=6, horizon=horizon, dt=0.5, seed=cal_seed)
            calibrator, fitted_report, rq = fit_predictor_calibration(
                predictor, cal_te)
            predictor.attach_calibration(calibrator, rq)
            # 独立测试集上的泛化校准 (唯一变量: 是否做保序映射)
            ind_te = build_parametric_dataset(
                n_per_kind=n_per_kind, n_steps=6 + horizon + 4,
                window=6, horizon=horizon, dt=0.5, seed=test_seed)
            indep = evaluate_predictor(predictor, ind_te)
        return {"status": "calibrated", "version": __version__,
                "calibration_seed": cal_seed, "test_seed": test_seed,
                "fitted_on_calibration_set": fitted_report,
                "independent_test": {"calibration": indep.get("calibration"),
                                     "interval": indep.get("interval")}}

    # ---------- v2.4.8 OOD / 漂移检测服务接口 ----------
    def detect_ood(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """
        对入参场景序列 [N, W, RAW] 做 OOD/漂移检测。未挂载预测器或检测器 -> 409。
        返回每样本马氏距离、阈值、OOD 命中率与整批漂移摘要。
        """
        predictor = self._require_predictor()
        if not getattr(predictor, "has_ood_detector", False):
            raise ServiceNotReady("当前预测器未挂载 OOD 检测器")
        seq = body.get("sequence")
        if seq is None:
            raise ValueError("缺少字段 'sequence' (形状 [N, W, RAW])")
        try:
            window = torch.tensor(seq, dtype=torch.float32)
        except Exception as e:
            raise ValueError(f"sequence 无法解析为张量: {e}")
        if window.dim() == 2:
            window = window.unsqueeze(0)
        if window.dim() != 3:
            raise ValueError("sequence 需为 [W, RAW] 或 [N, W, RAW]")
        # 校验展平维与检测器一致 (window*raw)
        expected = predictor.ood_detector.dim
        if window.reshape(window.size(0), -1).size(1) != expected:
            raise ValueError(
                f"sequence 展平维 {window.reshape(window.size(0),-1).size(1)} "
                f"与检测器维 {expected} 不一致")
        with self._lock:
            scores = predictor.ood_score(window)
            threshold = float(predictor.ood_detector.threshold_)
            flags = scores > threshold
            # v2.5.1: 记录 OOD 触发率 (任一 OOD 样本即触发)
            self.metrics.record_ood(bool(flags.any()))
        return {
            "status": "ok", "version": __version__,
            "threshold": round(threshold, 6),
            "scores": [round(float(s), 6) for s in scores],
            "ood": [bool(f) for f in flags],
            "ood_rate": round(float(flags.float().mean()), 4),
            "score_mean": round(float(scores.mean()), 6),
            "score_max": round(float(scores.max()), 6),
        }

    def list_checkpoints(self) -> Dict[str, Any]:
        """列出 checkpoints/ 下可用件 (名/字节/修改时间), 不做反序列化。"""
        base = os.path.abspath(self.checkpoints_dir)
        items = []
        if os.path.isdir(base):
            for fn in sorted(os.listdir(base)):
                if not fn.endswith(".pt"):
                    continue
                p = os.path.join(base, fn)
                if not os.path.isfile(p):
                    continue
                st = os.stat(p)
                items.append({"name": fn, "bytes": st.st_size,
                              "modified": time.strftime(
                                  "%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime))})
        return {"status": "ok", "checkpoint_dir": base, "count": len(items),
                "checkpoints": items}

    def load(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """按白名单名加载 checkpoint (含其校准器) 并挂载为当前预测器。"""
        path = self._safe_checkpoint_path(body.get("name"))
        with self._lock:
            predictor, meta = load_predictor(path)
            self.engine.attach_predictor(predictor)
            self.trained = True
            self.last_train = meta.get("metrics", {})
            self._load_stack.append(path)   # v2.5.1: 记录加载栈用于回滚
        logger.info("load checkpoint name=%s calibrated=%s",
                    os.path.basename(path),
                    bool(getattr(predictor, "is_calibrated", False)))
        return {"status": "loaded", "version": __version__,
                "path": path, "source_version": meta.get("udos_version"),
                "calibrated": bool(getattr(predictor, "is_calibrated", False)),
                "saved_at": meta.get("saved_at")}

    # ---------- v2.5.1 服务指标 / 回滚 / 快照 ---------- #
    def metrics(self) -> Dict[str, Any]:
        """返回服务指标快照 (JSON)。Prometheus 文本由 HTTP 层转换。"""
        return {"status": "ok", "version": __version__,
                "metrics": self.metrics.snapshot()}

    def rollback(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """
        回滚到上一已加载 checkpoint。维护加载栈: 当前在栈顶, 回滚弹出当前并加载
        前一个。栈中仅剩一个 (启动时预加载) 或无历史时返回 409 (ServiceNotReady)。
        """
        with self._lock:
            if len(self._load_stack) < 2:
                raise ServiceNotReady(
                    "无历史 checkpoint 可回滚 (加载栈长度 < 2)")
            # 弹出当前, 加载前一个
            self._load_stack.pop()
            prev_path = self._load_stack[-1]
            predictor, meta = load_predictor(prev_path)
            self.engine.attach_predictor(predictor)
            self.trained = True
            self.last_train = meta.get("metrics", {})
        return {"status": "rolled_back", "version": __version__,
                "restored_path": prev_path,
                "source_version": meta.get("udos_version"),
                "stack_depth": len(self._load_stack)}

    def export_snapshot_endpoint(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """导出无状态推理快照 (配置+校准器+OOD 统计, 不含权重)。"""
        predictor = self._require_predictor()
        with self._lock:
            snap = export_snapshot(predictor)
        return {"status": "ok", "version": __version__,
                "snapshot": snap, "has_weights": False}

    def import_snapshot_endpoint(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """从无状态快照恢复推理后处理 (校准器+OOD), 权重必须已存在。"""
        predictor = self._require_predictor()
        snap = body.get("snapshot")
        if snap is None:
            raise ValueError("缺少字段 'snapshot'")
        with self._lock:
            import_snapshot(predictor, snap)
        return {"status": "ok", "version": __version__,
                "calibrated": bool(getattr(predictor, "is_calibrated", False)),
                "has_ood": bool(getattr(predictor, "has_ood_detector", False))}

    # ---------- v2.6.0+dev6 反事实 / 辨识 / 风险 / 快照差分 ---------- #
    @staticmethod
    def _parse_window(body: Dict[str, Any], predictor) -> tuple:
        """解析 window [W,RAW]|[N,W,RAW] 与可选 scene_params, 返回 (window[3D], sp)。"""
        seq = body.get("window")
        if seq is None:
            raise ValueError("缺少字段 'window' (形状 [W,RAW] 或 [N,W,RAW])")
        try:
            window = torch.tensor(seq, dtype=torch.float32)
        except Exception as e:
            raise ValueError(f"window 无法解析为张量: {e}")
        if window.dim() == 2:
            window = window.unsqueeze(0)
        if window.dim() != 3:
            raise ValueError("window 需为 [W,RAW] 或 [N,W,RAW]")
        if window.size(-1) != predictor.raw_dim:
            raise ValueError(
                f"window 最后一维 {window.size(-1)} 应为 {predictor.raw_dim}")
        sp = body.get("scene_params")
        if sp is not None:
            try:
                sp = torch.tensor(sp, dtype=torch.float32)
            except Exception as e:
                raise ValueError(f"scene_params 无法解析为张量: {e}")
            if sp.dim() == 1:
                sp = sp.unsqueeze(0)
        return window, sp

    def counterfactual(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """POST /counterfactual: 在当前预测器上做干预式反事实 rollout。"""
        predictor = self._require_predictor()
        window, sp = self._parse_window(body, predictor)
        horizon = int(body.get("horizon", 2))
        if not (1 <= horizon <= 8):
            raise ValueError("horizon 需在 [1,8]")
        intervention = body.get("intervention")
        if intervention is not None and not isinstance(intervention, dict):
            raise ValueError("intervention 需为 dict 或 null")
        with self._lock:
            out = CounterfactualEngine(predictor).counterfactual(
                window, horizon, scene_params=sp, intervention=intervention)
        return {
            "status": "ok", "version": __version__, "horizon": horizon,
            "baseline": out["baseline"].tolist(),
            "counterfactual": out["counterfactual"].tolist(),
            "ate_by_step": [round(float(x), 6) for x in out["ate_by_step"]],
            "ate_mean": round(out["ate_mean"], 6),
            "final_state_diff": out["final_state_diff"].tolist(),
            "intervention": intervention,
        }

    def identify(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """POST /identify: 仅凭观测窗口反推隐藏场景物理参数。"""
        predictor = self._require_predictor()
        window, _ = self._parse_window(body, predictor)
        horizon = int(body.get("horizon", 2))
        grid_size = int(body.get("grid_size", 5))
        if not (1 <= horizon <= window.size(1) - 1):
            raise ValueError("horizon 需在 [1, window-1]")
        if not (2 <= grid_size <= 12):
            raise ValueError("grid_size 需在 [2,12]")
        with self._lock:
            out = SceneParameterIdentifier(
                predictor, grid_size=grid_size).identify(window, horizon=horizon)
        return {
            "status": "ok", "version": __version__,
            "identified_params": [round(float(x), 6)
                                  for x in out["identified_params"]],
            "param_names": out["param_names"],
            "loss_min": round(out["loss_curve_min"], 6),
            "n_grid": out["n_grid"],
        }

    def risk(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """POST /risk: 聚合区间宽/OOD/置信为风险分与等级。"""
        predictor = self._require_predictor()
        window, sp = self._parse_window(body, predictor)
        horizon = int(body.get("horizon", 1))
        if not (1 <= horizon <= 4):
            raise ValueError("horizon 需在 [1,4]")
        with self._lock:
            out = RiskGrader().grade(predictor, window, scene_params=sp,
                                      horizon=horizon)
        return {"status": "ok", "version": __version__,
                "risk_score": out["risk_score"],
                "risk_level": out["risk_level"],
                "components": out["components"]}

    def diff_checkpoints(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """POST /diff-checkpoints: 白名单内两 checkpoint 数值对比 (不需当前预测器)。"""
        name_a = body.get("name_a")
        name_b = body.get("name_b")
        if not name_a or not name_b:
            raise ValueError("缺少字段 'name_a' / 'name_b'")
        path_a = self._safe_checkpoint_path(name_a)
        path_b = self._safe_checkpoint_path(name_b)
        n_per_kind = int(body.get("n_per_kind", 16))
        if not (1 <= n_per_kind <= 128):
            raise ValueError("n_per_kind 需在 [1,128]")
        with self._lock:
            res = compare_checkpoints(path_a, path_b,
                                      n_per_kind=n_per_kind)
        # a/b_metrics 已是 JSON 友好 (全浮点/列表); 直接透传
        return {"status": "ok", "version": __version__,
                **{k: v for k, v in res.items()}}

    def health(self) -> Dict[str, Any]:
        # DIAG-008: 读 scene_memory 需与写操作 (internalize/reset) 共用同一把锁,
        # 避免并发迭代 dict 视图; 仅在锁内做 O(1) 拷贝, 不持锁做耗时操作。
        with self._lock:
            scenes = list(self.engine.scene_memory.keys())
        return {"status": "ok", "service": "udos-reasoning-engine",
                "version": __version__,
                "predictor_trained": self.trained,
                "internalized_scenes": scenes}

    # ---------- v4.5.4 开源资源注册表 ---------- #
    def _get_registry(self):
        if self._registry is None:
            with self._lock:
                if self._registry is None:
                    self._registry = build_default_registry(self._resource_profile)
        return self._registry

    def resources_list(self, query: str = "") -> Dict[str, Any]:
        """GET /resources[?...]: 列注册表, 可按 profile/kind/status/license/priority/level 过滤。"""
        from urllib.parse import parse_qs
        q = parse_qs(query or "")
        reg = self._get_registry()
        rows = reg.list(
            profile=(q.get("profile", [None])[0]),
            kind=(q.get("kind", [None])[0]),
            status=(q.get("status", [None])[0]),
            license=(q.get("license", [None])[0]),
            priority=(q.get("priority", [None])[0]),
            level=(q.get("level", [None])[0]))
        return {"status": "ok", "version": __version__,
                "profile": reg.profile,
                "summary": reg.summary(profile=(q.get("profile", [None])[0])),
                "resources": rows}

    def resources_set_profile(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """POST /resources/profile {profile: performance|full}: 切换视图 profile。
        切换不改主权重、不改默认数值, 仅重算可见子集与探测缓存。"""
        profile = body.get("profile")
        if profile not in ("performance", "full"):
            raise ValueError("profile 需为 performance|full")
        reg = self._get_registry()
        reg.apply_profile(profile)
        return {"status": "ok", "version": __version__,
                "profile": reg.profile, "summary": reg.summary()}

    def resource_probe(self, rid: str) -> Dict[str, Any]:
        """POST /resources/{id}/probe: 真实能力探测 (带缓存)。未知 id -> KeyError(404)。"""
        reg = self._get_registry()
        return reg.probe(rid)

    def resource_invoke(self, rid: str, body: Dict[str, Any]) -> Dict[str, Any]:
        """POST /resources/{id}/invoke: 统一调用。
        未知 id -> 404; 缺 action -> 400; L3/缺失 -> 503 (ResourceUnavailable)。"""
        reg = self._get_registry()
        action = body.get("action", "")
        return reg.invoke(rid, action, body.get("params") or {})

    # ---------- v5.0.1 安全 / 备份 / 受控自治 ---------- #
    def _get_auth(self):
        if self._auth is None:
            with self._lock:
                if self._auth is None:
                    from .auth import AuthService
                    from .authz import AuthzMiddleware
                    self._auth = AuthService()
                    self._authz = AuthzMiddleware(self._auth)
        return self._auth

    def auth_bootstrap(self, body):
        from .auth import AuthError
        auth = self._get_auth()
        auth.bootstrap_owner(body.get("username", ""), body.get("password", ""))
        return {"status": "ok", "message": "owner 已初始化"}

    def auth_login(self, body):
        from .auth import AuthError
        auth = self._get_auth()
        token, info = auth.authenticate(body.get("username", ""),
                                        body.get("password", ""))
        return {"status": "ok", "token": token, "user": info}

    def auth_list_users(self, principal):
        return {"status": "ok", "users": self._get_auth().list_users()}

    def auth_audit(self, principal):
        return {"status": "ok", "events": self._get_auth().audit.list()}

    def _get_backup(self, body=None):
        import os, tempfile
        if self._backup is None:
            from .backup import BackupManager
            tdir = (os.environ.get("UDOS_BACKUP_DIR")
                    or tempfile.mkdtemp(prefix="udos_backup_"))
            pwd = os.environ.get("UDOS_BACKUP_PASSWORD") or "demo-recover-roundtrip"
            self._backup = BackupManager(tdir, pwd)
        return self._backup

    def backup_run(self, principal):
        b = self._get_backup()
        fn = b.create("manual", {"version": __version__, "ts": time.time()})
        return {"status": "ok", "file": fn, "kept": b.list()}

    def backup_verify(self, principal):
        b = self._get_backup()
        out = b.restore("manual")
        return {"status": "ok", "roundtrip_version": out.get("version")}

    def _get_autonomy(self):
        if self._autonomy is None:
            from .autonomy import AutonomyLoop
            a = AutonomyLoop()
            a.set_baseline(float(self.last_train.get("mse",
                             self.last_train.get("eval_mse", 0.045556))))
            self._autonomy = a
        return self._autonomy

    def autonomy_control(self, body):
        a = self._get_autonomy()
        op = body.get("op")
        if op == "start":
            a.enable(); return {"status": "ok", "state": "enabled"}
        if op == "stop":
            a.disable(); return {"status": "ok", "state": "disabled"}
        if op == "kill":
            a.kill(); return {"status": "ok", "state": "killed"}
        if op == "cycle":
            out = a.run_one_cycle(bool(body.get("data_ok", True)),
                                  float(body.get("candidate_mse", 9.9)))
            return {"status": "ok", "cycle": out}
        raise ValueError("op 需为 start|stop|kill|cycle")

    # ---------- v5.0.2 情报分析服务(9 锁定端点) ---------- #
    def _get_intel(self):
        if self._intel is None:
            from .intelligence.api import IntelAPI
            self._intel = IntelAPI()
        return self._intel

    # ---------- v5.1.0 KV Cache 分层卸载(opt-in) ---------- #
    def _kvcache_enabled(self) -> bool:
        import os
        return os.environ.get("UDOS_KVCACHE", "off") in ("on", "1")

    def kvcache_sim_run(self, body):
        if not self._kvcache_enabled():
            raise RuntimeError("UDOS_KVCACHE=off; opt-in 未启用")
        from .kvcache import run_simulation
        self._last_kvcache_sim = run_simulation(body)
        return self._last_kvcache_sim

    def kvcache_state(self):
        if not self._kvcache_enabled():
            raise RuntimeError("UDOS_KVCACHE=off; opt-in 未启用")
        from .kvcache import qat_status, gpu_available
        return {"enabled": True, "gpu_available": gpu_available(),
                "qat": qat_status()}

    def kvcache_cost(self, body):
        if not self._kvcache_enabled():
            raise RuntimeError("UDOS_KVCACHE=off; opt-in 未启用")
        from .kvcache.cost_ledger import cost_retain_vs_recompute
        for k in ("tier", "hit_rate", "recall_cost", "recompute_cost"):
            if k not in body:
                raise ValueError(f"缺字段 {k}")
        return cost_retain_vs_recompute(body["tier"], float(body["hit_rate"]),
                                       float(body["recall_cost"]),
                                       float(body["recompute_cost"]))

    def kvcache_metrics_text(self) -> str:
        if not self._kvcache_enabled():
            raise RuntimeError("UDOS_KVCACHE=off; opt-in 未启用")
        s = getattr(self, "_last_kvcache_sim", None) or {
            "hit_rate": 0.0, "recompute_units": 0, "ttft_proxy_ms": 0.0,
            "compression_ratio": 1.0}
        lines = [
            "# HELP udos_kvcache_hit_rate KV cache hit rate (synthetic seed)",
            "# TYPE udos_kvcache_hit_rate gauge",
            f"udos_kvcache_hit_rate {s.get('hit_rate', 0.0)}",
            f"udos_kvcache_recompute_units {s.get('recompute_units', 0)}",
            f"udos_kvcache_ttft_proxy_ms {s.get('ttft_proxy_ms', 0.0)}",
            f"udos_kvcache_compression_ratio {s.get('compression_ratio', 1.0)}",
        ]
        return "\n".join(lines) + "\n"

    def intel_infrastructure(self):
        from .kvcache import qat_status, gpu_available
        from .kvcache.sim import run_simulation
        enabled = self._kvcache_enabled()
        s = getattr(self, "_last_kvcache_sim", None)
        if s is None:
            try:
                s = run_simulation({"n_tokens": 1000, "seed": 7})
            except Exception:
                s = {"hit_rate": 0.0, "recompute_units": 0,
                     "ttft_proxy_ms": 0.0}
        return {
            "kvcache_enabled": enabled,
            "gpu_available": gpu_available(),
            "qat": qat_status()["qat"],
            "hit_rate": s.get("hit_rate"),
            "recompute_units": s.get("recompute_units"),
            "ttft_proxy_ms": s.get("ttft_proxy_ms"),
            "mode": "on" if enabled else "off-static-synthetic",
            "disclaimer": "CPU 机制原型、非真实 GPU/QAT 数据、非厂商数字",
        }

    # ---------- v5.3.0 类脑树突(opt-in UDOS_BRAIN) ---------- #
    def _brain_enabled(self) -> bool:
        import os
        return os.environ.get("UDOS_BRAIN", "off") in ("on", "1")

    def brain_sim_run(self, body):
        if not self._brain_enabled():
            raise RuntimeError("UDOS_BRAIN=off; opt-in 未启用")
        from .dendrite.brain import brain_sim
        seed = body.get("seed", 7)
        if not isinstance(seed, int):
            raise ValueError("seed 须为整数")
        return brain_sim(seed)

    def brain_dhs_bench(self, body):
        if not self._brain_enabled():
            raise RuntimeError("UDOS_BRAIN=off; opt-in 未启用")
        from .dendrite.brain import brain_dhs
        wc = body.get("worker_count", 4)
        if not isinstance(wc, int) or wc <= 0:
            raise ValueError("worker_count 须为正整数")
        return brain_dhs(None, wc)

    def brain_robust_run(self, body):
        if not self._brain_enabled():
            raise RuntimeError("UDOS_BRAIN=off; opt-in 未启用")
        from .dendrite.brain import brain_robustness
        return brain_robustness(int(body.get("seed", 7)))

    def intel_brain(self):
        from .dendrite.brain import brain_status
        return brain_status()

    # ---------- v5.4.3 精细生物物理内核(opt-in UDOS_FINESIM) ---------- #
    def _finesim_enabled(self) -> bool:
        import os
        return os.environ.get("UDOS_FINESIM", "off") in ("on", "1")

    def _finesim_guard(self):
        if not self._finesim_enabled():
            raise RuntimeError("UDOS_FINESIM=off; opt-in 未启用")

    def finesim_cable(self, body):
        self._finesim_guard()
        from .finesim.cable import cable_params, verify_lambda
        return {"params": cable_params(), "verify": verify_lambda()}

    def finesim_hh(self, body):
        self._finesim_guard()
        from .finesim.hh import simulate, f_I_curve
        I = float(body.get("current", 10.0))
        return {"step": simulate(I), "f_I_curve": f_I_curve()}

    def finesim_hines_dhs(self, body):
        self._finesim_guard()
        from .finesim.hines_dhs import compare
        return compare()

    def finesim_nmda(self, body):
        self._finesim_guard()
        from .finesim.nmda import falsifiable_control
        return falsifiable_control()

    def finesim_payeur(self, body):
        self._finesim_guard()
        from .finesim.payeur import payeur_demo
        return payeur_demo()

    def finesim_robustness(self, body):
        self._finesim_guard()
        from .finesim.payeur import synaptic_position_robustness
        return synaptic_position_robustness()

    def intel_finesim(self):
        return {"enabled": self._finesim_enabled(), "gpu_available": False,
                "neural_simulator": "ENV_BLOCKED",
                "note": "CPU 参考/类比内核, 非 NEURON 运行",
                "disclaimer": "analogy not reproduction"}

    # ---------- v2.7.0.dev6 MPC / 在线 / 主动 / 实验 ---------- #
    def policy_select(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """POST /policy/select: MPC 式候选动作 rollout 优选 + 全排序。"""
        predictor = self._require_predictor()
        window, sp = self._parse_window(body, predictor)
        actions = body.get("actions")
        if actions is None:
            raise ValueError("缺少字段 'actions' (候选动作 dict 列表)")
        if not isinstance(actions, list):
            raise ValueError("'actions' 需为 dict 列表")
        horizon = int(body.get("horizon", 4))
        if not (1 <= horizon <= 8):
            raise ValueError("horizon 需在 [1,8]")
        lambda_risk = float(body.get("lambda_risk", 1.0))
        with self._lock:
            out = MPCActionSelector(predictor, horizon=horizon,
                                    lambda_risk=lambda_risk).select(
                window, scene_params=sp, candidate_actions=actions)
        return {
            "status": "ok", "version": __version__, "horizon": horizon,
            "lambda_risk": lambda_risk,
            "best_action": out["best_action"],
            "best_score": out["best_score"],
            "best_index": out["best_index"],
            "ranked_actions": out["ranked_actions"],
            "no_valid_action": out["no_valid_action"],
        }

    def online_adapt(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """POST /online/adapt: 推入观测窗口, 漂移触发时再校准 (默认不改权重)。"""
        predictor = self._require_predictor()
        seq = body.get("window")
        if seq is None:
            raise ValueError("缺少字段 'window' (形状 [W,RAW] 或 [N,W,RAW])")
        try:
            window = torch.tensor(seq, dtype=torch.float32)
        except Exception as e:
            raise ValueError(f"window 无法解析为张量: {e}")
        if window.dim() == 2:
            window = window.unsqueeze(0)
        if window.dim() != 3:
            raise ValueError("window 需为 [W,RAW] 或 [N,W,RAW]")
        if window.size(-1) != predictor.raw_dim:
            raise ValueError(
                f"window 最后一维 {window.size(-1)} 应为 {predictor.raw_dim}")
        enable_ft = bool(body.get("enable_finetune", False))
        ft_epochs = int(body.get("finetune_epochs", 3))
        if not (0 <= ft_epochs <= 10):
            raise ValueError("finetune_epochs 需在 [0,10]")
        with self._lock:
            if self._online is None:
                ref = build_parametric_dataset(
                    n_per_kind=16, n_steps=14, window=6, horizon=4,
                    dt=0.5, seed=90210).X
                self._online = OnlineAdapter(
                    ref, enable_finetune=enable_ft, finetune_epochs=ft_epochs)
            self._online.enable_finetune = enable_ft
            self._online.finetune_epochs = ft_epochs
            self._online.observe(window)
            drifted_before = self._online.is_drifted()
            drift = self._online.drift_score()
            # 校准集: 独立种子小样本 (与训练分布同口径), 仅在触发时实际使用
            cal_ds = build_parametric_dataset(
                n_per_kind=16, n_steps=14, window=6, horizon=4,
                dt=0.5, seed=5123)
            res = self._online.check_and_adapt(predictor, cal_ds)
        return {
            "status": "ok", "version": __version__,
            "adapted": bool(res.get("adapted", False)),
            "reason": res.get("reason"),
            "drift_score": None if (drift != drift) else round(float(drift), 6),
            "detected_before_adapt": drifted_before,
            "weights_modified": bool(res.get("weights_modified", False)),
            "log": res.get("log"),
            "log_length": len(self._online.adaptation_log),
        }

    def active_sample(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """POST /active/sample: 对样本池算信息增益分并选 top-K。"""
        predictor = self._require_predictor()
        pool = body.get("sample_pool")
        if pool is None:
            raise ValueError("缺少字段 'sample_pool' (形状 [N,W,RAW])")
        try:
            pool_t = torch.tensor(pool, dtype=torch.float32)
        except Exception as e:
            raise ValueError(f"sample_pool 无法解析为张量: {e}")
        if pool_t.dim() != 3:
            raise ValueError("sample_pool 需为 [N,W,RAW]")
        if pool_t.size(0) == 0:
            raise ValueError("sample_pool 不能为空")
        if pool_t.size(-1) != predictor.raw_dim:
            raise ValueError(
                f"sample_pool 最后一维 {pool_t.size(-1)} 应为 {predictor.raw_dim}")
        k = body.get("k")
        if k is None:
            raise ValueError("缺少字段 'k'")
        k = int(k)
        if k <= 0:
            raise ValueError("k 需为正整数")
        sp = body.get("scene_params")
        if sp is not None:
            try:
                sp = torch.tensor(sp, dtype=torch.float32)
            except Exception as e:
                raise ValueError(f"scene_params 无法解析为张量: {e}")
        with self._lock:
            indices, scores = UncertaintySampler().select_top_k(
                predictor, pool_t, k, scene_params=sp)
        return {
            "status": "ok", "version": __version__,
            "pool_size": int(pool_t.size(0)), "k": int(k),
            "indices": [int(i) for i in indices],
            "scores": [round(float(s), 8) for s in scores],
        }

    def experiments(self) -> Dict[str, Any]:
        """GET /experiments: 从注册表 JSON 读取实验列表 (不存在则空)。"""
        reg = ExperimentRegistry().load()
        return {"status": "ok", "version": __version__,
                "registry_path": reg.path,
                "count": len(reg), "experiments": reg.list()}

    # ---------- v2.8.2 Physical Loop / MultiTask 服务端点 ---------- #
    def loop_step(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """POST /loop/step: 单步 Physical Loop 闭环 (未训练 409, 非法输入 400)。"""
        predictor = self._require_predictor()
        window, sp = self._parse_window(body, predictor)
        horizon = int(body.get("horizon", 4))
        if not (1 <= horizon <= 8):
            raise ValueError("horizon 需在 [1,8]")
        actions = body.get("actions")
        if actions is not None and not isinstance(actions, list):
            raise ValueError("'actions' 需为 dict 列表")
        with self._lock:
            if self._loop is None:
                self._loop = PhysicalLoopRunner(predictor, horizon=horizon)
            out = self._loop.run(window, scene_params=sp,
                                 candidate_actions=actions)
        return {
            "status": "ok", "version": __version__, "horizon": horizon,
            "prediction": out["prediction"].tolist(),
            "prediction_shape": list(out["prediction"].shape),
            "steps": [s["name"] for s in out["loop_state"]["steps"]],
            "action_history_len": len(self._loop.loop_state["action_history"]),
            "correction": self._loop.loop_state["correction"],
        }

    def multitask_predict(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """POST /multitask/predict: 多任务三头联合推理 (未训练 409, 非法输入 400)。"""
        predictor = self._require_predictor()
        window, sp = self._parse_window(body, predictor)
        with self._lock:
            if self._mth is None:
                mth = MultiTaskHead(predictor, latent_dim=32, enable=True)
                torch.manual_seed(0)
                mth.register_head("spatial", SpatialCoordHead(32, n_pts=4))
                mth.register_head("action", ActionTrajectoryHead(
                    32, horizon=4, action_dim=predictor.raw_dim))
                mth.register_head("future", FutureStateHead(
                    32, horizon=4, state_dim=predictor.raw_dim))
                self._mth = mth
            out = self._mth.forward(window, scene_params=sp)
        return {
            "status": "ok", "version": __version__,
            "spatial": out["spatial"].tolist(),
            "action": out["action"].tolist(),
            "future": {"trajectory": out["future"]["trajectory"].tolist(),
                       "uncertainty": out["future"]["uncertainty"].tolist()},
        }

    # ---------- v2.9.2 retargeting / affordance 服务端点 ---------- #
    @staticmethod
    def _morph_from_body(d: Dict[str, Any], key: str) -> MorphologyConfig:
        """从请求体构造 MorphologyConfig: 支持预设名或显式 dict。"""
        if not isinstance(d, dict) or key not in d:
            raise ValueError(f"缺少形态字段 {key!r}")
        spec = d[key]
        if isinstance(spec, str):
            return MorphologyLibrary().get(spec)
        if not isinstance(spec, dict):
            raise ValueError(f"{key} 须为预设名字符串或形态 dict")
        try:
            return MorphologyConfig(
                dof=int(spec["dof"]), control_freq=float(spec["control_freq"]),
                joint_limits=spec.get("joint_limits"),
                kinematics=spec.get("kinematics"),
                name=spec.get("name", key))
        except KeyError as e:
            raise ValueError(f"{key} 缺字段: {e}")

    def retarget_convert(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """POST /retarget/convert: 动作重定向 (未训练 409, 非法输入 400)。"""
        self._require_predictor()
        actions = body.get("actions")
        if actions is None:
            raise ValueError("缺少字段 'actions'")
        try:
            a = torch.as_tensor(actions, dtype=torch.float32)
        except Exception as e:
            raise ValueError(f"actions 无法解析: {e}")
        src = self._morph_from_body(body, "source")
        tgt = self._morph_from_body(body, "target")
        try:
            rt = ActionRetargeter(src, tgt)
            out = rt.retarget(a)
        except ValueError as e:
            raise ValueError(f"重定向失败: {e}")
        return {
            "status": "ok", "version": __version__,
            "src_dof": src.dof, "tgt_dof": tgt.dof,
            "actions": out.tolist(),
            "shape": list(out.shape),
        }

    def affordance_score(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """POST /affordance/score: 可供性打分 (未训练 409, 非法输入 400)。"""
        self._require_predictor()
        state = body.get("state")
        objects = body.get("objects")
        if state is None or objects is None:
            raise ValueError("缺少字段 'state' / 'objects'")
        try:
            s = torch.as_tensor(state, dtype=torch.float32)
            o = torch.as_tensor(objects, dtype=torch.float32)
        except Exception as e:
            raise ValueError(f"输入无法解析: {e}")
        radius = float(body.get("reach_radius", 2.0))
        scorer = AffordanceScorer(reach_radius=radius)
        res = scorer.score(s, o)
        return {
            "status": "ok", "version": __version__,
            "scores": res["scores"].tolist(),
            "best_part": [int(b) for b in res["best_part"]],
            "reachable": res["reachable"].tolist(),
            "suggestion": res["suggestion"],
        }

    # ---------- v3.5.1 SFM 空间查询 / 碰撞服务端点 (零外挂, 不需训练) ---------- #
    @staticmethod
    def _build_spatial_scene(objects: Any) -> SpatialScene:
        """从请求体 objects 列表构造 SpatialScene; 非法 -> ValueError(400)。"""
        if not isinstance(objects, list) or len(objects) == 0:
            raise ValueError("缺少字段 'objects' (非空列表)")
        sc = SpatialScene()
        for od in objects:
            if not isinstance(od, dict) or "object_id" not in od \
                    or "position" not in od:
                raise ValueError("每个物体需 object_id 与 position 字段")
            try:
                sc.add(SpatialObject(
                    object_id=od["object_id"], position=od["position"],
                    velocity=od.get("velocity", (0.0, 0.0, 0.0)),
                    radius=float(od.get("radius", 0.5))))
            except (ValueError, TypeError) as e:
                raise ValueError(f"物体 {od.get('object_id')!r} 非法: {e}")
        return sc

    def _resolve_scene(self, body: Dict[str, Any]) -> SpatialScene:
        """请求体带 objects 则临时构造; 否则用服务注册场景; 都无 -> 409。"""
        if "objects" in body:
            return self._build_spatial_scene(body["objects"])
        if self._spatial_scene is None:
            raise ServiceNotReady(
                "未提供 objects 且未注册空间场景 (POST 带 objects 或先注册)")
        return self._spatial_scene

    def spatial_query(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """POST /spatial/query: 空间查询 (range/box/raycast/los)。

        字段: op(默认range) + objects 可选 + op 相关参数。
        非法输入 400; 无场景 409; 未知路由 404。
        """
        scene = self._resolve_scene(body)
        op = body.get("op", "range")
        q = SpatialQueryEngine(scene)
        if op == "range":
            center = body.get("center")
            radius = body.get("radius")
            if center is None or radius is None:
                raise ValueError("range 需 center/radius")
            res = q.range_search(center, float(radius))
            return {"status": "ok", "version": __version__, "op": op,
                    "results": res}
        if op == "box":
            lo = body.get("box_min")
            hi = body.get("box_max")
            if lo is None or hi is None:
                raise ValueError("box 需 box_min/box_max")
            return {"status": "ok", "version": __version__, "op": op,
                    "results": q.box_query(lo, hi)}
        if op == "raycast":
            o = body.get("origin")
            d = body.get("direction")
            if o is None or d is None:
                raise ValueError("raycast 需 origin/direction")
            return {"status": "ok", "version": __version__, "op": op,
                    **q.raycast(o, d)}
        if op == "los":
            a, b = body.get("a"), body.get("b")
            if a is None or b is None:
                raise ValueError("los 需 a/b 物体 id")
            return {"status": "ok", "version": __version__, "op": op,
                    **q.line_of_sight(a, b)}
        raise ValueError(f"未知 op {op!r}; 支持 range/box/raycast/los")

    def spatial_collision(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """POST /spatial/collision: 场景接触检测。

        字段: objects 可选; 非法 400; 无场景 409。
        """
        scene = self._resolve_scene(body)
        contacts = self._collision.detect_contacts(scene)
        return {"status": "ok", "version": __version__,
                "n_objects": len(scene), "contacts": contacts,
                "n_contacts": len(contacts)}

    # ---------- v3.0.1 future_multimodal / eval5d 服务端点 ---------- #
    def future_predict(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """POST /future/predict: 多模态未来预测 (未训练 409, 非法输入 400)。"""
        predictor = self._require_predictor()
        window, sp = self._parse_window(body, predictor)
        horizon = int(body.get("horizon", 4))
        if not (1 <= horizon <= 8):
            raise ValueError("horizon 需在 [1,8]")
        with self._lock:
            if self._fmh is None:
                torch.manual_seed(0)
                self._fmh = FutureMultimodalHead(
                    32, horizon=horizon, rgb_dim=8, depth_dim=4, mask_dim=4)
            latent = MultiTaskHead(predictor, latent_dim=32).encode(
                window, scene_params=sp)
            out = self._fmh(latent, scene_params=sp)
        return {
            "status": "ok", "version": __version__, "horizon": horizon,
            "rgb": out["rgb"].tolist(),
            "depth": out["depth"].tolist(),
            "mask": out["mask"].tolist(),
            "shapes": {k: list(v.shape) for k, v in out.items()},
        }

    def eval_5d(self) -> Dict[str, Any]:
        """GET /eval/5d: UDOS 内部五维评测 (未训练 409)。

        **UDOS 内部基准, 非 PhysBrain 榜单分数**。
        """
        predictor = self._require_predictor()
        with self._lock:
            if self._eval is None:
                self._eval = FiveDimensionEvaluator(predictor, seed=2025)
            scores = self._eval.evaluate()
            composite = self._eval.composite_score(scores)
            md = self._eval.metadata()
        return {
            "status": "ok", "version": __version__,
            "scores": scores, "composite": round(composite, 2),
            "is_internal_benchmark": md["is_internal_benchmark"],
            "not_physbrain_leaderboard": md["not_physbrain_leaderboard"],
        }

    # ---------- v3.1.2 ActionPiece tokenize / detokenize 端点 ---------- #
    def _ensure_action_tokenizer(self, action_dim: int):
        """懒构造合成动作 k-means tokenizer (analogy, 非真机)。"""
        if self._tokenizer is None or self._tokenizer.action_dim != action_dim:
            from .action_piece import ActionPieceTokenizer
            g = torch.Generator().manual_seed(0)
            synth = torch.randn(400, action_dim, generator=g)
            self._tokenizer = ActionPieceTokenizer(
                action_dim, codebook_size=16, seed=42).fit(synth)
        return self._tokenizer

    def action_tokenize(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """POST /action/tokenize: 连续动作 -> 离散 token (未训练 409, 非法 400)。"""
        self._require_predictor()
        actions = body.get("actions")
        if actions is None:
            raise ValueError("缺少字段 'actions'")
        try:
            a = torch.as_tensor(actions, dtype=torch.float32)
        except Exception as e:
            raise ValueError(f"actions 无法解析: {e}")
        if a.dim() == 1:
            a = a.unsqueeze(0)
        with self._lock:
            tok = self._ensure_action_tokenizer(a.size(1))
            ids = tok.encode(a)
        return {
            "status": "ok", "version": __version__,
            "tokens": ids.tolist(), "shape": list(a.shape),
            "codebook_size": tok.codebook_size,
        }

    def action_detokenize(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """POST /action/detokenize: 离散 token -> 连续动作 (未训练 409, 非法 400)。"""
        self._require_predictor()
        tokens = body.get("tokens")
        if tokens is None:
            raise ValueError("缺少字段 'tokens'")
        try:
            t = torch.as_tensor(tokens, dtype=torch.long).reshape(-1)
        except Exception as e:
            raise ValueError(f"tokens 无法解析: {e}")
        with self._lock:
            # 用已构造 tokenizer 的动作维度; 若未构造则按 6 维默认
            dim = getattr(self._tokenizer, "action_dim", 6)
            tok = self._ensure_action_tokenizer(dim)
            out = tok.decode(t)
        return {
            "status": "ok", "version": __version__,
            "actions": out.tolist(), "shape": list(out.shape),
        }

    # ---------- v3.2.2 Ego360 / ICL 端点 ---------- #
    def augment_generate(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """POST /augment/generate: 对输入状态序列做 Ego360 启发数据增强。

        无状态 (不需已训练模型); 非法输入 400。
        """
        from .ego_data import SyntheticEgoAugmenter
        window = body.get("window")
        if window is None:
            raise ValueError("缺少字段 'window'")
        try:
            w = torch.as_tensor(window, dtype=torch.float32)
        except Exception as e:
            raise ValueError(f"window 无法解析: {e}")
        if w.dim() == 1:
            w = w.unsqueeze(0)
        aug = SyntheticEgoAugmenter(
            view_rotate_deg=float(body.get("view_rotate_deg", 0.0)),
            view_translate_xyz=body.get("view_translate_xyz", (0.0, 0.0, 0.0)),
            traj_perturb=float(body.get("traj_perturb", 0.0)),
            noise_sigma=float(body.get("noise_sigma", 0.0)),
            time_scale=float(body.get("time_scale", 1.0)),
            seed=int(body.get("seed", 42)))
        with self._lock:
            out = aug.augment_sequence(w)
        return {
            "status": "ok", "version": __version__,
            "augmented": out.tolist(), "shape": list(out.shape),
            "config": aug.describe(),
        }

    def icl_predict(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """POST /icl/predict: few-shot 上下文注入下一步预测 (未训练 409, 非法 400)。"""
        from .incontext import InContextLearner
        predictor = self._require_predictor()
        window = body.get("window")
        if window is None:
            raise ValueError("缺少字段 'window'")
        try:
            w = torch.as_tensor(window, dtype=torch.float32)
        except Exception as e:  # DIAG-003: 与 /predict 等端点对齐, 非法类型 -> 400
            raise ValueError(f"非法 window: {e}")
        if w.dim() == 1:
            w = w.unsqueeze(0)
        examples = body.get("examples")
        ex_t = ([torch.as_tensor(e, dtype=torch.float32) for e in examples]
                if examples else None)
        task_desc = body.get("task_desc")
        td = torch.as_tensor(task_desc, dtype=torch.float32) if task_desc else None
        sp = body.get("scene_params")
        sp_t = torch.as_tensor(sp, dtype=torch.float32) if sp else None
        with self._lock:
            icl = InContextLearner(predictor)
            pred = icl.predict(w, examples=ex_t, task_desc=td, scene_params=sp_t)
        return {
            "status": "ok", "version": __version__,
            "prediction": pred.tolist(), "shape": list(pred.shape),
        }

    # ------------------------------------------------------------------ #
    # v3.4.1 ICM 上下文记忆端点 (避开既有 /icl/predict)
    # ------------------------------------------------------------------ #
    def _get_predictor(self):
        return self._require_predictor()

    def icm_demo_register(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """POST /icm/demo/register: 注册一条演示到服务级 ICM 记忆库。

        字段: input_window[W,6], result[6], action[6] 可选, scene_params[4] 可选,
              kind 可选。未训练 -> 409; 非法输入 -> 400。
        """
        from .icm import DemonstrationEpisode, DemonstrationMemory, ICMAggregator
        predictor = self._require_predictor()
        win = body.get("input_window")
        result = body.get("result")
        if win is None or result is None:
            raise ValueError("缺少字段 'input_window' / 'result'")
        try:
            w = torch.as_tensor(win, dtype=torch.float32)
            r = torch.as_tensor(result, dtype=torch.float32)
        except Exception as e:
            raise ValueError(f"非法 input_window/result: {e}")
        sp = body.get("scene_params")
        sp_t = torch.as_tensor(sp, dtype=torch.float32) if sp is not None else None
        with self._lock:
            if self._icm_memory is None:
                self._icm_memory = DemonstrationMemory()
            if self._icm_agg is None:
                self._icm_agg = ICMAggregator(predictor)
            ep = DemonstrationEpisode(w, r,
                                      action=body.get("action"),
                                      scene_params=sp_t,
                                      kind=body.get("kind", "unspecified"))
            self._icm_memory.register(ep)
            self._icm_agg.cache_residual(ep, scene_params=sp_t)
            n = self._icm_memory.size
        logger.info("ICM demo registered, memory size=%d", n)
        return {"status": "ok", "version": __version__, "memory_size": n}

    def icm_predict(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """POST /icm/predict: 演示条件化预测 (检索+聚合, 零梯度)。

        字段: window[W,6], k 可选(默认3), scene_params[4] 可选。
        未训练 -> 409; 未注册任何演示 -> 409; 非法输入 -> 400。
        """
        predictor = self._require_predictor()
        if self._icm_memory is None or self._icm_memory.size == 0:
            raise ServiceNotReady("ICM 记忆库为空, 请先 POST /icm/demo/register")
        window = body.get("window")
        if window is None:
            raise ValueError("缺少字段 'window'")
        try:
            w = torch.as_tensor(window, dtype=torch.float32)
        except Exception as e:
            raise ValueError(f"非法 window: {e}")
        if w.dim() == 2:
            w = w.unsqueeze(0)
        k = self._coerce_int(body.get("k"), 3, "k")
        if k < 0:
            raise ValueError("k 须 >= 0")
        sp = body.get("scene_params")
        sp_t = torch.as_tensor(sp, dtype=torch.float32) if sp is not None else None
        with self._lock:
            if self._icm_agg is None:
                from .icm import ICMAggregator
                self._icm_agg = ICMAggregator(predictor)
            pred = self._icm_agg.predict(w[0], memory=self._icm_memory,
                                          k=k, scene_params=sp_t)
        return {
            "status": "ok", "version": __version__,
            "prediction": pred.tolist(), "shape": list(pred.shape),
            "memory_size": self._icm_memory.size, "k": k,
        }

    # ---------- v3.6.1 PWM 潜在世界模型端点 ---------- #
    def _ensure_wm(self, predictor) -> LatentWorldModel:
        """懒构造+离线拟合外挂潜在世界模型 (服务级单例; 只读 predictor)。"""
        if self._wm is None:
            wm = LatentWorldModel(predictor)
            fit_ds = build_parametric_dataset(n_per_kind=16, n_steps=14,
                                              window=6, horizon=4, dt=0.5,
                                              seed=6060)
            wm.fit(fit_ds, epochs=20)
            self._wm = wm
            logger.info("WM latent model lazily initialized, n_params=%d",
                        wm.n_params)
        return self._wm

    def wm_imagine(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """POST /wm/imagine: 潜在空间多步想象 (未训练 409, 非法输入 400)。

        字段: window [W,6]|[N,W,6], horizon(1..16, 默认4), scene_params 可选。
        第 0 步逐位锚定主 predictor 单步; H>1 步为外挂潜在转移解码 (合成类比)。
        """
        predictor = self._require_predictor()
        window, sp = self._parse_window(body, predictor)
        horizon = self._coerce_int(body.get("horizon"), 4, "horizon")
        if not (1 <= horizon <= 16):
            raise ValueError("horizon 需在 [1,16]")
        with self._lock:
            wm = self._ensure_wm(predictor)
            roll = wm.imagine_rollout(window, horizon, scene_params=sp,
                                       compare_real=True)
        return {
            "status": "ok", "version": __version__, "horizon": horizon,
            "states": roll["states"].tolist(),
            "latents_shape": list(roll["latents"].shape),
            "step_mse_vs_real": [round(float(x), 6) for x in roll["step_mse"]],
            "wm_params": wm.n_params,
            "main_params_untouched": True,
            "analogy_not_reproduction": True,
        }

    def wm_conservation(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """POST /wm/conservation: 对 (真实 or 想象) rollout 做动量/能量守恒检验。

        字段: window, horizon(1..16, 默认4), scene_params 可选,
              source("real"|"imagine", 默认"real"), mass(默认1.0), spring_k 可选。
        未训练 409; 非法输入 400。
        """
        predictor = self._require_predictor()
        window, sp = self._parse_window(body, predictor)
        horizon = self._coerce_int(body.get("horizon"), 4, "horizon")
        if not (1 <= horizon <= 16):
            raise ValueError("horizon 需在 [1,16]")
        source = str(body.get("source", "real"))
        if source not in ("real", "imagine"):
            raise ValueError("source 须为 'real' 或 'imagine'")
        mass = self._coerce_float(body.get("mass"), 1.0, "mass")
        spring_k = body.get("spring_k")
        with self._lock:
            if source == "real":
                traj = predictor.rollout(window, horizon, scene_params=sp)
            else:
                wm = self._ensure_wm(predictor)
                traj = wm.imagine(window, horizon, scene_params=sp)
            rep = ConservationChecker(
                mass=mass,
                spring_k=float(spring_k) if spring_k is not None else None
            ).check(traj)
        r = rep["batch"][0]
        return {
            "status": "ok", "version": __version__, "horizon": horizon,
            "source": source,
            "momentum_violation": r["momentum_violation"],
            "momentum_step_delta": r["momentum_step_delta"],
            "energy_violation": r["energy_violation"],
            "conserved": r["conserved"],
            "momentum_per_step": r["momentum_per_step"],
            "energy_per_step": r["energy_per_step"],
        }

    # ---------- v3.7.1 分层神经控制 (大脑/小脑/脊髓) ---------- #
    def _ensure_neural(self, predictor) -> HierarchicalController:
        """懒构造分层神经控制器 (只读外挂, 不改主权重)。"""
        if self._neural is None:
            self._neural = HierarchicalController(predictor)
            logger.info("HierarchicalController lazily initialized")
        return self._neural

    def neural_step(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """POST /neural/step: 一步分层神经控制 (未训练 409, 非法输入 400)。

        字段: window [W,6]|[N,W,6], scene_params 可选, candidate_actions 可选,
              collision_obstacles / collision_radius / speed_limit /
              position_bounds 可选 (首次构造时生效)。
        返回: command / priority_winner / safety_state / 各层诊断 / 反射事件。
        """
        predictor = self._require_predictor()
        window, sp = self._parse_window(body, predictor)
        actions = body.get("candidate_actions")
        if actions is not None and not isinstance(actions, list):
            raise ValueError("'candidate_actions' 需为 dict 列表")
        if actions is not None and not all(isinstance(a, dict) for a in actions):
            raise ValueError("'candidate_actions' 每项需为 dict")
        with self._lock:
            ctrl = self._ensure_neural(predictor)
            out = ctrl.step(window, scene_params=sp, candidate_actions=actions)
        return {
            "status": "ok", "version": __version__,
            "step_index": out["step_index"],
            "command": [round(float(x), 6) for x in out["command"].tolist()],
            "priority_winner": out["priority_winner"],
            "safety_state": out["safety_state"],
            "reflex_triggered": out["reflex_triggered"],
            "reflex_events": out["reflex_events"],
            "cortex": {"ran": out["cortex"].get("ran", False),
                       "planned": out["cortex"].get("planned", False),
                       "elapsed_ms": out["cortex"].get("elapsed_ms")},
            "cerebellum_elapsed_ms": out["cerebellum"]["elapsed_ms"],
            "spinal_elapsed_ms": out["spinal"]["elapsed_ms"],
            "priority_matrix": out["priority_matrix"],
            "analogy_not_reproduction": True,
        }

    def neural_reflex_log(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """POST /neural/reflex/log: 查询反射事件日志 (未训练 409)。

        字段: reset(可选 bool, True => 查询后清空日志)。
        """
        predictor = self._require_predictor()
        reset_flag = bool(body.get("reset", False))
        with self._lock:
            ctrl = self._ensure_neural(predictor)
            log = list(ctrl.reflex_log)
            if reset_flag:
                ctrl.reflex_log = []
        return {
            "status": "ok", "version": __version__,
            "count": len(log), "events": log,
            "scheduler": ctrl.scheduler_summary(),
        }

    # ---- v3.8.2 数字孪生 / 全域闭环 (opt-in 外挂) ------------------- #
    def twin_scene(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """POST /twin/scene: 创建或查询合成数字孪生场景 (非法参数 400)。

        字段(可选, 提供即创建新场景): n_agents(>=1) / n_obstacles(>=0) /
            seed(int) / bounds(>0) / reset(bool)。
        无字段且已有场景 => 查询当前快照。
        """
        create_fields = ("n_agents", "n_obstacles", "seed", "bounds")
        with self._lock:
            if not any(k in body for k in create_fields):
                if self._twin is None:
                    raise ServiceNotReady(
                        "尚未创建数字孪生场景, 请先 POST /twin/scene 提供参数")
                return {
                    "status": "ok", "version": __version__, "action": "query",
                    "scene": self._twin.snapshot(),
                    "summary": self._twin.summary(),
                }
            n_agents = self._coerce_int(body.get("n_agents"), 4, "n_agents")
            n_obstacles = self._coerce_int(body.get("n_obstacles"), 6, "n_obstacles")
            seed = self._coerce_int(body.get("seed"), 0, "seed")
            bounds = self._coerce_float(body.get("bounds"), 10.0, "bounds")
            try:
                self._twin = DigitalTwinScene(
                    n_agents=n_agents, n_obstacles=n_obstacles, seed=seed,
                    bounds=bounds)
            except ValueError as e:
                raise ValueError(f"非法孪生场景参数: {e}")
            # 新场景 => 重置闭环编排器 (如已存在)
            self._twin_loop = None
            logger.info("DigitalTwinScene created: n_agents=%d n_obstacles=%d "
                        "seed=%d bounds=%.1f", n_agents, n_obstacles,
                        seed, bounds)
            return {
                "status": "ok", "version": __version__, "action": "create",
                "scene": self._twin.snapshot(),
                "summary": self._twin.summary(),
                "analogy_not_reproduction": True,
            }

    def twin_step(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """POST /twin/step: 数字孪生一步闭环 (未建场景 409, 非法 400)。

        字段: dt(默认1.0); 可选 window[W,6] 在已挂载 predictor 时附跑一步
            ClosedLoopOrchestrator (大脑->小脑->脊髓->WM反馈->感知更新)。
        主推进: 多体冲突消解 + 积分一步 (DigitalTwinScene.step)。
        """
        if self._twin is None:
            raise ServiceNotReady(
                "尚未创建数字孪生场景, 请先 POST /twin/scene")
        dt = self._coerce_float(body.get("dt"), 1.0, "dt")
        if dt <= 0:
            raise ValueError("dt 必须 > 0")
        with self._lock:
            step_rep = self._twin.step(dt=dt)
            loop_out = None
            window = body.get("window")
            if window is not None:
                predictor = self._require_predictor()
                if self._twin_loop is None:
                    self._twin_loop = ClosedLoopOrchestrator(predictor)
                w, sp = self._parse_window({"window": window}, predictor)
                loop_out = self._twin_loop.step(w, scene_params=sp)
            snap = self._twin.snapshot()
        out = {
            "status": "ok", "version": __version__,
            "step": step_rep["step"],
            "n_conflicts": step_rep["n_conflicts"],
            "yielded_agents": step_rep["yielded"],
            "scene": snap,
            "analogy_not_reproduction": True,
        }
        if loop_out is not None:
            out["closed_loop"] = {
                "command": [round(float(x), 6) for x in
                            loop_out["command"].tolist()],
                "priority_winner": loop_out["priority_winner"],
                "wm_feedback": loop_out["wm_feedback"],
                "safety_state": loop_out["safety_state"],
            }
        return out

    def wla_er(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """POST /wla/er: 统一具身推理头 (WLA 类比) 结构化代理输出。

        字段: window [W,6]|[N,W,6], scene_params 可选。
        未训练 409; 非法输入 400。外挂零梯度, 不改主权重。
        """
        from .embodied import EmbodiedReasoningHead  # 局部导入避免循环
        predictor = self._require_predictor()
        window, sp = self._parse_window(body, predictor)
        with self._lock:
            head = EmbodiedReasoningHead(predictor, enable=True)
            out = head(window, scene_params=sp)
        return {
            "status": "ok", "version": __version__,
            "spatial_relation": out["spatial_relation"].tolist(),
            "target_point": out["target_point"].tolist(),
            "trajectory_proxy": out["trajectory_proxy"].tolist(),
            "latent_dim": out["latent_dim"],
            "main_params_untouched": True,
            "analogy_not_reproduction": True,
        }

    # ---------- v4.1.2 自规划自监督线端点 ---------- #
    def _ensure_curriculum(self, predictor):
        """懒构造课程生成器 + 可解性自验证器 (服务级单例)。"""
        if self._cur_gen is None:
            from .curriculum import CurriculumGenerator, SolvabilityVerifier
            self._cur_gen = CurriculumGenerator(base_seed=4120)
            self._cur_ver = SolvabilityVerifier(predictor)
        return self._cur_gen, self._cur_ver

    def curriculum_generate(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """POST /curriculum/generate: 生成一节课程 (stage)。未训练 409; 非法 400。"""
        predictor = self._require_predictor()
        stage = self._coerce_int(body.get("stage"), 0, "stage")
        if stage < 0:
            raise ValueError("stage 需 >=0")
        with self._lock:
            gen, _ = self._ensure_curriculum(predictor)
            lesson = gen.make_lesson(stage)
        return {"status": "ok", "version": __version__,
                "stage": stage, "n_samples": lesson["n_samples"],
                "kinds": lesson["kinds"], "spread": lesson["spread"]}

    def curriculum_solvable(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """POST /curriculum/solvable: 单任务可解性自验证。未训练 409; 非法 400。"""
        predictor = self._require_predictor()
        window, sp = self._parse_window(body, predictor)
        horizon = self._coerce_int(body.get("horizon"), 1, "horizon")
        if horizon < 1:
            raise ValueError("horizon 需 >=1")
        with self._lock:
            _, ver = self._ensure_curriculum(predictor)
            out = ver.verify(window, scene_params=sp, horizon=horizon)
        return {"status": "ok", "version": __version__, **out}

    def _ensure_selfsup(self, predictor):
        """懒拟合 PWM + 伪标签器 (服务级单例; 只读 predictor)。"""
        if self._selfsup is None:
            from .selfsup import PWMConsistencyPseudoLabeler
            fit_ds = build_parametric_dataset(n_per_kind=8, n_steps=14,
                                              window=6, horizon=3, dt=0.5,
                                              seed=4121)
            self._selfsup = PWMConsistencyPseudoLabeler(
                predictor, fit_dataset=fit_ds, fit_epochs=12)
        return self._selfsup

    def selfsup_pseudo_label(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """POST /selfsup/pseudo-label: PWM 一致性伪标签。未训练 409; 非法 400。"""
        predictor = self._require_predictor()
        window, sp = self._parse_window(body, predictor)
        horizon = self._coerce_int(body.get("horizon"), 3, "horizon")
        if not (1 <= horizon <= 8):
            raise ValueError("horizon 需在 [1,8]")
        with self._lock:
            pl = self._ensure_selfsup(predictor)
            out = pl.pseudo_label(window, horizon=horizon, scene_params=sp)
        return {"status": "ok", "version": __version__,
                "step_mse": [round(float(x), 6) for x in out["step_mse"]],
                "consistency": [round(float(x), 6) for x in out["consistency"]],
                "mean_consistency": out["mean_consistency"],
                "wm_params": out["wm_params"],
                "main_params_untouched": True}

    def _ensure_selfplan(self, predictor):
        if self._decomposer is None:
            from .selfplan import GoalDecomposer
            self._decomposer = GoalDecomposer(predictor, horizon=2)
        return self._decomposer

    def selfplan_decompose(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """POST /selfplan/decompose: goal->子目标链。未训练 409; 非法 400。"""
        predictor = self._require_predictor()
        window, sp = self._parse_window(body, predictor)
        goal = body.get("goal")
        if goal is None:
            raise ValueError("缺少字段 'goal' [6]")
        try:
            goal_t = torch.as_tensor(goal, dtype=torch.float32)
        except Exception as e:
            raise ValueError(f"非法 goal: {e}")
        k = self._coerce_int(body.get("n_subgoals"), 4, "n_subgoals")
        if k < 1:
            raise ValueError("n_subgoals 需 >=1")
        with self._lock:
            de = self._ensure_selfplan(predictor)
            out = de.decompose(window, goal_t, n_subgoals=k, scene_params=sp)
        return {"status": "ok", "version": __version__, **out}

    def selfplan_decide(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """POST /selfplan/decide: 停止/自我纠正判据。未训练 409; 非法 400。"""
        predictor = self._require_predictor()
        window, sp = self._parse_window(body, predictor)
        goal = body.get("goal")
        if goal is None:
            raise ValueError("缺少字段 'goal' [6]")
        try:
            goal_t = torch.as_tensor(goal, dtype=torch.float32)
        except Exception as e:
            raise ValueError(f"非法 goal: {e}")
        k = self._coerce_int(body.get("n_subgoals"), 4, "n_subgoals")
        conf = self._coerce_float(body.get("confidence"), 1.0, "confidence")
        with self._lock:
            de = self._ensure_selfplan(predictor)
            from .selfplan import StopCorrectController
            scc = StopCorrectController(de, confidence_fn=lambda w: conf)
            out = scc.decide(window, goal_t, n_subgoals=k, scene_params=sp)
        return {"status": "ok", "version": __version__, **out}

    # ------------------------------------------------------------------ #
    # v4.2.2: 完全自训练线 HTTP 端点 (自生成三元组 / 自博弈探索)
    # ------------------------------------------------------------------ #
    def _ensure_selftrain(self, predictor):
        """懒构造自生成三元组生成器 + 自博弈探索器 (服务级单例)。"""
        if self._selftrain_gen is None:
            from .self_train import (TransitionTripletGenerator,
                                     SelfPlayExplorer)
            from .world_model import LatentWorldModel
            fit_ds = build_parametric_dataset(n_per_kind=8, n_steps=14,
                                              window=6, horizon=3, dt=0.5,
                                              seed=4220)
            wm = LatentWorldModel(predictor, action_dim=0)
            wm.fit(fit_ds, epochs=12, lr=1e-2, seed=4221)
            self._selftrain_gen = TransitionTripletGenerator(
                predictor, wm, action_dim=1)
            self._selfplay_expl = SelfPlayExplorer(self._selftrain_gen)
        return self._selftrain_gen, self._selfplay_expl

    def selftrain_triplets(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """POST /selftrain/triplets: 世界模型自生成 (s,a,s') 三元组 + 质量。"""
        predictor = self._require_predictor()
        n = self._coerce_int(body.get("n"), 16, "n")
        if not (1 <= n <= 128):
            raise ValueError("n 需在 [1,128]")
        seed = self._coerce_int(body.get("seed"), 0, "seed")
        with self._lock:
            gen, _ = self._ensure_selftrain(predictor)
            ds = build_parametric_dataset(n_per_kind=8, n_steps=14,
                                          window=6, horizon=3, dt=0.5,
                                          seed=seed)
            trip = gen.generate(ds, n=n, seed=seed)
            q = gen.triplet_quality(trip)
        return {"status": "ok", "version": __version__,
                "n": len(trip), "action_dim": trip.action_dim,
                "quality": q, "main_params_untouched": True}

    def selftrain_self_play(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """POST /selftrain/self-play: 自博弈探索 + 守恒规则校验。"""
        predictor = self._require_predictor()
        n = self._coerce_int(body.get("n"), 16, "n")
        if not (1 <= n <= 128):
            raise ValueError("n 需在 [1,128]")
        seed = self._coerce_int(body.get("seed"), 0, "seed")
        with self._lock:
            _, expl = self._ensure_selftrain(predictor)
            ds = build_parametric_dataset(n_per_kind=8, n_steps=14,
                                          window=6, horizon=3, dt=0.5,
                                          seed=seed)
            trip, rep = expl.explore(ds, n=n)
        return {"status": "ok", "version": __version__,
                "n": len(trip), "self_play_report": rep,
                "main_params_untouched": True}

    def _ensure_self_evo(self, predictor):
        """懒构造配置自优化评估器 (服务级单例; 固定基准负载)。"""
        if self._self_evo_evaluator is None:
            from .self_evolution import ConfigEvaluator
            fit_ds = build_parametric_dataset(n_per_kind=8, n_steps=14,
                                              window=6, horizon=3, dt=0.5,
                                              seed=4320)
            self._self_evo_evaluator = ConfigEvaluator(
                predictor, fit_ds.X[:16], repeats=2, atol_tol=1e-4)
        return self._self_evo_evaluator

    def self_evo_search(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """POST /self-evolution/search: 配置自优化搜索 (保真硬门)。"""
        predictor = self._require_predictor()
        with self._lock:
            ev = self._ensure_self_evo(predictor)
            from .self_evolution import ConfigSearcher
            res = ConfigSearcher(ev).search()
        return {"status": "ok", "version": __version__,
                "n_candidates": res["n_candidates"],
                "n_acceptable": res["n_acceptable"],
                "default_cost": res["default_cost"],
                "best_cost": res["best_cost"],
                "improved": res["improved"],
                "selected": res["selected"],
                "verdict": res["verdict"],
                "main_params_untouched": True}

    def self_evo_ab(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """POST /self-evolution/ab: 搜索后 vs 默认配置 同合同 A/B。"""
        predictor = self._require_predictor()
        with self._lock:
            ev = self._ensure_self_evo(predictor)
            from .self_evolution import ConfigSearcher, ConfigAB, ConfigSpec
            searched = ConfigSearcher(ev).search()["selected"]
            ab = ConfigAB(ev, searched=ConfigSpec.from_dict(searched)).run()
        return {"status": "ok", "version": __version__,
                "A_wins": ab["A_wins"],
                "cost_saving_pct": ab["cost_saving_pct"],
                "verdict": ab["verdict"],
                "main_params_untouched": True}

    def self_evo_long_horizon(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """POST /self-evolution/long-horizon: 长程任务闭环类比。"""
        predictor = self._require_predictor()
        horizon = self._coerce_int(body.get("horizon"), 4, "horizon")
        if not (1 <= horizon <= 16):
            raise ValueError("horizon 需在 [1,16]")
        n_sub = self._coerce_int(body.get("n_sub"), 2, "n_sub")
        if not (1 <= n_sub <= horizon):
            raise ValueError("n_sub 需在 [1, horizon]")
        seed = self._coerce_int(body.get("seed"), 0, "seed")
        with self._lock:
            from .self_evolution import LongHorizonLoop
            res = LongHorizonLoop(predictor, horizon=horizon,
                                  n_sub=n_sub).run(seed=seed)
        return {"status": "ok", "version": __version__,
                "verified": res["verified"],
                "n_subtasks": res["n_subtasks"],
                "n_recovered": res["n_recovered"],
                "trajectory_finite": res["trajectory_finite"],
                "main_params_untouched": True}

    # ---- v4.4.0 多智能体协作线 ---------------------------------------- #
    def _ensure_collab(self):
        if self._collab_registry is None:
            from .collab_agents import build_default_registry
            self._collab_registry = build_default_registry()
        return self._collab_registry

    def collab_select(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """POST /collab/select: 拓扑选择器决策树 (图1四问 + 拒绝自治分支)。"""
        from .collab_topology import TopologySelector
        allow_mesh = bool(body.get("allow_mesh", False))
        sel = TopologySelector(allow_mesh=allow_mesh).select(
            q_plan=bool(body.get("q_plan", False)),
            q_expert=bool(body.get("q_expert", False)),
            q_tool=bool(body.get("q_tool", False)),
            q_autonomy=bool(body.get("q_autonomy", False)),
            task_hint=str(body.get("task_hint", "")))
        return {"status": "ok", "version": __version__, **sel.to_dict()}

    def collab_run(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """POST /collab/run: 按拓扑执行一次协作 (star/chain; mesh 需 opt-in)。"""
        from .collab_orchestrator import StarOrchestrator
        from .collab_handoff import ChainHandoff
        from .collab_swarm import MeshSwarm
        reg = self._ensure_collab()
        topology = str(body.get("topology", "star")).lower()
        task_id = str(body.get("task_id", "svc"))
        goal = str(body.get("goal", "")).strip()
        if not goal:
            raise ValueError("goal 不能为空")
        if topology == "star":
            subtasks = body.get("subtasks")
            if not isinstance(subtasks, list) or not subtasks:
                raise ValueError("star 需非空 subtasks 列表")
            eng = StarOrchestrator(reg)
            res = eng.run(task_id, goal, subtasks,
                          owner=str(body.get("owner", "orchestrator")),
                          has_stop=bool(body.get("has_stop", True)),
                          context=str(body.get("context", "")))
        elif topology == "chain":
            routes = body.get("routes")
            chain = body.get("chain")
            if not isinstance(routes, dict) or not routes:
                raise ValueError("chain 需 routes {specialty: tool}")
            if not isinstance(chain, list) or not chain:
                raise ValueError("chain 需 specialty 序列")
            eng = ChainHandoff(reg, routes)
            res = eng.run(task_id, goal, chain,
                          dict(body.get("payload", {})),
                          owner=str(body.get("owner", "triage")),
                          has_stop=bool(body.get("has_stop", True)),
                          context=str(body.get("context", "")))
        elif topology == "mesh":
            if not bool(body.get("allow_mesh", False)):
                raise ValueError("mesh/swarm 默认关, 需 allow_mesh=true (opt-in)")
            tags = body.get("need_tags")
            if not isinstance(tags, list) or not tags:
                raise ValueError("mesh 需 need_tags 列表")
            eng = MeshSwarm(reg, enabled=True)
            res = eng.run(task_id, goal, tags, dict(body.get("payload", {})))
        else:
            raise ValueError(f"未知拓扑: {topology} (star/chain/mesh)")
        # 存 trace 供 /collab/trace/{id} 查询
        self._collab_traces[task_id] = eng.traces
        return {"status": "ok", "version": __version__, **res}

    def collab_handoff(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """POST /collab/handoff: Transfer Bundle 五要素完整性校验交接。"""
        from .transfer_bundle import TransferBundle
        bundle = body.get("bundle")
        if not isinstance(bundle, dict):
            raise ValueError("需 body.bundle (五要素 dict)")
        tb = TransferBundle.from_dict(bundle)
        tb.validate()  # 缺关键要素 -> ValueError -> 400
        new_owner = body.get("reassign_to")
        if new_owner:
            tb.reassign_owner(str(new_owner),
                              reason=str(body.get("reason", "handoff")))
        return {"status": "ok", "version": __version__,
                "valid": True, "bundle": tb.to_dict()}

    def collab_trace(self, trace_id: str) -> Dict[str, Any]:
        """GET /collab/trace/{id}: 责任链查询 (事件序列 + 完整性 + 收口人)。"""
        # 路径穿越守卫: 只允许简单 id, 拒绝 .. / 分隔符
        if (not trace_id or "/" in trace_id or ".." in trace_id
                or os.sep in trace_id):
            raise ValueError("非法 trace_id (路径穿越守卫)")
        # trace 存储 key 可能是 task_id; 先试直接, 再试 trace-<id>
        chain = self._collab_traces.get(trace_id)
        if chain is None and f"trace-{trace_id}" in self._collab_traces:
            chain = self._collab_traces[f"trace-{trace_id}"]
        if chain is None:
            raise KeyError(f"未找到 trace: {trace_id}")
        rep = chain.integrity_report(trace_id if trace_id.startswith("trace-")
                                     else f"trace-{trace_id}")
        return {"status": "ok", "version": __version__, **rep}

    # ------------------------------------------------------------------ #
    # v4.5.1 隐式思考线
    # ------------------------------------------------------------------ #
    def _ensure_latent_reasoner(self) -> LatentReasoner:
        if self._latent_reasoner is None:
            self._latent_reasoner = LatentReasoner(self.engine)
        return self._latent_reasoner

    def reason_latent(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """POST /reason/latent: 隐式思考 (effort=none|low|high|max)。

        字段: scene (PCE dict), effort (none/low/high/max), horizon(1..16, 默认1),
              query?。非法 effort -> 400; 未挂载预测器 -> 409。
        返回 answer/effort/隐式 tick/路径数/是否显式化/延迟/可选显式链与隐式摘要。
        """
        self._require_predictor()
        scene = self._scene(body)
        effort = normalize_effort(body.get("effort"))   # 非法 -> ValueError(400)
        horizon = self._coerce_int(body.get("horizon"), 1, "horizon")
        if not (1 <= horizon <= 16):
            raise ValueError("horizon 需在 [1,16]")
        query = str(body.get("query", ""))
        with self._lock:
            reasoner = self._ensure_latent_reasoner()
            out = reasoner.reason(scene, effort=effort, horizon=horizon,
                                  query=query)
        return {"status": "ok", "version": __version__, **out,
                "main_params_untouched": True,
                "analogy_not_reproduction": True}

    def reason_route(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """POST /reason/route: 难度评估 + 推荐 effort 档位 + 理由。

        字段: scene, task_complexity?(0..1), needs_deeper?(0..1)。
        未挂载预测器 -> 409。
        """
        self._require_predictor()
        scene = self._scene(body)
        task_cx = self._coerce_float(body.get("task_complexity"), 0.0,
                                     "task_complexity")
        needs_deeper = self._coerce_float(body.get("needs_deeper"), 0.0,
                                          "needs_deeper")
        with self._lock:
            r = self.engine.reason(scene)
            sig = DifficultySignals(ctm_convergence=float(r.convergence()),
                                    task_complexity=task_cx,
                                    needs_deeper=needs_deeper)
            decision = self._reasoning_router.route(sig)
        return {"status": "ok", "version": __version__,
                **decision.to_dict()}

    def demo(self) -> Dict[str, Any]:
        from demos.scene_factory import build_factory_scene
        scene = build_factory_scene(n_steps=12)
        body = {"scene": json.loads(PCEParser.dumps(scene))}
        out_i = self.internalize(body)
        out_r = self.reason({**body, "query": "demo"})
        out_x = self.reset({"scene_id": scene.scene_id})
        return {"internalize": out_i, "reason": out_r, "reset": out_x}


def make_handler(service: UDOSService):
    class _Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            # 访问日志降级为 DEBUG: 生产 INFO 级不刷屏, 需要时开 UDOS_LOG_LEVEL=DEBUG
            logger.debug("%s - %s", self.address_string(), fmt % args)

        def _send(self, code: int, payload: Dict[str, Any]):
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _read_body(self) -> Dict[str, Any]:
            n = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(n) if n else b"{}"
            try:
                body = json.loads(raw.decode("utf-8") or "{}")
            except json.JSONDecodeError as e:
                raise ValueError(f"非法 JSON: {e}")
            if not isinstance(body, dict):
                raise ValueError("请求体必须是 JSON 对象")
            return body

        def _authz_principal(self):
            # UDOS_AUTH=off 时 authz 未初始化 -> 透明返回 None (旧行为)
            if getattr(service, "_authz", None) is None:
                return None
            return service._authz.principal_from_headers(self.headers)

        def _authz_authorize(self, method, authz, path, principal):
            if authz is None:
                return
            authz.authorize(method, path, principal)

        def do_GET(self):
            try:
                principal = self._authz_principal()
                self._authz_authorize("GET", service._authz, self.path, principal)
                if self.path.rstrip("/") in ("/health", ""):
                    return self._send(200, service.health())
                if self.path.rstrip("/") == "/demo":
                    return self._send(200, service.demo())
                if self.path.rstrip("/") == "/checkpoints":
                    return self._send(200, service.list_checkpoints())
                if self.path.rstrip("/") == "/experiments":
                    return self._send(200, service.experiments())
                if self.path.rstrip("/") == "/eval/5d":
                    return self._send(200, service.eval_5d())
                if self.path.rstrip("/") == "/intel/health":
                    return self._send(200, service._get_intel().health())
                if self.path.split("?")[0].rstrip("/") == "/intel/coefficient":
                    return self._send(200, service._get_intel().coefficient())
                if self.path.rstrip("/") == "/intel/capability":
                    return self._send(200, service._get_intel().capability())
                if self.path.rstrip("/") == "/intel/routes":
                    return self._send(200, service._get_intel().routes())
                if self.path.split("?")[0].rstrip("/") == "/intel/predictions":
                    status = None
                    if "?" in self.path:
                        from urllib.parse import parse_qs, urlparse
                        q = parse_qs(urlparse(self.path).query)
                        status = q.get("status", [None])[0]
                    return self._send(200, service._get_intel().predictions(status))
                if self.path.rstrip("/") == "/intel/models":
                    return self._send(200, service._get_intel().models())
                if self.path.rstrip("/") == "/intel/asi":
                    return self._send(200, service._get_intel().asi())
                if self.path.rstrip("/") == "/kvcache/state":
                    try:
                        return self._send(200, service.kvcache_state())
                    except RuntimeError as e:
                        return self._send(503, {"status": "error", "message": str(e)})
                if self.path.rstrip("/") == "/kvcache/metrics":
                    try:
                        txt = service.kvcache_metrics_text()
                    except RuntimeError as e:
                        return self._send(503, {"status": "error", "message": str(e)})
                    self.send_response(200)
                    self.send_header("Content-Type",
                                     "text/plain; version=0.0.4; charset=utf-8")
                    self.send_header("Content-Length", str(len(txt.encode())))
                    self.end_headers()
                    self.wfile.write(txt.encode())
                    return
                if self.path.rstrip("/") == "/intel/infrastructure":
                    return self._send(200, service.intel_infrastructure())
                if self.path.rstrip("/") == "/intel/brain":
                    return self._send(200, service.intel_brain())
                if self.path.rstrip("/") == "/intel/finesim":
                    return self._send(200, service.intel_finesim())
                if self.path.rstrip("/") == "/metrics":
                    # v2.5.1: Prometheus 文本格式
                    prom = service.metrics.prometheus_text()
                    self.send_response(200)
                    self.send_header("Content-Type",
                                     "text/plain; version=0.0.4; charset=utf-8")
                    self.send_header("Content-Length", str(len(prom.encode())))
                    self.end_headers()
                    self.wfile.write(prom.encode())
                    return
                # v4.4.1: GET /collab/trace/{id} 责任链查询 (路径参数)
                if self.path.startswith("/collab/trace/"):
                    tid = self.path[len("/collab/trace/"):].split("?", 1)[0]
                    return self._send(200, service.collab_trace(tid))
                # v4.5.4: GET /resources[?profile=&kind=&status=&license=&priority=&level=]
                if self.path.split("?", 1)[0].rstrip("/") == "/resources":
                    query = self.path.split("?", 1)[1] if "?" in self.path else ""
                    return self._send(200, service.resources_list(query))
                self._send(404, {"status": "error", "message": "not found"})
            except KeyError as e:            # trace 不存在 -> 404
                self._send(404, {"status": "error", "message": str(e)})
            except ValueError as e:          # 路径穿越/非法 id -> 400
                logger.warning("GET collab client error (400): %s", e)
                self._send(400, {"status": "error", "message": str(e)})
            except ServiceNotReady as e:   # DIAG-001: 与 do_POST 对齐, 未训练 -> 409
                self._send(409, {"status": "error", "message": str(e)})
            except AuthError as e:
                self._send(401 if e.code in ("unauthenticated", "no_token",
                           "invalid_token", "expired") else 403,
                           {"status": "error", "code": e.code, "message": str(e)})
            except Exception as e:  # 服务不因单个请求崩溃
                logger.exception("GET 处理失败")
                self._send(500, {"status": "error", "message": str(e)})

        def do_POST(self):
            route = self.path.rstrip("/")
            try:
                principal = self._authz_principal()
                self._authz_authorize("POST", service._authz,
                                      self.path, principal)
            except AuthError as e:
                return self._send(
                    401 if e.code in ("unauthenticated", "no_token",
                        "invalid_token", "expired", "locked",
                        "invalid_credentials") else 403,
                    {"status": "error", "code": e.code, "message": str(e)})
            # v5.0.1: 安全/备份/自治新路由
            try:
                if route == "/auth/login":
                    return self._send(200, service.auth_login(self._read_body()))
                if route == "/auth/bootstrap":
                    return self._send(200, service.auth_bootstrap(self._read_body()))
                if route == "/auth/users":
                    return self._send(200, service.auth_list_users(principal))
                if route == "/auth/audit":
                    return self._send(200, service.auth_audit(principal))
                if route == "/auth/logout":
                    hdr = self.headers.get("Authorization", "")
                    token = hdr[7:].strip() if hdr.startswith("Bearer ") else ""
                    service._auth.logout(token, principal["username"])
                    return self._send(200, {"status": "ok"})
                if route == "/backup/run":
                    return self._send(200, service.backup_run(principal))
                if route == "/backup/verify":
                    return self._send(200, service.backup_verify(principal))
                if route == "/autonomy/control":
                    return self._send(200, service.autonomy_control(
                        self._read_body()))
                # v5.0.2: 情报分析服务(2 个 POST)
                if route == "/intel/coefficient/step":
                    return self._send(200, service._get_intel().coefficient_step(self._read_body()))
                if route == "/intel/predictions/score":
                    return self._send(200, service._get_intel().predictions_score(self._read_body()))
                # v5.1.0: KV Cache 分层(opt-in, off -> 503)
                if route == "/kvcache/sim/run":
                    return self._send(200, service.kvcache_sim_run(self._read_body()))
                if route == "/kvcache/cost":
                    return self._send(200, service.kvcache_cost(self._read_body()))
                # v5.3.0: 类脑树突(opt-in)
                if route == "/brain/sim/run":
                    return self._send(200, service.brain_sim_run(self._read_body()))
                if route == "/brain/dhs/benchmark":
                    return self._send(200, service.brain_dhs_bench(self._read_body()))
                if route == "/brain/robustness/run":
                    return self._send(200, service.brain_robust_run(self._read_body()))
                # v5.4.3: 精细生物物理内核(opt-in)
                if route == "/finesim/cable":
                    return self._send(200, service.finesim_cable(self._read_body()))
                if route == "/finesim/hh":
                    return self._send(200, service.finesim_hh(self._read_body()))
                if route == "/finesim/hines_dhs":
                    return self._send(200, service.finesim_hines_dhs(self._read_body()))
                if route == "/finesim/nmda_inhibition":
                    return self._send(200, service.finesim_nmda(self._read_body()))
                if route == "/finesim/payeur":
                    return self._send(200, service.finesim_payeur(self._read_body()))
                if route == "/finesim/robustness":
                    return self._send(200, service.finesim_robustness(self._read_body()))
            except NoIntelData as e:
                return self._send(409, {"status": "error", "message": str(e)})
            except RuntimeError as e:
                return self._send(503, {"status": "error", "message": str(e)})
            except AuthError as e:
                return self._send(
                    401 if e.code in ("unauthenticated", "no_token",
                        "invalid_token", "expired", "locked",
                        "invalid_credentials") else 403,
                    {"status": "error", "code": e.code, "message": str(e)})
            except ValueError as e:
                return self._send(400, {"status": "error", "message": str(e)})
            except Exception as e:  # pragma: no cover
                logger.exception("v5.0.1 route 处理失败")
                return self._send(500, {"status": "error", "message": str(e)})
            if route == "/resources/profile":
                t0 = time.perf_counter()
                try:
                    body = self._read_body()
                    self._send(200, service.resources_set_profile(body))
                except ValueError as e:
                    logger.warning("resources profile client error (400): %s", e)
                    self._send(400, {"status": "error", "message": str(e)})
                except Exception as e:       # noqa: BLE001
                    logger.exception("POST %s 处理失败", route)
                    self._send(500, {"status": "error", "message": str(e)})
                finally:
                    dt = time.perf_counter() - t0
                    service.metrics.record(route, dt)
                return
            # v4.5.4: /resources/{id}/probe 与 /resources/{id}/invoke (前缀路由)
            m = re.match(r"^/resources/([^/]+)/(probe|invoke)$", route)
            if m:
                rid, sub = m.group(1), m.group(2)
                t0 = time.perf_counter()
                try:
                    body = self._read_body()
                    if sub == "probe":
                        out = service.resource_probe(rid)
                    else:
                        out = service.resource_invoke(rid, body)
                    self._send(200, out)
                except KeyError as e:        # 未知资源 id -> 404
                    self._send(404, {"status": "error",
                                     "message": f"未知资源 id: {e}"})
                except ValueError as e:      # 缺 action/非法参数 -> 400
                    logger.warning("resources client error (400): %s", e)
                    self._send(400, {"status": "error", "message": str(e)})
                except ResourceUnavailable as e:  # L3/缺失/无外网 -> 503
                    logger.info("resources %s unavailable (503): %s", rid, e)
                    self._send(503, {"status": "error", "message": str(e),
                                     "resource_status": e.status,
                                     "install_hint": e.install_hint,
                                     "requires": e.requires})
                except Exception as e:       # noqa: BLE001
                    logger.exception("POST %s 处理失败", route)
                    self._send(500, {"status": "error", "message": str(e)})
                finally:
                    dt = time.perf_counter() - t0
                    service.metrics.record(route, dt)
                    logger.info("POST %s handled in %.4fs", route, dt)
                return
            handler = {"/internalize": service.internalize,
                       "/reason": service.reason,
                       "/reset": service.reset,
                       "/train": service.train,
                       "/evaluate": service.evaluate,
                       "/save": service.save,
                       "/calibrate": service.calibrate,
                       "/detect-ood": service.detect_ood,
                       "/predict": service.predict,
                       "/load": service.load,
                       "/rollback": service.rollback,
                       "/export-snapshot": service.export_snapshot_endpoint,
                       "/import-snapshot": service.import_snapshot_endpoint,
                       "/counterfactual": service.counterfactual,
                       "/identify": service.identify,
                       "/risk": service.risk,
                       "/diff-checkpoints": service.diff_checkpoints,
                       "/policy/select": service.policy_select,
                       "/online/adapt": service.online_adapt,
                       "/active/sample": service.active_sample,
                       "/loop/step": service.loop_step,
                       "/multitask/predict": service.multitask_predict,
                       "/retarget/convert": service.retarget_convert,
                       "/affordance/score": service.affordance_score,
                       "/future/predict": service.future_predict,
                       "/action/tokenize": service.action_tokenize,
                       "/action/detokenize": service.action_detokenize,
                       "/augment/generate": service.augment_generate,
                       "/icl/predict": service.icl_predict,
                       "/icm/predict": service.icm_predict,
                       "/icm/demo/register": service.icm_demo_register,
                       "/spatial/query": service.spatial_query,
                       "/spatial/collision": service.spatial_collision,
                       "/wm/imagine": service.wm_imagine,
                       "/wm/conservation": service.wm_conservation,
                       "/neural/step": service.neural_step,
                       "/neural/reflex/log": service.neural_reflex_log,
                       "/twin/step": service.twin_step,
                       "/twin/scene": service.twin_scene,
                       "/wla/er": service.wla_er,
                       "/curriculum/generate": service.curriculum_generate,
                       "/curriculum/solvable": service.curriculum_solvable,
                       "/selfsup/pseudo-label": service.selfsup_pseudo_label,
                       "/selfplan/decompose": service.selfplan_decompose,
                       "/selfplan/decide": service.selfplan_decide,
                       "/selftrain/triplets": service.selftrain_triplets,
                       "/selftrain/self-play": service.selftrain_self_play,
                       "/self-evolution/search": service.self_evo_search,
                       "/self-evolution/ab": service.self_evo_ab,
                       "/self-evolution/long-horizon":
                           service.self_evo_long_horizon,
                       "/collab/select": service.collab_select,
                       "/collab/run": service.collab_run,
                       "/collab/handoff": service.collab_handoff,
                       "/reason/latent": service.reason_latent,
                       "/reason/route": service.reason_route}.get(route)
            if handler is None:
                return self._send(404, {"status": "error",
                                        "message": f"未知路由 {route}"})
            t0 = time.perf_counter()
            try:
                body = self._read_body()
                self._send(200, handler(body))
            except ServiceNotReady as e:   # 依赖未训练/挂载的资源 -> 409
                logger.info("POST %s not ready: %s", route, e)
                self._send(409, {"status": "error", "message": str(e)})
            except AuthError as e:
                self._send(401 if e.code in ("unauthenticated", "no_token",
                           "invalid_token", "expired", "locked",
                           "invalid_credentials") else 403,
                           {"status": "error", "code": e.code, "message": str(e)})
            except (ValueError, TypeError) as e:  # FIX-301: 客户端输入问题 (含 null/非数值类型) -> 400 而非 500;
                logger.warning("POST %s client error (400): %s", route, e)
                self._send(400, {"status": "error", "message": str(e)})
            except Exception as e:
                logger.exception("POST %s 处理失败", route)
                self._send(500, {"status": "error", "message": str(e)})
            finally:
                dt = time.perf_counter() - t0
                service.metrics.record(route, dt)
                logger.info("POST %s handled in %.4fs", route, dt)

    return _Handler


def create_server(host: str, port: int, preset: str,
                  checkpoint: Optional[str] = None) -> ThreadingHTTPServer:
    service = UDOSService(preset=preset, checkpoint=checkpoint)
    httpd = ThreadingHTTPServer((host, port), make_handler(service))
    httpd.service = service  # 暴露给测试
    return httpd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--preset", choices=list(_PRESETS), default="small")
    ap.add_argument("--checkpoint", default=None,
                    help="v2.1 预加载训练好的 PhysicsPredictor checkpoint 路径")
    args = ap.parse_args()
    # 服务启动走统一配置: INFO 级, 全部输出 stderr (不污染 HTTP 响应)
    configure_logging(level=logging.INFO)
    httpd = create_server(args.host, args.port, args.preset,
                          checkpoint=args.checkpoint)
    logger.info("UDOS 推理服务启动: http://%s:%s (preset=%s, pretrained=%s)",
                args.host, args.port, args.preset, bool(args.checkpoint))
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("收到中断, 关闭服务")
    finally:
        httpd.server_close()


if __name__ == "__main__":  # pragma: no cover
    main()
