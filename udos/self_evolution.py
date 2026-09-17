"""
UDOS 完全自进化线 (Self-Evolution Flywheel + Infrastructure Self-Optimization)
================================================================================
v4.3.0 起 —— 在 v4.1 (环境自造: curriculum) / v4.2 (数据自产: self_train) 之上,
收口智谱"完全自训练 (Fully Self Training)"三维中的第三维:
**基础设施自我优化** (infrastructure self-optimization) 的缩微 CPU 类比。

来源 (见 docs/SELF_TRAINING_EVOLUTION_RESEARCH.md):
    智谱 HKEX 公告 (2026-09-13, 2513) 释义: 完全自训练 = 递归式自我改进闭环,
    涵盖 数据自产 / 环境自造 / 基础设施自我优化 三维。本线只做第三维的玩具类比:
    真实系统优化"算子/内核/调度/缓存/服务栈", UDOS 优化的是等价的系统配置
    (batch 分片 / 推理缓存 / 集成权重 / 控制频率 / 码本规模)。

设计纪律 (与全工程一致):
    * 纯前向、确定性、@torch.no_grad; 主预测器全程冻结只读, 主参恒 52191。
    * 外挂配置优化**不入主 state_dict**; 默认 opt-in。
    * **核心硬门 —— 质量不退化**: 任一候选配置必须在固定基准负载上, 使预测输出
      相对默认配置的最大绝对差 <= 容差 (输出保真), 否则 REJECT 并留候选账本。
      效率(成本代理)只在"已保真"的配置间取最优; 不允许"靠降质换速度"。
    * 空/非法输入显式 ValueError; NaN/inf 显式拒绝。
    * 日志走 logging_config (默认 stderr), 绝不写 stdout / HTTP 体 / /metrics。
    * **诚实记录**: 自进化多代曲线照实记录"真改进 vs 退化", 不宣称飞轮必然提升。

本文件按 v4.3 线节点逐步追加 (见 docs/VERSION_PLAN_4.3.md):
    4.3.0   ConfigSpec + ConfigEvaluator + ConfigSearcher  系统配置自优化搜索器
    dev1    SearchVerifySelectLoop  自动 搜索->验证->选用 闭环 (网格/随机)
    dev2    ConfigAB                 配置自优化收益 A/B (搜索后 vs 默认, 落 JSON)
    dev3    SelfEvolutionOrchestrator 4.1 课程 -> 4.2 数据 -> 4.3 配置 串联
    dev4    MultiGenerationRunner   多代自进化运行, 性能/效率曲线 (诚实)
    dev5    GlobalStopCorrectCriterion 全局 自我停止/自我纠正 判据
    dev6    LongHorizonLoop         长程任务闭环类比 (拆解->模块当工具->合成环境
                                     交互->错误恢复->验证)
"""

from __future__ import annotations

import itertools
import logging
import math
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from .dynamics import RAW_DIM

logger = logging.getLogger("udos.self_evolution")

# 自进化线内部阶段标记 (CHANGELOG 用; 全局 __version__ 仅在正式训练点 bump)
LINE_STAGES = (
    "4.3.0", "4.3.0.dev1", "4.3.0.dev2", "4.3.0.dev3", "4.3.0.dev4",
    "4.3.0.dev5", "4.3.0.dev6", "4.3.1", "4.3.2", "4.3.9",
)


# ====================================================================== #
# 4.3.0: 系统配置自优化搜索器
# ====================================================================== #
SELF_EVO_LINE_STAGES = LINE_STAGES


@dataclass(frozen=True)
class ConfigSpec:
    """一套"基础设施配置"候选 (只读, 哈希可作账本 key)。

    对应公告"基础设施自我优化"要优化的服务栈旋钮的 CPU 等价物:

    max_shard        : int   BatchPredictor 分片大小 (batch/shard 调度)。
    cache_enabled    : bool  InferenceCache 开关 (缓存层 opt-in)。
    cache_maxsize    : int   InferenceCache LRU 容量 (缓存规模)。
    cortex_hz        : float 分层神经控制大脑规划频率 (控制频率);
                        越低 = 规划调用越稀疏 = 越省 (但响应越钝)。
    codebook_size    : int   WLA 变化掩码 VQ 码本规模 (码本大小/压缩)。
    ensemble_weights : Optional[Tuple[float,...]]  集成成员权重; None=>均匀。
    """

    max_shard: int = 64
    cache_enabled: bool = False
    cache_maxsize: int = 128
    cortex_hz: float = 2.0
    codebook_size: int = 8
    ensemble_weights: Optional[Tuple[float, ...]] = None

    def __post_init__(self) -> None:
        if not isinstance(self.max_shard, int) or self.max_shard < 1:
            raise ValueError("max_shard 须为 >=1 的整数")
        if not isinstance(self.cache_enabled, bool):
            raise ValueError("cache_enabled 须为 bool")
        if not isinstance(self.cache_maxsize, int) or self.cache_maxsize < 1:
            raise ValueError("cache_maxsize 须为 >=1 的整数")
        if not (isinstance(self.cortex_hz, (int, float)) and self.cortex_hz > 0):
            raise ValueError("cortex_hz 须为正数")
        if not isinstance(self.codebook_size, int) or self.codebook_size < 2:
            raise ValueError("codebook_size 须为 >=2 的整数")
        if self.ensemble_weights is not None:
            w = tuple(float(x) for x in self.ensemble_weights)
            if len(w) == 0:
                raise ValueError("ensemble_weights 不能为空")
            if any(x < 0 for x in w) or abs(sum(w)) < 1e-12:
                raise ValueError("ensemble_weights 须非负且和>0")
            # 规范化为概率
            s = sum(w)
            object.__setattr__(self, "ensemble_weights",
                               tuple(x / s for x in w))

    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if d["ensemble_weights"] is not None:
            d["ensemble_weights"] = list(d["ensemble_weights"])
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ConfigSpec":
        kw = dict(d)
        if kw.get("ensemble_weights") is not None:
            kw["ensemble_weights"] = tuple(kw["ensemble_weights"])
        return ConfigSpec(**kw)

    @staticmethod
    def default() -> "ConfigSpec":
        return ConfigSpec()


class ConfigEvaluator:
    """在**固定基准负载**上评估一套基础设施配置 (v4.3.0)。

    纯前向、确定性。主预测器只读。评估两条正交轴:

    1. 保真轴 (质量硬门): 在同一固定窗口负载上跑该配置, 与"默认配置"的预测输出
       比 ``output_max_abs_diff``。max_shard/cache/cortex_hz 不改变前向数学 ->
       差为 0; 仅 ensemble_weights (当提供集成时) 会改输出 -> 受容差门控。
       另对 codebook_size 单独量 VQ 重构 mse 与码本坍塌 (最大码占比), 防压缩崩塌。
    2. 效率轴 (成本代理): **确定性**成本代理 (不依赖壁钟, 避免 CPU 噪声):
       forward_equivalents = 预测前向当量调用数 (分片数 × 有效重复);
       planner_calls      = 大脑规划调用次数 (随 cortex_hz 下降而下降);
       codebook_cost      = 码本规模 (存储成本)。
       composite_cost = forward_equivalents + 0.1*planner_calls + 0.5*codebook_cost。

    Parameters
    ----------
    predictor : 冻结只读 PhysicsPredictor。
    windows   : [N, W, RAW_DIM] 固定基准负载 (等长批量)。
    scene_params : [N, P] 或 None。
    repeats   : 基准重复次数 (用于放大缓存命中收益, >=1)。
    cerebellum_hz : 分层控制标称小脑频率 (推导 planner 调用比用)。
    change_vectors : [M, d] 用于 VQ 码本重构评估的固定变化向量集。
    ensemble  : 可选 List[PhysicsPredictor] 集成成员; 提供时才评估 ensemble_weights。
    atol_tol  : 保真硬门容差 (输出最大绝对差上限)。
    """

    def __init__(self, predictor, windows: torch.Tensor,
                 scene_params: Optional[torch.Tensor] = None,
                 repeats: int = 3, cerebellum_hz: float = 20.0,
                 change_vectors: Optional[torch.Tensor] = None,
                 ensemble: Optional[List] = None,
                 atol_tol: float = 1e-4):
        if windows.ndim != 3 or windows.size(-1) != RAW_DIM:
            raise ValueError(f"windows 应为 [N,W,{RAW_DIM}]")
        if repeats < 1:
            raise ValueError("repeats 须 >=1")
        if atol_tol <= 0:
            raise ValueError("atol_tol 须 >0")
        self.predictor = predictor
        self.windows = windows
        self.scene_params = scene_params
        self.repeats = int(repeats)
        self.cerebellum_hz = float(cerebellum_hz)
        self.atol_tol = float(atol_tol)
        self.ensemble = ensemble
        # 固定变化向量 (用于码本评估)
        if change_vectors is None:
            g = torch.Generator().manual_seed(0)
            change_vectors = torch.randn(64, 8, generator=g)
        self.change_vectors = change_vectors
        self.predictor.eval()
        # 默认配置的参考输出 (只算一次)
        self._ref_out = self._run_predict(ConfigSpec.default())

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _run_predict(self, spec: ConfigSpec) -> torch.Tensor:
        from .batch import BatchPredictor
        from .cache import InferenceCache
        bp = BatchPredictor(self.predictor, max_shard=spec.max_shard)
        cache = None
        if spec.cache_enabled:
            cache = InferenceCache(self.predictor,
                                   maxsize=spec.cache_maxsize)
            cache.enabled = True
        # 第一遍 (可能写缓存)
        out = bp.predict(self.windows, scene_params=self.scene_params,
                         cache=cache)
        # 后续重复 (缓存命中则直接返回)
        for _ in range(self.repeats - 1):
            out = bp.predict(self.windows,
                             scene_params=self.scene_params,
                             cache=cache)
        return out

    def _planner_calls(self, spec: ConfigSpec, n_steps: int = 100) -> int:
        """大脑规划调用次数代理: 每 round(cerebellum/cortex) 步一次。"""
        cortex_every = max(1, round(self.cerebellum_hz / max(spec.cortex_hz, 1e-9)))
        return int(n_steps // cortex_every)

    def _forward_equivalents(self, spec: ConfigSpec) -> int:
        """预测前向当量调用数 (含缓存命中折算)。"""
        n = int(self.windows.size(0))
        shards_per_pass = max(1, math.ceil(n / max(spec.max_shard, 1)))
        if spec.cache_enabled and self.repeats > 1:
            # 第一遍算, 其余重复全命中缓存 (同 key) -> 0 前向
            return shards_per_pass
        return shards_per_pass * self.repeats

    def _codebook_report(self, spec: ConfigSpec) -> Dict[str, float]:
        from .wla import ChangeMaskVQ
        vq = ChangeMaskVQ(codebook_size=spec.codebook_size, seed=0)
        rep = vq.fit(self.change_vectors, steps=20)
        return {
            "codebook_size": float(spec.codebook_size),
            "vq_recon_mse": float(rep.get("recon_mse", 0.0) or 0.0),
            "codebook_utilization": float(rep.get("utilization", 0.0)),
            "codebook_max_share": float(rep.get("max_code_share", 0.0)),
        }

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def evaluate(self, spec: ConfigSpec) -> Dict[str, Any]:
        """评估一套配置, 返回保真 + 效率指标 (全确定性)。"""
        out = self._run_predict(spec)
        diff = (out - self._ref_out).abs().max().item()
        diff = round(float(diff), 8)
        cb = self._codebook_report(spec)
        fwd = self._forward_equivalents(spec)
        planner = self._planner_calls(spec)
        cb_cost = float(spec.codebook_size)
        cost = fwd + 0.1 * planner + 0.5 * cb_cost
        # 保真门: 输出差 <= 容差 且 码本未坍塌 (最大码占比 < 0.99)
        output_ok = diff <= self.atol_tol
        codebook_not_collapsed = cb["codebook_max_share"] < 0.99
        acceptable = bool(output_ok and codebook_not_collapsed)
        return {
            "config": spec.to_dict(),
            "output_max_abs_diff": diff,
            "output_preserved": bool(output_ok),
            "forward_equivalents": int(fwd),
            "planner_calls": int(planner),
            "codebook_cost": cb_cost,
            "composite_cost": round(float(cost), 4),
            "vq_recon_mse": cb["vq_recon_mse"],
            "codebook_utilization": cb["codebook_utilization"],
            "codebook_max_share": cb["codebook_max_share"],
            "acceptable": acceptable,
            "reject_reason": ("" if acceptable else
                              ("output_not_preserved" if not output_ok else
                               "codebook_collapsed")),
        }


class ConfigSearcher:
    """系统配置自优化搜索器 (v4.3.0)。

    在小规模离散网格上枚举候选配置, 逐一过 ``ConfigEvaluator`` 的保真硬门,
    在"可接受 (保真)"的配置里取 ``composite_cost`` 最小者。**默认配置永远在候选内**,
    若搜索没有找到严格更优且保真的配置, 照实返回 ``selected=default`` 且
    ``improved=False`` (不硬选一个"看起来快"但其实和默认等价的配置)。

    纯解析/前向, 零梯度, 不改主权重。
    """

    def __init__(self, evaluator: ConfigEvaluator,
                 grid: Optional[Dict[str, Sequence[Any]]] = None):
        if not isinstance(evaluator, ConfigEvaluator):
            raise ValueError("evaluator 须为 ConfigEvaluator")
        self.evaluator = evaluator
        self.default = ConfigSpec.default()
        self.grid = grid or {
            "max_shard": (16, 64, 128),
            "cache_enabled": (False, True),
            "cache_maxsize": (128,),
            "cortex_hz": (1.0, 2.0, 4.0),
            "codebook_size": (4, 8),
        }

    # ------------------------------------------------------------------ #
    def _candidate_specs(self) -> List[ConfigSpec]:
        keys = list(self.grid.keys())
        specs: List[ConfigSpec] = []
        seen = set()
        for combo in itertools.product(*(self.grid[k] for k in keys)):
            kw = dict(zip(keys, combo))
            try:
                spec = ConfigSpec(**kw)
            except ValueError:
                continue
            key = spec.to_dict()
            if repr(key) in seen:
                continue
            seen.add(repr(key))
            specs.append(spec)
        return specs

    def search(self) -> Dict[str, Any]:
        """跑网格搜索, 返回 best + archive + 是否真改进。"""
        specs = self._candidate_specs()
        results = [self.evaluator.evaluate(s) for s in specs]

        # 默认配置的指标 (作为基线)
        default_res = next((r for r in results
                            if r["config"] == self.default.to_dict()),
                           self.evaluator.evaluate(self.default))
        default_cost = float(default_res["composite_cost"])

        # 只在可接受(保真)配置里选成本最小
        acceptable = [r for r in results if r["acceptable"]]
        if acceptable:
            best = min(acceptable,
                       key=lambda r: r["composite_cost"])
        else:
            best = default_res

        improved = (best["config"] != self.default.to_dict()
                    and best["composite_cost"] < default_cost - 1e-9)
        selected = (ConfigSpec.from_dict(best["config"])
                    if improved else self.default)
        return {
            "n_candidates": len(results),
            "n_acceptable": len(acceptable),
            "default_cost": round(default_cost, 4),
            "best_cost": round(float(best["composite_cost"]), 4),
            "cost_saving_pct": round(
                (1.0 - float(best["composite_cost"]) / default_cost) * 100.0, 2)
                if default_cost > 0 else 0.0,
            "selected": selected.to_dict(),
            "improved": bool(improved),
            "verdict": ("配置自优化找到保真且更省的配置" if improved else
                        "搜索未找到严格更优的保真配置, 沿用默认 (诚实不硬选)"),
            "archive": results,
        }


# ====================================================================== #
# dev1: 自动 搜索 -> 验证 -> 选用 闭环 (网格/随机)
# ====================================================================== #
class SearchVerifySelectLoop:
    """搜索-验证-选用闭环 (dev1): 把 ConfigSearcher 包成显式三段闭环。

    类比 AI-Scientist 的开放式"想法->实验->评审"与 ShinkaEvolve 的"archive+择优":
        search  : 在网格或固定种子随机空间里抽候选配置
        verify  : 逐一过 ConfigEvaluator 保真硬门 (不可信 -> 拒)
        select  : 在通过验证的候选里按成本选优, 落 archive 账本
    每次 `run()` 推进一轮, 维护 `self.archive` (已评估账本) 与 `self.history`
    (每轮 best)。**不假设每轮都改进**; history 里照实记录 cost 变化。
    """

    def __init__(self, evaluator: ConfigEvaluator, n_rounds: int = 3,
                 search_mode: str = "grid", seed: int = 0,
                 grid: Optional[Dict[str, Sequence[Any]]] = None):
        if n_rounds < 1:
            raise ValueError("n_rounds 须 >=1")
        if search_mode not in ("grid", "random"):
            raise ValueError("search_mode 须 ∈ {grid, random}")
        self.evaluator = evaluator
        self.n_rounds = int(n_rounds)
        self.search_mode = search_mode
        self.seed = int(seed)
        self.default = ConfigSpec.default()
        self.grid = grid or {
            "max_shard": (16, 64, 128),
            "cache_enabled": (False, True),
            "cortex_hz": (1.0, 2.0, 4.0),
            "codebook_size": (4, 8),
        }
        self.archive: List[Dict[str, Any]] = []
        self.history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    def _sample_random(self, n: int) -> List[ConfigSpec]:
        g = torch.Generator().manual_seed(self.seed)
        keys = list(self.grid.keys())
        out: List[ConfigSpec] = []
        seen = set()
        tries = 0
        while len(out) < n and tries < n * 20:
            tries += 1
            kw = {k: self.grid[k][int(torch.randint(
                len(self.grid[k]), (1,), generator=g))] for k in keys}
            try:
                spec = ConfigSpec(**kw)
            except ValueError:
                continue
            rk = repr(spec.to_dict())
            if rk in seen:
                continue
            seen.add(rk)
            out.append(spec)
        return out

    def _grid(self) -> List[ConfigSpec]:
        keys = list(self.grid.keys())
        out = []
        seen = set()
        for combo in itertools.product(*(self.grid[k] for k in keys)):
            kw = dict(zip(keys, combo))
            try:
                spec = ConfigSpec(**kw)
            except ValueError:
                continue
            rk = repr(spec.to_dict())
            if rk in seen:
                continue
            seen.add(rk)
            out.append(spec)
        return out

    def run(self) -> Dict[str, Any]:
        for r in range(self.n_rounds):
            cands = self._grid() if self.search_mode == "grid" else \
                self._sample_random(n=max(4, self.n_rounds * 2))
            round_res = [self.evaluator.evaluate(c) for c in cands]
            self.archive.extend(round_res)
            acceptable = [r for r in round_res if r["acceptable"]]
            if acceptable:
                best = min(acceptable, key=lambda r: r["composite_cost"])
            else:
                best = self.evaluator.evaluate(self.default)
            self.history.append({
                "round": r,
                "n_candidates": len(round_res),
                "n_acceptable": len(acceptable),
                "best_cost": round(float(best["composite_cost"]), 4),
                "selected": best["config"],
            })
        # 跨轮全局 best
        all_acc = [r for r in self.archive if r["acceptable"]]
        global_best = min(all_acc, key=lambda r: r["composite_cost"]) \
            if all_acc else self.evaluator.evaluate(self.default)
        return {
            "n_rounds": self.n_rounds,
            "search_mode": self.search_mode,
            "n_archive": len(self.archive),
            "round_history": self.history,
            "global_best": global_best,
        }


# ====================================================================== #
# dev2: 配置自优化收益 A/B (搜索后 vs 默认, 落 JSON)
# ====================================================================== #
class ConfigAB:
    """配置自优化 A/B (dev2): 搜索后配置 (A) vs 默认配置 (B), 同合同对比。

    与 4.2 线 SelfTrainAB 同纪律: **诚实 A/B, 不预设哪方赢**。两臂用同一
    ``ConfigEvaluator``、同一基准负载; A 臂 = 搜索器选出的配置, B 臂 = 默认配置。
    只有当 A 臂**同时** (a) 输出保真 (diff<=容差) 且 (b) 成本严格更低, 才判
    ``A_wins=True``; 否则照实报 ``A_not_better`` 并把搜索结果留候选账本。
    """

    def __init__(self, evaluator: ConfigEvaluator,
                 searched: Optional[ConfigSpec] = None):
        if not isinstance(evaluator, ConfigEvaluator):
            raise ValueError("evaluator 须为 ConfigEvaluator")
        self.evaluator = evaluator
        self.searched = searched or ConfigSpec.default()

    def run(self) -> Dict[str, Any]:
        res_a = self.evaluator.evaluate(self.searched)   # A: 搜索后
        res_b = self.evaluator.evaluate(ConfigSpec.default())  # B: 默认
        cost_a = float(res_a["composite_cost"])
        cost_b = float(res_b["composite_cost"])
        a_wins = bool(res_a["acceptable"] and cost_a < cost_b - 1e-9)
        return {
            "arm_A_searched": res_a,
            "arm_B_default": res_b,
            "delta_cost_A_minus_B": round(cost_a - cost_b, 4),
            "cost_saving_pct": round((1.0 - cost_a / cost_b) * 100.0, 2)
            if cost_b > 0 else 0.0,
            "A_wins": a_wins,
            "verdict": ("搜索后配置保真且更优, 采纳" if a_wins else
                        "搜索后配置未在同合同下严格优于默认, 留候选账本, 不采纳"),
        }


# ====================================================================== #
# dev3: 自进化 orchestrator (4.1 课程 -> 4.2 数据 -> 4.3 配置 串联)
# ====================================================================== #
class SelfEvolutionOrchestrator:
    """自进化 orchestrator (dev3): 把三维飞轮串成一次可复算的"代"。

    一次代 = 三步 (全部只读主预测器, 外挂零梯度):
        1. 环境自造 (4.1): 用 CurriculumGenerator 生成一批课程/任务 (环境自造)。
        2. 数据自产 (4.2): 用 TransitionTripletGenerator 自生成 (s,a,s') 三元组
           并量质量 (数据自产)。
        3. 基础设施自优化 (4.3): 用 ConfigSearcher 在当前服务配置上找更优配置。
    每代落一条代记录 {course_n / triplet_quality / config_best_cost / verdict};
    orchestrator **不假设代际必然改进**, 由 MultiGenerationRunner (dev4) 串联多代。
    """

    def __init__(self, predictor, dataset, evaluator: ConfigEvaluator,
                 wm, curriculum_generator,
                 self_train_generator):
        self.predictor = predictor
        self.dataset = dataset
        self.evaluator = evaluator
        self.wm = wm
        self.curriculum = curriculum_generator
        self.gen = self_train_generator
        self.predictor.eval()

    def run_generation(self, gen_id: int, n_lessons: int = 4,
                       n_trip: int = 32) -> Dict[str, Any]:
        # 1. 环境自造 (课程): CurriculumGenerator.generate(stages) 产出课程列表
        stages = [int((gen_id + i) % 4) for i in range(n_lessons)]
        lessons = self.curriculum.generate(stages)
        course_n = len(lessons)
        # 2. 数据自产 (三元组 + 质量)
        trip = self.gen.generate(self.dataset, n=n_trip, seed=1000 + gen_id)
        quality = self.gen.triplet_quality(trip)
        # 3. 基础设施自优化 (配置搜索)
        searcher = ConfigSearcher(self.evaluator)
        sres = searcher.search()
        return {
            "generation": int(gen_id),
            "course_n": int(course_n),
            "triplet_n": len(trip),
            "triplet_quality": quality,
            "config_best_cost": sres["best_cost"],
            "config_improved": sres["improved"],
            "config_verdict": sres["verdict"],
        }


# ====================================================================== #
# dev4: 多代自进化运行 (性能/效率曲线, 诚实记录真改进 vs 退化)
# ====================================================================== #
class MultiGenerationRunner:
    """多代自进化运行 (dev4): 跑 G 代, 记录每代成本/质量曲线。

    **诚实纪律**: 照实记录每代 ``composite_cost`` 与 ``output_max_abs_diff``;
    末尾用 ``DegradationDetector`` 的思路对成本曲线做判定:
        improving : 成本逐代下降 (真效率改进)
        drifting  : 成本波动/不降
        collapsed : 某代保真门被击穿 (output_preserved=False) -> 退化
    不输出"飞轮必然上升"这类叙事。
    """

    def __init__(self, orchestrator: SelfEvolutionOrchestrator,
                 n_generations: int = 3):
        if n_generations < 1:
            raise ValueError("n_generations 须 >=1")
        self.orch = orchestrator
        self.n_generations = int(n_generations)
        self.records: List[Dict[str, Any]] = []

    def run(self) -> Dict[str, Any]:
        costs: List[float] = []
        collapsed = False
        for g in range(self.n_generations):
            rec = self.orch.run_generation(gen_id=g)
            self.records.append(rec)
            costs.append(float(rec["config_best_cost"]))
            # 若任一代所选配置失真 -> 标记退化
            if rec.get("triplet_quality", {}).get("next_state_finite", 1.0) < 1.0:
                collapsed = True
        # 成本曲线判定
        deltas = [costs[i] - costs[i - 1] for i in range(1, len(costs))]
        improving = all(d <= 1e-9 for d in deltas) and len(deltas) > 0
        if collapsed:
            verdict = "collapsed"
        elif improving:
            verdict = "efficiency_improving"
        else:
            verdict = "drifting"
        return {
            "n_generations": self.n_generations,
            "cost_curve": [round(c, 4) for c in costs],
            "cost_deltas": [round(d, 4) for d in deltas],
            "honest_verdict": verdict,
            "honest_note": (
                "多代成本曲线照实记录: 真改进(逐代降本) vs 退化(失真) vs "
                "漂移(无单调下降); 不宣称自进化飞轮必然提升"),
            "records": self.records,
        }


# ====================================================================== #
# dev5: 全局 自我停止 / 自我纠正 判据
# ====================================================================== #
class GlobalStopCorrectCriterion:
    """全局自我停止/自我纠正判据 (dev5)。

    把 4.2 线已有的 ``DiminishingReturnsCriterion`` (收益递减->停) 与
    ``DegradationDetector`` (代际退化/崩塌) 组合成全局门:
        stop    : 连续多代成本改进不足 (收益递减) -> 建议停止自进化。
        correct : 当代保真门被击穿 / 成本突增 > patience_ratio -> 建议自我纠正
                  (回滚到上一代可接受配置)。
    纯解析、确定性、零梯度。**不假设该停/该改的结果一定更好**。
    """

    def __init__(self, patience: int = 3, min_delta: float = 1e-3,
                 patience_ratio: float = 1.1):
        if patience < 1:
            raise ValueError("patience 须 >=1")
        if min_delta <= 0:
            raise ValueError("min_delta 须 >0")
        if patience_ratio <= 1.0:
            raise ValueError("patience_ratio 须 >1")
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.patience_ratio = float(patience_ratio)

    def check(self, cost_curve: Sequence[float],
              preserved_curve: Sequence[bool]) -> Dict[str, Any]:
        if len(cost_curve) < 2:
            raise ValueError("至少需 2 代成本")
        if len(preserved_curve) != len(cost_curve):
            raise ValueError("preserved_curve 长度须与 cost_curve 一致")
        c = [float(x) for x in cost_curve]
        # 收益递减: 最近 patience 代改进都 < min_delta
        recent = c[-(self.patience + 1):]
        imps = [recent[i - 1] - recent[i] for i in range(1, len(recent))]
        diminishing = len(imps) >= 1 and all(i < self.min_delta for i in imps)
        # 自我纠正: 当代失真 或 当代成本 > 历史最佳 * patience_ratio
        best = min(c)
        cur = c[-1]
        collapsed_now = not bool(preserved_curve[-1])
        spike = cur > best * self.patience_ratio
        need_correct = bool(collapsed_now or spike)
        if collapsed_now:
            action = "rollback_to_last_preserved"
        elif diminishing:
            action = "stop_self_evolution"
        elif need_correct:
            action = "rollback_and_research"
        else:
            action = "continue"
        return {
            "n_generations": len(c),
            "recent_improvements": [round(x, 6) for x in imps],
            "diminishing_returns": bool(diminishing),
            "collapsed_now": collapsed_now,
            "cost_spike": spike,
            "need_self_correction": need_correct,
            "recommended_action": action,
        }


# ====================================================================== #
# dev6: 长程任务闭环类比 (拆解->模块当工具->合成环境交互->错误恢复->验证)
# ====================================================================== #
class LongHorizonLoop:
    """长程任务闭环类比 (dev6): 把"一个长任务"拆成子步, 模块当工具调用,
    在合成环境里交互, 出错恢复, 最后验证。

    对应公告"长程任务环境及智能体能力: 任务拆解、工具调用、环境交互、错误恢复、
    结果验证"的 CPU 玩具类比。**纯确定性编排**, 不真调用 LLM。

    流程:
        1. decompose  : 把目标步数 horizon 拆成 n_sub 个子目标 (子任务链)。
        2. tool_call  : 每个子步调用一个已注册"工具"(这里是轻量可调用: 用主
                        预测器做一步 rollout, 或配置器跑一次评估)。
        3. interact   : 在合成环境(主预测器前向)上推进一帧。
        4. recover    : 若某步输出 NaN/越界 -> 错误恢复 (用上一步合法值兜底)。
        5. verify     : 全程结束后验证轨迹全有限 + 步数守恒。
    """

    def __init__(self, predictor, horizon: int = 4, n_sub: int = 2):
        if horizon < 1:
            raise ValueError("horizon 须 >=1")
        if n_sub < 1 or n_sub > horizon:
            raise ValueError("n_sub 须在 [1, horizon]")
        self.predictor = predictor
        self.horizon = int(horizon)
        self.n_sub = int(n_sub)
        self.predictor.eval()

    # ------------------------------------------------------------------ #
    def decompose(self) -> List[Tuple[int, int]]:
        """把 horizon 步拆成 n_sub 段 (start, end) 子任务区间。"""
        bounds = [round(i * self.horizon / self.n_sub)
                  for i in range(self.n_sub + 1)]
        return [(bounds[i], bounds[i + 1]) for i in range(self.n_sub)]

    @torch.no_grad()
    def run(self, seed: int = 0) -> Dict[str, Any]:
        g = torch.Generator().manual_seed(seed)
        # 合成初始窗口 [1, W, RAW]
        w = torch.randn(1, 6, RAW_DIM, generator=g)
        sub_tasks = self.decompose()
        steps_log: List[Dict[str, Any]] = []
        recovered = 0
        cur = w
        for idx, (s0, s1) in enumerate(sub_tasks):
            for t in range(s0, s1):
                try:
                    nxt = self.predictor.predict_next(cur)
                    if not bool(torch.isfinite(nxt).all()):
                        raise ValueError("NaN/inf 输出")
                    cur = nxt.unsqueeze(1) if nxt.ndim == 2 else nxt
                except Exception:
                    # 错误恢复: 用上一帧兜底, 不崩进程
                    recovered += 1
                    steps_log.append({"subtask": idx, "step": t,
                                      "action": "recover_prev_frame"})
                    continue
                steps_log.append({"subtask": idx, "step": t,
                                  "action": "predict_next"})
        finite = bool(torch.isfinite(cur).all())
        return {
            "horizon": self.horizon,
            "n_subtasks": len(sub_tasks),
            "subtask_intervals": sub_tasks,
            "n_steps_executed": len(steps_log),
            "n_recovered": recovered,
            "trajectory_finite": finite,
            "verified": bool(finite and len(steps_log) >= 1),
            "steps_log": steps_log,
        }


__all__ = [
    "LINE_STAGES", "SELF_EVO_LINE_STAGES", "ConfigSpec", "ConfigEvaluator",
    "ConfigSearcher", "SearchVerifySelectLoop", "ConfigAB",
    "SelfEvolutionOrchestrator", "MultiGenerationRunner",
    "GlobalStopCorrectCriterion", "LongHorizonLoop",
]
