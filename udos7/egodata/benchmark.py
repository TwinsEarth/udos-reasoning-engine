"""数据飞轮基准：被动上量 vs Coverage-aware 主动采集。"""
from __future__ import annotations

from typing import Dict

from ..contracts import EvidenceGrade
from .collection import (build_pools, run_strategy, passive_select,
                         active_select, flywheel_curve)
from .coverage import CoverageMap
from .processing import process_batch

# 外部公开报道（unverified，仅作概念参照，不与本仓数字互证）
EXTERNAL_REFERENCE = {
    "source": "Maxinsights 媒体报道 / Dyna Robotics Dyna-2 技术报告（转述）",
    "claims_unverified": [
        "Maxinsights 累计 150 万+小时精品 egocentric 数据，历史交付 200 万+小时",
        "数据良率 Yield 98%，目标 1000 万小时精品数据",
        "Dyna-2 从 1000 小时到 100 万小时预测持续改善、未见饱和",
        "MaxVLM 具身视频理解对标公开 SOTA、推理成本约 1/10",
        "Coverage-aware Collection：Agent 反向诊断数据盲区并定向补采",
        "Task Coverage → State Coverage；Experience Scale 与 Density 并重"],
}


def _oracle_cells(pool) -> int:
    accepted, _, _ = process_batch(list(pool))
    m = CoverageMap()
    for e in accepted:
        m.add(e)
    return m.state_count()


def run_flywheel_benchmark(pool_size: int = 160, budget: int = 48,
                           seed: int = 0, K: int = 16) -> Dict:
    passive_pool, active_pool = build_pools(pool_size, budget, seed=seed, K=K)
    oracle = max(_oracle_cells(passive_pool), _oracle_cells(active_pool))

    passive = run_strategy(passive_pool, budget, active=False)
    active = run_strategy(active_pool, budget, active=True)
    # 消融：同一偏态池上只换选择策略，隔离“主动选择”的贡献
    ablation = run_strategy(passive_pool, budget, active=True)

    def frac(cells):
        return round(cells / max(1, oracle), 3)

    return {
        "module": "udos7.egodata egocentric experience flywheel",
        "version": "v7.4.0",
        "evidence_grade": EvidenceGrade.CPU_PROTO.value,
        "config": {"pool_size": pool_size, "budget_episodes": budget,
                   "raw_minutes_per_episode": 3.0,
                   "budget_raw_minutes": budget * 3.0,
                   "defect_rate": 0.25},
        "oracle_state_cells": oracle,
        "passive_collection": {
            "yield": passive["ledger"]["yield_rate"],
            "state_cells": passive["final_state_cells"],
            "oracle_fraction": frac(passive["final_state_cells"]),
            "redundancy_rate": passive["redundancy_rate"],
            "mean_density": passive["mean_density"],
            "reject_reasons": passive["reject_reasons"],
            "coverage_curve": passive["coverage_curve"]["cumulative_cells"],
            "marginal_curve": passive["coverage_curve"]["marginal_new_cells"]},
        "coverage_aware_collection": {
            "yield": active["ledger"]["yield_rate"],
            "state_cells": active["final_state_cells"],
            "oracle_fraction": frac(active["final_state_cells"]),
            "redundancy_rate": active["redundancy_rate"],
            "mean_density": active["mean_density"],
            "reject_reasons": active["reject_reasons"],
            "coverage_curve": active["coverage_curve"]["cumulative_cells"],
            "marginal_curve": active["coverage_curve"]["marginal_new_cells"]},
        "ablation_greedy_on_skewed_pool": {
            "state_cells": ablation["final_state_cells"],
            "oracle_fraction": frac(ablation["final_state_cells"]),
            "yield": ablation["ledger"]["yield_rate"]},
        "external_reference": EXTERNAL_REFERENCE,
        "known_limits": [
            "世界为点质量具身环境，非真实第一视角视频/手部追踪",
            "缺陷类型为合成注入，真实 QC 依赖视觉模型（MaxVLM 类，需 LLM/视觉 key）",
            "状态单元为手工离散，非学习得到的状态表征"],
        "gates": ["真实 egocentric 采集网络与手部追踪/SLAM",
                  "视频理解大模型（MaxVLM 类）与多 LLM 交叉打分 key",
                  "跨本体 Embodiment Gap 的真机验证"],
    }
