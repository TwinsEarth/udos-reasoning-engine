"""v7.2 Agent 军团 / Scaling Law / 分层聚合契约测试。

要点（诚实口径）：
- 组织编制按公式精确、树对象有界（亿级不实例化亿对象）；
- 吞吐扩展是 *实测* 墙钟（固定任务、变并发）；
- 集成质量曲线前 D 个为独立信息源(verified)，其后为近相关副本(cpu-proto)，
  只用于刻画饱和，不冒充独立 LLM。
"""
import math
import os

import pytest

from udos7.agents.legion import (build_org, fit_quality_saturation,
                                 hierarchical_select, io_bound_curve,
                                 level_headcounts, org_level_sizes,
                                 quality_curve, throughput_curve)

CKPT = os.environ.get("UDOS7_CKPT", "checkpoints7/worldmodel_v7.0.3.pt")


@pytest.fixture(scope="module")
def model():
    from udos7.persistence import load_worldmodel
    m, _ = load_worldmodel(CKPT)
    return m


# --------------------------------------------------------------------------
# 组织树：精确编制 + 有界物化
# --------------------------------------------------------------------------
def test_org_sizes_exact_and_bounded():
    sizes = org_level_sizes(100_000_000, span=10)
    assert sizes[0] == 100_000_000 and sizes[-1] == 1
    big = build_org(100_000_000, span=10, max_nodes=2000)
    lc = level_headcounts(big)
    assert lc["agent"] == 100_000_000
    assert big.node_count() <= 2100                 # 预览对象有界
    # 小规模完整展开 agent 叶子
    small = build_org(37, span=8)
    assert level_headcounts(small)["agent"] == 37
    leaves = [n for n in _iter_nodes(small) if n.level == "agent"]
    assert len(leaves) == 37


def _iter_nodes(n):
    yield n
    for c in n.children:
        yield from _iter_nodes(c)


def test_hierarchical_select_picks_global_best():
    teams = {
        "t1": [{"id": "a", "score": 0.2}, {"id": "b", "score": 0.6}],
        "t2": [{"id": "c", "score": 0.9}, {"id": "d", "score": 0.3}],
    }
    res = hierarchical_select(teams)
    assert res["winner"]["id"] == "c"
    assert set(res["team_winners"].values()) == {"b", "c"}


# --------------------------------------------------------------------------
# 吞吐 Scaling Law：实测
# --------------------------------------------------------------------------
def test_throughput_curve_measured(model):
    rec = throughput_curve(model, n_tasks=12, workers_list=(1, 4))
    assert [r["agents"] for r in rec] == [1, 4]
    assert all(r["tasks"] == 12 for r in rec)
    assert all(r["tasks_per_s"] > 0 for r in rec)
    assert rec[0]["speedup_vs_1"] == 1.0
    # 墙钟必须为正且被记录（无测量不结论）
    assert all(r["wall_s"] > 0 for r in rec)


# --------------------------------------------------------------------------
# 集成质量 Scaling Law：校准加权 + 证据分级 + 饱和拟合
# --------------------------------------------------------------------------
def test_quality_curve_weighting_and_saturation(model):
    rec = quality_curve(model, n_samples=24, k_list=(1, 2, 4))
    key = "weighted_ensemble_mse"
    assert rec[0]["agents"] == 1 and rec[0]["evidence"] == "verified"
    assert rec[1]["evidence"] == "verified"
    assert rec[2]["evidence"] == "cpu-proto"        # 超出独立源数→近相关副本
    assert rec[2]["independent"] == 2
    # 校准加权集成不得差于两个专家里的较差者（避免被差专家等权拖垮）
    for r in rec:
        assert r[key] <= max(r["best_single_mse"], r["naive_equal_2_mse"]) + 1e-6
    # 同质化扩招到 k=4 不应显著偏离最强单专家（饱和，而非无限提升）
    assert abs(rec[2][key] - rec[2]["best_single_mse"]) < 0.05
    fit = fit_quality_saturation(rec)
    assert fit is not None and fit["model"] == "mse(k)=a+b/k"
    assert math.isfinite(fit["saturation_a"])


def test_io_bound_curve_near_linear():
    rec = io_bound_curve(n_tasks=32, latency_s=0.02,
                         concurrency_list=(1, 2, 4, 8))
    # I/O 等待型远程 Agent：并发翻倍墙钟近半（speedup 接近线性，容差宽松）
    assert rec[0]["speedup_vs_1"] == 1.0
    assert rec[-1]["speedup_vs_1"] >= 5.0           # 8 并发至少 5×（理想 8×）
    for r in rec:
        assert r["wall_s"] >= r["ideal_wall_s"] * 0.8
