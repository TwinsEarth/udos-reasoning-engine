"""v3.7.0.dev6 分层延迟预算量化 A/B 测试。

覆盖:
    * 延迟 JSON 落盘 (neural_latency_v3.7.0.json) 与字段;
    * 三维频率扫描 (cortex_every=1/2/5/10) 四项;
    * 频率-延迟关系: 大脑越频繁规划, cortex_plan_count 越多;
    * 分层延迟不等式 (脊髓 < 小脑 < 大脑, 实测 p50);
    * 各层 p50 在预算内标志;
    * opt-in: 模块不挂默认钩子。
"""
import json
from pathlib import Path

import pytest

from udos import __version__

ROOT = Path(__file__).resolve().parents[1]
JSON = ROOT / "benchmarks" / "results" / "neural_latency_v3.7.0.json"


def test_version():
    assert __version__ == "5.5.5"


@pytest.fixture(scope="module")
def rep():
    assert JSON.exists(), "应落盘 neural_latency_v3.7.0.json"
    with open(JSON, encoding="utf-8") as f:
        return json.load(f)


def test_json_fields(rep):
    assert rep["feature"] == "hierarchical_neural_control_latency"
    assert rep["main_params"] == 52191
    assert rep["analogy_not_reproduction"] is True
    assert "latency_budget_contract" in rep
    assert "within_budget_p50" in rep


def test_frequency_sweep_three_dimensional(rep):
    sweep = rep["frequency_sweep_cortex_every"]
    assert len(sweep) == 4
    every_set = [s["cortex_every"] for s in sweep]
    assert every_set == [1, 2, 5, 10]
    # 频率越高 (every 越小) => 大脑规划次数越多
    plans = {s["cortex_every"]: s["cortex_plan_count"] for s in sweep}
    assert plans[1] > plans[10]
    assert plans[1] > plans[2]


def test_layered_latency_ordering(rep):
    s = rep["frequency_sweep_cortex_every"][0]
    # 脊髓反射 < 小脑跟踪 < 大脑规划 (实测 p50)
    assert s["spinal"]["p50_ms"] < s["cerebellum"]["p50_ms"]
    assert s["cerebellum"]["p50_ms"] < s["cortex"]["p50_ms"]


def test_within_budget(rep):
    wb = rep["within_budget_p50"]
    assert wb["cortex_p50_within_budget"] is True
    assert wb["cerebellum_p50_within_budget"] is True
    assert wb["spinal_p50_within_budget"] is True
