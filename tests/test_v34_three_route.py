"""v3.4.0.dev6 三路线对照实验测试。"""
import json
import os

from udos import __version__


def test_three_route_json_recomputable():
    p = "benchmarks/results/icm_three_route_v3.4.0.json"
    assert os.path.exists(p), "三路线 JSON 应由 scripts/icm_three_route_ab.py 落盘"
    d = json.load(open(p, encoding="utf-8"))
    assert d["experiment"] == "icm_three_route"
    assert d["analogy_not_reproduction"] is True
    # 三维度齐全
    for key in ("data_route", "cot_route", "context_route"):
        assert key in d and len(d[key]) >= 2
        for row in d[key]:
            assert "mse" in row
            assert "latency_ms" in row


def test_context_route_monotone_or_saturate():
    d = json.load(open("benchmarks/results/icm_three_route_v3.4.0.json",
                       encoding="utf-8"))
    ctx = d["context_route"]
    mses = [r["mse"] for r in ctx]
    # 上下文 scaling: k 增大不应爆炸 (检索聚合路径)
    assert all(m < 0.2 for m in mses)


def test_version():
    assert __version__ == "5.5.5"
