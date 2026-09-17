"""
v3.3.0.dev4 节点55: 效率 Pareto A/B (参数/延迟/精度三维) 测试
====================================================================
纪律:
    * benchmarks/results/efficiency_pareto_v3.3.0.json 落盘且可复算;
    * 四种形态 (full/pruned_v2/student_v2/moe_addon) 三维指标齐全;
    * 推荐配置标注; 默认关 (default_unchanged=True); 被否决候选保留。
analogy, not reproduction。
"""
import json
from pathlib import Path

from udos import __version__

ROOT = Path(__file__).resolve().parents[1]
PARETO = ROOT / "benchmarks" / "results" / "efficiency_pareto_v3.3.0.json"


def load():
    return json.loads(PARETO.read_text(encoding="utf-8"))


def test_pareto_json_present_and_wellformed():
    d = load()
    # JSON 在 dev4 生成, 后续 dev 节点版本号递增; 只断言属 3.3.0 线
    assert d["version"].startswith("3.3.0")
    for k in ("metrics", "rows", "recommended_config",
              "rejected_candidates", "default_unchanged"):
        assert k in d
    assert set(d["metrics"]) == {"n_params", "predict_ms", "eval_mse"}


def test_four_configs_compare():
    d = load()
    names = {r["config"] for r in d["rows"]}
    assert names == {"full", "pruned_v2", "student_v2", "moe_addon"}
    by = {r["config"]: r for r in d["rows"]}
    # 全量模型参数最大
    assert by["full"]["n_params"] > by["student_v2"]["n_params"]
    # 延迟为正
    for r in d["rows"]:
        assert r["predict_ms"] > 0
        assert r["n_params"] > 0
    # 全量 eval_mse 为有限数值
    assert by["full"]["eval_mse"] > 0


def test_recommended_and_default_off():
    d = load()
    # 激进剪枝/蒸馏不重训精度退化 => 推荐 full, 默认不变 (诚实 opt-in)
    assert d["recommended_config"] == "full"
    assert d["default_unchanged"] is True
    # 被否决候选显式保留
    assert len(d["rejected_candidates"]) >= 1


def test_pruned_accuracy_degrades_honestly_recorded():
    """剪枝不重训精度退化应被照实记录 (不是假装无损)。"""
    d = load()
    by = {r["config"]: r for r in d["rows"]}
    # 剪枝后 eval_mse 显著高于 full (退化被记录)
    assert by["pruned_v2"]["eval_mse"] > by["full"]["eval_mse"]
    assert by["pruned_v2"]["channel_sparsity"] > 0
