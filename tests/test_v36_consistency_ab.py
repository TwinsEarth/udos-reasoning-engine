"""v3.6.0.dev4 世界模型 vs 预测器一致性 A/B 测试。

覆盖:
    * A/B JSON 落盘 (wm_consistency_v3.6.0.json) 与必需字段;
    * 逐步 MSE 序列 (第 0 步=0 锚点, 末步>0 发散);
    * 守恒违反量两段对比;
    * opt-in 裁决 + 被否决候选保留;
    * 版本断言。
"""
import json
from pathlib import Path

from udos import __version__

ROOT = Path(__file__).resolve().parents[1]
AB = ROOT / "benchmarks" / "results" / "wm_consistency_v3.6.0.json"


def test_version():
    assert __version__ == "5.5.5"


def test_json_written():
    assert AB.exists(), "应落盘 wm_consistency_v3.6.0.json"
    d = json.loads(AB.read_text(encoding="utf-8"))
    for k in ("per_step_mse_imagine_vs_real", "real_rollout_conservation",
              "imagine_rollout_conservation", "latency_real_rollout_ms",
              "latency_imagine_ms", "decision", "rejected_candidates",
              "main_params", "wm_params"):
        assert k in d, f"缺少字段 {k}"


def test_stepwise_mse_anchor_and_drift():
    d = json.loads(AB.read_text(encoding="utf-8"))
    mse = d["per_step_mse_imagine_vs_real"]
    assert len(mse) == d["horizon"]
    # 第 0 步逐位锚定 => 严格 0
    assert mse[0] == 0.0
    # 末步发散 (合成类比: 潜在外挂转移随 horizon 累积误差)
    assert mse[-1] > 0.0


def test_main_params_untouched():
    d = json.loads(AB.read_text(encoding="utf-8"))
    assert d["main_params"] == 52191
    assert d["wm_params"] > 0


def test_opt_in_decision():
    d = json.loads(AB.read_text(encoding="utf-8"))
    # 裁决应明确 opt-in, 不替换主 rollout
    assert "opt-in" in d["decision"]


def test_rejected_candidates_kept():
    d = json.loads(AB.read_text(encoding="utf-8"))
    rej = d["rejected_candidates"]
    assert isinstance(rej, list) and len(rej) >= 1
    for c in rej:
        assert "candidate" in c and "reason" in c
