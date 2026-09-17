"""
v4.5.0 latent_reasoner 测试
- none 档逐位等价锚点 (effort=none 与 engine.reason 原路径数值一致)
- 四档 effort 骨架: K/sigma/externalize/显式链
- 确定性: 同一 effort 多次调用结果一致
"""
import torch

from udos.ctm_engine import CTMConfig
from udos.gpm_engine import GPMConfig, TinyBaseModel
from udos.reasoning import UDOSReasoningEngine
from udos.latent_reasoner import (
    LatentReasoner, EFFORT_TABLE, normalize_effort, EFFORT_LEVELS,
)
import pytest


def _engine():
    base = TinyBaseModel(hidden=32, n_layers=2)
    gpm = GPMConfig(feature_dim=32, latent_size=32, n_latents=8, lora_rank=4,
                    layer_indices=(0, 1), num_pre_head_layers=1, heads=2)
    ctm = CTMConfig(iterations=8, d_model=64, d_input=32, heads=2,
                    n_synch_out=16, n_synch_action=16, memory_length=8,
                    nlm_hidden=16, out_dims=32, certainty_threshold=0.0,
                    n_random_pairing_self=2)
    return UDOSReasoningEngine(ctm, gpm, base_model=base)


def _reasoner():
    return LatentReasoner(_engine())


# --------------------------------------------------------------------------- #
# 四档 effort 配置表
# --------------------------------------------------------------------------- #
def test_effort_table_shape():
    assert set(EFFORT_TABLE) == {"none", "low", "high", "max"}
    # K 递增, 显式化仅 high/max
    assert EFFORT_TABLE["none"].k == 1 and EFFORT_TABLE["none"].sigma == 0.0
    assert EFFORT_TABLE["low"].k == 2
    assert EFFORT_TABLE["high"].k == 4
    assert EFFORT_TABLE["max"].k == 8
    assert EFFORT_TABLE["high"].externalize and EFFORT_TABLE["max"].externalize
    assert not EFFORT_TABLE["low"].externalize


def test_normalize_effort():
    assert normalize_effort("none") == "none"
    assert normalize_effort(" MAX ") == "max"
    for bad in ("nope", "", "medium", None, 3):
        with pytest.raises(ValueError):
            normalize_effort(bad)


# --------------------------------------------------------------------------- #
# none 档逐位等价锚点 (红线)
# --------------------------------------------------------------------------- #
def test_none_bitwise_anchor(tiny_scene, torch_seed):
    """effort=none 必须与 engine.reason 原路径逐位等价。

    逐位等价的强保证: none 路径直接委托 engine.reason(),
    因此其内部 ticks 数与单次 reason 完全一致 (不乘 K),
    且收敛分与 engine.reason().convergence() 到全精度一致。
    """
    eng = _engine()
    reasoner = LatentReasoner(eng)

    r = eng.reason(tiny_scene, query="q", horizon=1)
    out = reasoner.reason(tiny_scene, effort="none", horizon=1)

    # 结构字段
    assert out["effort"] == "none"
    assert out["n_paths"] == 1
    assert out["externalized"] is False
    assert out["explicit_chain"] == []
    # ticks_used 必须等于单次 engine.reason 的 ticks_used (不乘 K)
    assert out["internal_ticks"] == r.ticks_used == 8
    # 收敛分逐位 (6 位) 一致
    assert out["latent_summary"]["convergence_selected"] == pytest.approx(
        r.convergence(), abs=1e-6)
    assert out["latent_summary"]["mode"] == "none-bitwise-identical"


def _drop_timing(d: dict) -> dict:
    """比较时剔除墙钟 latency (不可复现), 其余数值必须逐位一致。"""
    d = {k: v for k, v in d.items() if k != "latency_ms"}
    return d


def test_none_prediction_vector_bitwise(tiny_scene, torch_seed):
    """dev2 加固锚点: none 路径 prediction_vector 与直接 engine.reason 逐位一致。"""
    eng = _engine()
    reasoner = LatentReasoner(eng)
    r = eng.reason(tiny_scene, query="q", horizon=1)
    out = reasoner.reason(tiny_scene, effort="none", horizon=1)
    direct = [round(x, 6) for x in r.prediction.tolist()]
    assert out["prediction_vector"] == direct


def test_none_is_deterministic(tiny_scene, torch_seed):
    reasoner = _reasoner()
    a = reasoner.reason(tiny_scene, effort="none")
    b = reasoner.reason(tiny_scene, effort="none")
    assert _drop_timing(a) == _drop_timing(b)


# --------------------------------------------------------------------------- #
# low/high/max 骨架
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("eff", ["low", "high", "max"])
def test_optin_effort_runs(tiny_scene, torch_seed, eff):
    reasoner = _reasoner()
    out = reasoner.reason(tiny_scene, effort=eff, horizon=1)
    spec = EFFORT_TABLE[eff]
    assert out["n_paths"] == spec.k
    assert out["internal_ticks"] == spec.k * 8   # CTM iterations=8 per path
    # latent_summary 字段齐全
    ls = out["latent_summary"]
    assert ls["n_paths"] == spec.k
    assert 0.0 <= ls["convergence_selected"] <= 1.0
    assert 0.0 <= ls["path_disagreement"]
    assert 0 <= ls["selected_path"] < spec.k


def test_high_max_externalize_chain(tiny_scene, torch_seed):
    reasoner = _reasoner()
    for eff in ("high", "max"):
        out = reasoner.reason(tiny_scene, effort=eff)
        assert out["externalized"] is True
        assert len(out["explicit_chain"]) > 0
        # 第一步必为忠实性说明
        assert "忠实" in out["explicit_chain"][0]["text"]


def test_low_no_explicit_chain(tiny_scene, torch_seed):
    reasoner = _reasoner()
    out = reasoner.reason(tiny_scene, effort="low")
    assert out["externalized"] is False
    assert out["explicit_chain"] == []


def test_optin_deterministic(tiny_scene, torch_seed):
    reasoner = _reasoner()
    a = reasoner.reason(tiny_scene, effort="max")
    b = reasoner.reason(tiny_scene, effort="max")
    assert _drop_timing(a) == _drop_timing(b)


def test_latent_summary_keys(tiny_scene, torch_seed):
    reasoner = _reasoner()
    out = reasoner.reason(tiny_scene, effort="high")
    ls = out["latent_summary"]
    for key in ("mode", "n_paths", "per_path_ticks", "convergence_per_path",
                "convergence_mean", "convergence_selected", "selected_path",
                "path_disagreement", "confidence"):
        assert key in ls


# --------------------------------------------------------------------------- #
# dev5: 可追溯字段统一 (探索路径数/选择理由/切换点/收口人)
# --------------------------------------------------------------------------- #
def test_trace_fields_present(tiny_scene, torch_seed):
    reasoner = _reasoner()
    out = reasoner.reason(tiny_scene, effort="max")
    ls = out["latent_summary"]
    # 4.4 Trace 统一字段
    assert ls["explored_paths"] == 8
    assert "selection_reason" in ls and "路径" in ls["selection_reason"]
    assert ls["switch_point"]["externalized"] is True
    assert ls["closer"] == "latent_orchestrator"
    # 隐式不黑盒: 即使 low 也有摘要
    out_low = reasoner.reason(tiny_scene, effort="low")
    assert "path_disagreement" in out_low["latent_summary"]
