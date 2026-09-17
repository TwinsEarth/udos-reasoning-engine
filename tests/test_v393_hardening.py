"""v3.9.3 加固: 全部 checkpoint 向后兼容 + 性能基准字段 + 边界守卫。

覆盖:
    * checkpoints/predictor_v*.pt 全部可加载、主参恒 52191、输出有限;
    * feature_latency_v3.9.2.json 字段齐全且延迟 >=0;
    * WLA 外挂边界守卫 (空/非法显式 ValueError, 不崩进程);
    * 版本断言。
"""
import json
from pathlib import Path

import pytest
import torch

from udos import __version__
from udos.persistence import load_predictor
from udos.embodied import ActionTriGroup, EmbodiedReasoningHead
from udos.wla import ChangeMaskVQ, RVQActionTokenizer

ROOT = Path(__file__).resolve().parents[1]
CKPT_DIR = ROOT / "checkpoints"


def test_version():
    assert __version__ == "5.5.5"


def test_all_checkpoints_backcompat():
    ckpts = sorted(CKPT_DIR.glob("predictor_v*.pt"))
    assert len(ckpts) >= 25, f"应至少 25 件, 实际 {len(ckpts)}"
    w = torch.randn(1, 6, 6)
    for p in ckpts:
        m, meta = load_predictor(p)
        out = m.predict_next(w)
        assert bool(torch.isfinite(out).all()), p.name
        assert sum(pp.numel() for pp in m.parameters()) == 52191, p.name


def test_perf_baseline_json():
    p = ROOT / "benchmarks" / "results" / "feature_latency_v3.9.2.json"
    assert p.exists()
    d = json.load(open(p, encoding="utf-8"))
    assert d["main_params"] == 52191
    for k in ("baseline_predict_next_ms", "er_head_ms",
              "change_mask_sparsity_ms", "rvq_encode_ms", "flow_decode_ms"):
        assert k in d["latency_ms"] and d["latency_ms"][k] >= 0


def test_external_modules_edge_guards():
    # 空/非法显式 ValueError, 不崩进程
    with pytest.raises(ValueError):
        ActionTriGroup(6, -1, 6)
    vq = ChangeMaskVQ(4)
    with pytest.raises(ValueError):
        vq.encode(torch.randn(3, 6))       # 未 fit
    tok = RVQActionTokenizer(4, 1)
    with pytest.raises(ValueError):
        tok.encode(torch.randn(3, 4))     # 未 fit
