"""v3.4.2 集成加固 + 19 checkpoint 兼容 + 性能基准测试。"""
import json
from pathlib import Path

import pytest
import torch

from udos import __version__
from udos.persistence import load_predictor

ROOT = Path(__file__).resolve().parents[1]
CKPT_DIR = ROOT / "checkpoints"


def test_all_19_checkpoints_load():
    """backcompat: v3.4.2 时 19 件 (v2.1.0..v3.4.0); v3.4.5 终点扩至 20 件。

    断言当前谱系全部 checkpoint 可加载 (>=19 下界; v3.4.5 后为 20)。
    """
    ckpts = sorted(CKPT_DIR.glob("predictor_v*.pt"))
    assert len(ckpts) >= 19, f"backcompat 应至少 19 件, 实际 {len(ckpts)}"
    w = torch.randn(1, 6, 6)
    for p in ckpts:
        m, meta = load_predictor(p)
        out = m.predict_next(w)
        assert out.shape[-1] == 6
        assert bool(torch.isfinite(out).all()), f"{p.name} 输出非有限"


def test_latency_json_written():
    p = ROOT / "benchmarks/results/feature_latency_v3.4.0.json"
    assert p.exists(), "latency JSON 应由 scripts/icm_feature_latency_v342.py 落盘"
    d = json.load(open(p, encoding="utf-8"))
    assert d["feature"] == "icm_component_latency"
    lat = d["latency_ms"]
    for key in ("zeroshot_predict_next_ms", "icm_k3_predict_ms",
                "event_detect_ms", "three_stream_split_ms",
                "pce_prompt_parse_ms"):
        assert key in lat and lat[key] >= 0


def test_version():
    assert __version__ == "5.5.5"
