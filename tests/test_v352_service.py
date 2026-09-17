"""v3.5.2 加固: 21 代 checkpoint 兼容 + 性能基准 + 端点复核。"""
import json
from pathlib import Path

import pytest
import torch

from udos import __version__
from udos.persistence import load_predictor

ROOT = Path(__file__).resolve().parents[1]
CKPT_DIR = ROOT / "checkpoints"


def test_version():
    assert __version__ == "5.5.5"


def test_backcompat_21_checkpoints():
    """backcompat 下限 21 件 (v2.1.0..v3.5.0); v3.6 线起只增不删 (现 ≥22 件)。"""
    ckpts = sorted(CKPT_DIR.glob("predictor_v*.pt"))
    assert len(ckpts) >= 21, f"应至少 21 件 checkpoint, 实际 {len(ckpts)}"
    w = torch.randn(1, 6, 6)
    for p in ckpts:
        m, meta = load_predictor(p)
        out = m.predict_next(w)
        assert bool(torch.isfinite(out).all()), p.name


def test_latest_v350_checkpoint_loads():
    m, meta = load_predictor(CKPT_DIR / "predictor_v3.5.0.pt")
    assert meta["udos_version"] == "3.5.0"


def test_latency_json_written():
    p = ROOT / "benchmarks/results/feature_latency_v3.5.0.json"
    assert p.exists()
    d = json.load(open(p, encoding="utf-8"))
    assert d["feature"] == "sfm_component_latency"
    lat = d["latency_ms"]
    for key in ("spatial_pairwise_distances_ms", "scene_graph_relate_ms",
                "occupancy_query_ms", "distance_field_at_ms",
                "collision_detect_ms", "nearest_neighbor_ms",
                "spatial_range_search_ms", "spatial_raycast_ms",
                "orthographic_project_ms", "transform_apply_ms"):
        assert key in lat and lat[key] >= 0


def test_spatial_ab_json_still_there():
    p = ROOT / "benchmarks/results/spatial_ab_v3.5.0.json"
    assert p.exists()
    d = json.load(open(p, encoding="utf-8"))
    assert d["decision"] == "opt-in default off"


def test_training_v350_json():
    p = ROOT / "benchmarks/results/training_v3.5.0.json"
    assert p.exists()
    d = json.load(open(p, encoding="utf-8"))
    assert d["n_params"] == 52191
    assert d["backcompat_checkpoints"] == 21
