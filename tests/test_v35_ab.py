"""v3.5.0.dev6 空间 A/B 测试: JSON 落盘 / 分辨率扫描 / opt-in / 被否决候选。"""
import json
from pathlib import Path

from udos import __version__

ROOT = Path(__file__).resolve().parents[1]
AB = ROOT / "benchmarks" / "results" / "spatial_ab_v3.5.0.json"


def test_version():
    assert __version__ == "5.5.5"


def test_ab_json_written():
    assert AB.exists(), "应落盘 spatial_ab_v3.5.0.json"
    d = json.loads(AB.read_text(encoding="utf-8"))
    for k in ("route_A_spatial_range_search_ms", "route_B_affordance_score_ms",
              "semantic_iou_A_vs_B", "occupancy_resolution_sweep",
              "decision", "rejected_candidates"):
        assert k in d, f"缺少字段 {k}"


def test_resolution_sweep_has_8_16_32():
    d = json.loads(AB.read_text(encoding="utf-8"))
    sw = d["occupancy_resolution_sweep"]
    assert set(sw) == {"8", "16", "32"}
    for k, v in sw.items():
        assert v["occupied_voxels"] > 0
        assert v["total_voxels"] == int(k) ** 3


def test_opt_in_default_off():
    d = json.loads(AB.read_text(encoding="utf-8"))
    assert d["decision"] == "opt-in default off"
    assert "rationale" in d and d["rationale"]


def test_rejected_candidates_kept():
    d = json.loads(AB.read_text(encoding="utf-8"))
    assert len(d["rejected_candidates"]) >= 1
    for c in d["rejected_candidates"]:
        assert "candidate" in c and "reason" in c
