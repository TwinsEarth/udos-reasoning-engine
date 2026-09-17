"""v5.0.2 情报分析服务锁定契约测试。"""
import pytest

from udos.intelligence import (
    ewma_next, singularity_gate, brier_single, agi_to_asi_years,
    s_curve_eta, model_aggregate, IntelAPI,
)
from udos.intelligence.prediction_scoring import score_predictions


def test_gate_single_source_not_trigger():
    ok, _ = singularity_gate(9.5, "core_tech", independent_sources=1,
                             human_confirmed=True)
    assert ok is False

def test_gate_human_unconfirmed_not_trigger():
    ok, _ = singularity_gate(9.5, "core_tech", 3, False)
    assert ok is False

def test_gate_all_pass():
    ok, _ = singularity_gate(9.5, "core_tech", 3, True)
    assert ok is True

def test_ewma_smooth_bounded():
    a = ewma_next(90.0, 50.0)
    assert abs(a - 62.0) < 1e-6

def test_eta_ordering_naming():
    out = s_curve_eta([(2020, 10), (2023, 40), (2026, 85)])
    assert out["eta_earliest"] <= out["eta_median"] <= out["eta_latest"]

def test_brier_bounds():
    assert brier_single(1.0, 1) == 0.0
    assert brier_single(0.0, 1) == 1.0

def test_asi_interval_monotone():
    assert agi_to_asi_years(90) < agi_to_asi_years(10)

def test_model_leader_bias_discount():
    out = model_aggregate([
        {"model_id": "a", "perspective": "leader", "median_eta_year": 2028.0},
        {"model_id": "b", "perspective": "researcher", "median_eta_year": 2031.0},
    ])
    assert out[0]["bias_adjustment"] == 1.5     # leader 后移
    assert out[0]["median_eta"] == 2029.5

def test_score_predictions():
    out = score_predictions([
        {"probability": 0.7, "outcome": 1},
        {"probability": 0.2, "outcome": 0},
        {"probability": 0.5, "outcome": 1},
        {"probability": 0.6},
    ])
    assert out[0]["score"] == 0.09 and out[0]["status"] == "verified"
    assert out[1]["score"] == 0.04 and out[1]["status"] == "falsified"
    assert out[2]["score"] == 0.25 and out[2]["status"] == "verified"
    assert out[3]["status"] == "pending" and out[3]["score"] is None

def test_score_predictions_bad():
    import pytest
    with pytest.raises(ValueError):
        score_predictions([{"probability": 1.5, "outcome": 1}])
    with pytest.raises(ValueError):
        score_predictions([])


def test_api_health_and_endpoints():
    a = IntelAPI()
    assert a.health()["engine_version"]
    assert "series" in a.coefficient()
    assert "dimensions" in a.capability()
    assert "routes" in a.routes()
    assert "models" in a.models()
    assert "asi_eta" in a.asi()

def test_api_step_gate():
    a = IntelAPI()
    out = a.coefficient_step({
        "events": [{"kind": "paper", "impact": 9.5, "credibility": 0.9,
                    "independent_sources": 2, "category": "core_tech"}],
        "human_confirmed": True})
    assert out["gate"]["triggered"] is True
    assert out["gate"]["human_required"] is True

def test_api_step_bad():
    a = IntelAPI()
    with pytest.raises(ValueError):
        a.coefficient_step({})


def test_http_matrix():
    import os, threading, json, urllib.request, urllib.error
    os.environ["UDOS_AUTH"] = "off"
    from udos.server import create_server
    httpd = create_server("127.0.0.1", 0, "small")
    t = threading.Thread(target=httpd.serve_forever, daemon=True); t.start()
    b = f"http://{httpd.server_address[0]}:{httpd.server_address[1]}"
    def get(p):
        try:
            with urllib.request.urlopen(b + p, timeout=15) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, None
    try:
        assert get("/intel/health")[0] == 200
        c, body = get("/intel/coefficient")
        assert c == 200 and "series" in body
        c, body = get("/intel/routes")
        assert c == 200 and body["routes"][0]["eta_earliest"]
        assert get("/intel/capability")[0] == 200
        assert get("/intel/models")[0] == 200
        assert get("/intel/asi")[0] == 200
        assert get("/intel/predictions?status=pending")[0] == 200
        assert get("/intel/unknown")[0] == 404
    finally:
        httpd.shutdown(); t.join(timeout=5)
        os.environ["UDOS_AUTH"] = "off"
