"""v3.8.3 加固: 24 代 checkpoint 兼容 + 性能基准 + 端点复核。

覆盖:
    * 24 代 checkpoint (v2.1.0..v3.8.0) 全部可加载且主参恒 52191、输出有限;
    * feature_latency_v3.8.0.json 字段;
    * /twin/step /twin/scene 端点 200 复核;
    * 分层/多体延迟预算为合成可测 (>=0)。
"""
import json
import threading
import urllib.request
import urllib.error
from pathlib import Path

import pytest
import torch

from udos import __version__
from udos.persistence import load_predictor
from udos.server import create_server

ROOT = Path(__file__).resolve().parents[1]
CKPT_DIR = ROOT / "checkpoints"
CKPT = str(ROOT / "checkpoints" / "predictor_v3.8.0.pt")


def test_version():
    assert __version__ == "5.5.5"


def test_backcompat_24_checkpoints():
    ckpts = sorted(CKPT_DIR.glob("predictor_v*.pt"))
    assert len(ckpts) >= 25, f"应至少 25 件, 实际 {len(ckpts)}"
    w = torch.randn(1, 6, 6)
    for p in ckpts:
        m, meta = load_predictor(p)
        out = m.predict_next(w)
        assert bool(torch.isfinite(out).all()), p.name
        assert sum(pp.numel() for pp in m.parameters()) == 52191, p.name


def test_latest_v380_meta():
    m, meta = load_predictor(CKPT)
    assert meta["udos_version"] == "3.8.0"
    assert sum(p.numel() for p in m.parameters()) == 52191


def test_feature_latency_json():
    p = ROOT / "benchmarks" / "results" / "feature_latency_v3.8.0.json"
    assert p.exists()
    d = json.load(open(p, encoding="utf-8"))
    assert d["main_params"] == 52191
    lat = d["latency_ms"]
    for key in ("baseline_predict_next_ms", "multi_agent_resolve_ms",
                "wm_scheduler_allocate_ms", "closed_loop_step_ms",
                "digital_twin_step_ms"):
        assert key in lat and lat[key] >= 0


@pytest.fixture(scope="module")
def server():
    torch.manual_seed(0)
    httpd = create_server("127.0.0.1", 0, preset="small", checkpoint=CKPT)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    host, port = httpd.server_address
    yield f"http://{host}:{port}"
    httpd.shutdown(); httpd.server_close(); t.join(timeout=5)


def _post(base, path, obj):
    data = json.dumps(obj).encode("utf-8")
    req = urllib.request.Request(base + path, data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


def test_twin_endpoints_recheck(server):
    code, body = _post(server, "/twin/scene",
                       {"n_agents": 3, "n_obstacles": 2, "seed": 5})
    assert code == 200, body
    code, body = _post(server, "/twin/step", {})
    assert code == 200, body
    assert "scene" in body
