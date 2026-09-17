"""v3.7.2 加固: 23 代 checkpoint 兼容 + 性能基准 + 端点复核。

覆盖:
    * 23 代 checkpoint (v2.1.0..v3.7.0) 全部可加载且主参恒 52191、输出有限;
    * v3.7.0 件元数据;
    * feature_latency_v3.7.0.json / neural_latency_v3.7.0.json 字段;
    * /neural/step 端点 200 复核。
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
CKPT = str(ROOT / "checkpoints" / "predictor_v3.7.0.pt")


def test_version():
    assert __version__ == "5.5.5"


def test_backcompat_23_checkpoints():
    ckpts = sorted(CKPT_DIR.glob("predictor_v*.pt"))
    assert len(ckpts) >= 23, f"backcompat 应至少 23 件 (v2.1.0 起), 实际 {len(ckpts)}"
    w = torch.randn(1, 6, 6)
    for p in ckpts:
        m, meta = load_predictor(p)
        out = m.predict_next(w)
        assert bool(torch.isfinite(out).all()), p.name
        assert sum(pp.numel() for pp in m.parameters()) == 52191, p.name


def test_latest_v370_checkpoint_meta():
    m, meta = load_predictor(CKPT)
    assert meta["udos_version"] == "3.7.0"
    assert sum(p.numel() for p in m.parameters()) == 52191


def test_feature_latency_json():
    p = ROOT / "benchmarks" / "results" / "feature_latency_v3.7.0.json"
    assert p.exists()
    d = json.load(open(p, encoding="utf-8"))
    assert d["feature"] == "hierarchical_neural_control_latency"
    assert d["main_params"] == 52191
    lat = d["latency_ms"]
    for key in ("neural_full_step_ms", "cortex_plan_ms",
                "cerebellum_track_ms", "spinal_reflex_ms",
                "baseline_predict_next_ms"):
        assert key in lat and lat[key] >= 0


def test_neural_latency_json():
    p = ROOT / "benchmarks" / "results" / "neural_latency_v3.7.0.json"
    assert p.exists()
    d = json.load(open(p, encoding="utf-8"))
    assert d["feature"] == "hierarchical_neural_control_latency"


@pytest.fixture(scope="module")
def server():
    torch.manual_seed(0)
    httpd = create_server("127.0.0.1", 0, preset="small", checkpoint=CKPT)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    host, port = httpd.server_address
    yield f"http://{host}:{port}"
    httpd.shutdown(); httpd.server_close(); t.join(timeout=5)


def test_neural_step_recheck(server):
    w = torch.randn(1, 6, 6).tolist()
    data = json.dumps({"window": w}).encode()
    req = urllib.request.Request(server + "/neural/step", data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    assert resp.status == 200
    assert len(body["command"]) == 6
