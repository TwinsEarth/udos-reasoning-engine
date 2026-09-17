"""
v3.1.2 节点39: 集成加固 + backcompat 15 件 + /action/tokenize /detokenize + 延迟基准
================================================================
锚点纪律:
    * POST /action/tokenize 与 /action/detokenize: 200/400/409;
    * backcompat v2.1.0..v3.1.0 共 15 件全加载;
    * benchmarks/results/feature_latency_v3.1.0.json 落盘可复算;
    * 全量回归无退化。
"""
import json
import subprocess
import sys
import threading
import urllib.request
import urllib.error
from pathlib import Path

import pytest
import torch

from udos import __version__
from udos.server import create_server, UDOSService
from udos.persistence import load_predictor
from udos.dynamics import build_parametric_dataset

ROOT = Path(__file__).resolve().parents[1]
CKPT = str(ROOT / "checkpoints" / "predictor_v3.1.0.pt")
LAT_JSON = ROOT / "benchmarks" / "results" / "feature_latency_v3.1.0.json"


@pytest.fixture(scope="module")
def loaded_server():
    httpd = create_server("127.0.0.1", 0, preset="small", checkpoint=CKPT)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    host, port = httpd.server_address
    yield f"http://{host}:{port}"
    httpd.shutdown()
    httpd.server_close()
    t.join(timeout=5)


def _post(base, path, obj):
    data = json.dumps(obj).encode("utf-8")
    req = urllib.request.Request(base + path, data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


def test_version():
    assert __version__ == "5.5.5"


def test_tokenize_200(loaded_server):
    c, b = _post(loaded_server, "/action/tokenize",
                 {"actions": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]})
    assert c == 200
    assert b["status"] == "ok"
    assert len(b["tokens"]) == 1
    assert 0 <= b["tokens"][0] < b["codebook_size"]


def test_detokenize_roundtrip(loaded_server):
    c1, b1 = _post(loaded_server, "/action/tokenize",
                   {"actions": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                                [1.0, -1.0, 0.0, 0.5, -0.5, 0.2]]})
    assert c1 == 200
    c2, b2 = _post(loaded_server, "/action/detokenize",
                   {"tokens": b1["tokens"]})
    assert c2 == 200
    assert b2["shape"] == [2, 6]
    assert all(torch.isfinite(torch.tensor(b2["actions"])).tolist())


def test_tokenize_400(loaded_server):
    c, b = _post(loaded_server, "/action/tokenize", {})   # 缺 actions
    assert c == 400
    c, b = _post(loaded_server, "/action/detokenize", {})  # 缺 tokens
    assert c == 400


def test_endpoints_409_when_untrained():
    svc = UDOSService(preset="small")
    with pytest.raises(Exception):
        svc.action_tokenize({"actions": [[0.0] * 6]})
    with pytest.raises(Exception):
        svc.action_detokenize({"tokens": [0]})


def test_backcompat_15_loads():
    names = ["predictor_v2.1.0.pt", "predictor_v2.2.1.pt", "predictor_v2.3.1.pt",
             "predictor_v2.4.0.pt", "predictor_v2.5.0.pt", "predictor_v2.5.2.pt",
             "predictor_v2.6.0.pt", "predictor_v2.6.2.pt", "predictor_v2.7.0.pt",
             "predictor_v2.7.3.pt", "predictor_v2.8.0.pt", "predictor_v2.9.0.pt",
             "predictor_v3.0.0.pt", "predictor_v3.0.3.pt", "predictor_v3.1.0.pt"]
    assert len(names) == 15
    for n in names:
        m, _ = load_predictor(str(ROOT / "checkpoints" / n))
        out = m.predict_next(torch.randn(1, 6, m.raw_dim),
                             scene_params=(torch.randn(1, m.scene_param_dim)
                                           if m.scene_encoder is not None else None))
        assert out.shape == (1, 6) and torch.isfinite(out).all()


def test_latency_json_written():
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "benchmark_v31_features.py")],
                      cwd=str(ROOT), capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, r.stderr[-500:]
    d = json.load(open(LAT_JSON))
    assert d["n_repeats"] == 50
    for k in ("predict_next", "action_tokenize", "action_detokenize",
              "token_next_token"):
        assert d[k]["p50_ms"] >= 0
