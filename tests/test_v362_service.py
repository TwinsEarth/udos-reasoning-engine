"""v3.6.2 加固: 22 代 checkpoint 兼容 + WM 性能基准 + 端点复核。

覆盖:
    * 22 代 checkpoint (v2.1.0..v3.6.0) 全部可加载且输出有限;
    * feature_latency_v3.6.0.json 落盘与字段;
    * WM 端点复核 (/wm/imagine 200);
    * 主参 52191 不变。
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
CKPT = str(ROOT / "checkpoints" / "predictor_v3.6.0.pt")


def test_version():
    assert __version__ == "5.5.5"


def test_backcompat_22_checkpoints():
    ckpts = sorted(CKPT_DIR.glob("predictor_v*.pt"))
    assert len(ckpts) >= 23, f"backcompat 应至少 23 件 (v2.1.0 起), 实际 {len(ckpts)}"
    w = torch.randn(1, 6, 6)
    for p in ckpts:
        m, meta = load_predictor(p)
        out = m.predict_next(w)
        assert bool(torch.isfinite(out).all()), p.name
        # 主参恒定 (外挂 WM 不入主 state_dict)
        assert sum(pp.numel() for pp in m.parameters()) == 52191, p.name


def test_latest_v360_checkpoint_loads():
    m, meta = load_predictor(CKPT)
    assert meta["udos_version"] == "3.6.0"
    assert sum(p.numel() for p in m.parameters()) == 52191


def test_feature_latency_json():
    p = ROOT / "benchmarks" / "results" / "feature_latency_v3.6.0.json"
    assert p.exists(), "应落盘 feature_latency_v3.6.0.json"
    d = json.load(open(p, encoding="utf-8"))
    assert d["feature"] == "pwm_world_model_latency"
    assert d["main_params"] == 52191
    lat = d["latency_ms"]
    for key in ("wm_encode_latent_ms", "wm_transit_step_ms", "wm_imagine_h4_ms",
                "wm_imagine_uncertain_ms", "wm_contact_predict_ms",
                "wm_conservation_check_ms", "baseline_predictor_rollout_h4_ms"):
        assert key in lat and lat[key] >= 0


@pytest.fixture(scope="module")
def live_server():
    torch.manual_seed(0)
    httpd = create_server("127.0.0.1", 0, preset="small", checkpoint=CKPT)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address
    base = f"http://{host}:{port}"
    yield base
    httpd.shutdown()
    httpd.server_close()
    thread.join(timeout=5)


def _post(base, path, obj):
    data = json.dumps(obj).encode("utf-8")
    req = urllib.request.Request(base + path, data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


def test_wm_imagine_endpoint_recheck(live_server):
    w = torch.randn(2, 6, 6).tolist()
    sp = torch.zeros(2, 4).tolist()
    code, body = _post(live_server, "/wm/imagine",
                       {"window": w, "scene_params": sp, "horizon": 4})
    assert code == 200, body
    assert body["status"] == "ok"


def test_wm_conservation_endpoint_recheck(live_server):
    w = torch.randn(2, 6, 6).tolist()
    sp = torch.zeros(2, 4).tolist()
    code, body = _post(live_server, "/wm/conservation",
                       {"window": w, "scene_params": sp, "horizon": 4})
    assert code == 200, body
    assert "conserved" in body
