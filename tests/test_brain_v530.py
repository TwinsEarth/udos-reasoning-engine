"""v5.3.0 类脑树突契约测试。"""
import os, threading, json, urllib.request, urllib.error


def test_compartment_attenuation():
    from udos.dendrite.compartment import Compartment, CompartmentTree
    t = CompartmentTree()
    t.add(Compartment("a", "apical", distance=2.0))
    t.add(Compartment("b", "basal", distance=0.5))
    assert t.passive_weight("a") < t.passive_weight("b")   # 远端衰减更强


def test_coincidence_and_inhibition():
    from udos.dendrite.dendritic_compute import (coincidence_detect,
        temporal_inhibition, inhibition_curve)
    assert coincidence_detect([1, 5], [2, 9], window=2) == 1
    # Δt<=0 不抑制; Δt>0 门控下降
    assert temporal_inhibition(0, 0) == 1.0
    assert temporal_inhibition(0, 3) < 1.0
    curve = inhibition_curve(range(0, 4))
    assert curve[0]["gate"] == 1.0
    assert curve[-1]["gate"] < curve[0]["gate"]


def test_multimodal_align():
    from udos.dendrite.multimodal_sync import align_channels
    out = align_channels({"a": [0, 1], "b": [1000, 1001]})
    assert "channels" in out


def test_dhs_numerical_consistency_and_count():
    from udos.dendrite.dhs_scheduler import run_voltages, layer_schedule, serial_order
    dag = {"n1": [], "n2": ["n1"], "n3": ["n1"], "n4": ["n2", "n3"]}
    v_serial = run_voltages(dag, 0.1)
    v_dhs = run_voltages(dag, 0.1)
    assert v_serial == v_dhs                 # 数值一致
    layers = layer_schedule(dag)
    assert len(layers) < len(serial_order(dag))   # 关键路径步数下降


def test_brain_service():
    from udos.dendrite.brain import brain_sim, brain_dhs, brain_robustness
    s = brain_sim(7)
    assert "temporal_inhibition_curve" in s
    d = brain_dhs(None, worker_count=4)
    assert d["worker_count"] == 4 and d["dhs_layers"] < d["serial_steps"]
    r = brain_robustness(7)
    assert "dendritic_frontend_keep" in r


def test_brain_http():
    os.environ["UDOS_BRAIN"] = "on"
    from udos.server import create_server
    httpd = create_server("127.0.0.1", 0, "small")
    t = threading.Thread(target=httpd.serve_forever, daemon=True); t.start()
    b = f"http://{httpd.server_address[0]}:{httpd.server_address[1]}"
    def post(p, d):
        r = urllib.request.Request(b+p, data=json.dumps(d).encode(),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(r, timeout=15) as resp:
                return resp.status
        except urllib.error.HTTPError as e:
            return e.code
    try:
        assert post("/brain/sim/run", {"seed": 1}) == 200
        assert post("/brain/dhs/benchmark", {"worker_count": 2}) == 200
        assert post("/brain/robustness/run", {}) == 200
        assert post("/brain/dhs/benchmark", {"worker_count": 0}) == 400
        with urllib.request.urlopen(b+"/intel/brain", timeout=15) as r:
            assert r.status == 200
    finally:
        httpd.shutdown(); t.join(timeout=5)
        os.environ["UDOS_BRAIN"] = "off"


def test_brain_off_503():
    os.environ["UDOS_BRAIN"] = "off"
    from udos.server import create_server
    httpd = create_server("127.0.0.1", 0, "small")
    t = threading.Thread(target=httpd.serve_forever, daemon=True); t.start()
    b = f"http://{httpd.server_address[0]}:{httpd.server_address[1]}"
    try:
        r = urllib.request.Request(b+"/brain/sim/run",
            data=b'{}', headers={"Content-Type": "application/json"})
        try:
            urllib.request.urlopen(r, timeout=15); assert False
        except urllib.error.HTTPError as e:
            assert e.code == 503
        with urllib.request.urlopen(b+"/intel/brain", timeout=15) as resp:
            assert resp.status == 200
    finally:
        httpd.shutdown(); t.join(timeout=5)
