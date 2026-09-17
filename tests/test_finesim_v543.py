"""v5.4.3 精细生物物理内核契约测试。"""
import os, threading, json, urllib.request, urllib.error


def test_cable_lambda():
    from udos.finesim.cable import cable_params, verify_lambda
    p = cable_params()
    assert p["lambda_mm"] > 0
    v = verify_lambda()
    assert v["max_fit_error"] < 1e-9


def test_hh_spikes():
    from udos.finesim.hh import simulate
    out = simulate(10.0)
    assert out["spikes"] > 0          # 阶跃电流产生动作电位
    assert out["peak_mV"] > 0


def test_hines_dhs_consistent():
    from udos.finesim.hines_dhs import compare
    c = compare()
    assert c["numerical_consistent"] is True
    assert c["dhs_parallel_layers"] < c["hines_serial_steps"]


def test_nmda_falsifiable():
    from udos.finesim.nmda import falsifiable_control
    f = falsifiable_control()
    assert f["with_mg_range"] > 0      # 有镁阻滞->时序敏感
    assert f["no_mg_range"] == 0.0     # 移除->不敏感(可证伪关键)


def test_payeur_four():
    from udos.finesim.payeur import payeur_demo, synaptic_position_robustness
    out = payeur_demo()
    for k in ("spatiotemporal_filter", "info_selection",
              "info_routing", "multiplexing"):
        assert k in out
    r = synaptic_position_robustness()
    assert r["far_dendrite_spread"] < r["near_soma_spread"]


def test_finesim_http():
    os.environ["UDOS_FINESIM"] = "on"
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
        assert post("/finesim/cable", {}) == 200
        assert post("/finesim/hh", {"current": 10}) == 200
        assert post("/finesim/hines_dhs", {}) == 200
        assert post("/finesim/nmda_inhibition", {}) == 200
        assert post("/finesim/payeur", {}) == 200
        assert post("/finesim/robustness", {}) == 200
        with urllib.request.urlopen(b+"/intel/finesim", timeout=15) as r:
            assert r.status == 200
    finally:
        httpd.shutdown(); t.join(timeout=5)
        os.environ["UDOS_FINESIM"] = "off"


def test_finesim_off_503():
    os.environ["UDOS_FINESIM"] = "off"
    from udos.server import create_server
    httpd = create_server("127.0.0.1", 0, "small")
    t = threading.Thread(target=httpd.serve_forever, daemon=True); t.start()
    b = f"http://{httpd.server_address[0]}:{httpd.server_address[1]}"
    try:
        r = urllib.request.Request(b+"/finesim/cable",
            data=b'{}', headers={"Content-Type": "application/json"})
        try:
            urllib.request.urlopen(r, timeout=15); assert False
        except urllib.error.HTTPError as e:
            assert e.code == 503
        with urllib.request.urlopen(b+"/intel/finesim", timeout=15) as resp:
            assert resp.status == 200
    finally:
        httpd.shutdown(); t.join(timeout=5)
