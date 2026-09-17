"""v4.5.5 通用全能级契约: 全量 79 条 schema 合法 + 四态自洽 + profile 切换
不改主预测逐位 (锚点) + 批量探测不卡启动。"""
import json
import threading
import urllib.error
import urllib.request

import pytest
import torch

from udos.server import create_server
from udos.connectors import build_default_registry


def test_full_registry_schema_license_status_selfconsistent():
    reg = build_default_registry("full")
    rows = reg.list(autoprobe=True)
    assert len(rows) == 79
    seen = set()
    for r in rows:
        assert r["id"] not in seen
        seen.add(r["id"])
        # schema 合法
        for k in ("kind", "license", "level", "priority", "requires"):
            assert k in r
        # license 非空
        assert r["license"] and isinstance(r["license"], str)
        # 四态自洽
        st = r["capability"]["status"]
        assert st in ("available", "degraded", "absent", "env_blocked")
        assert r["capability"]["level"] == r["level"]
        # requires 与 level 一致: L3 必声明 gpu/weights
        if r["level"] == "L3":
            assert r["requires"]["weights"] or r["requires"]["gpu"]
        if st == "absent":
            assert r["capability"]["detail"]


def test_every_connector_probe_never_raises():
    reg = build_default_registry("full")
    for rid in reg.ids():
        rep = reg.probe(rid)
        assert rep["capability"]["status"] in (
            "available", "degraded", "absent", "env_blocked")


def test_profile_switch_in_memory_bitwise_prediction_unchanged():
    # profile 切换只影响注册表视图, 绝不碰引擎主权重/前向。
    from udos.server import UDOSService
    torch.manual_seed(0)
    svc = UDOSService(preset="small")
    before = [p.detach().clone() for p in svc.engine.parameters()]
    reg = svc._get_registry()
    reg.apply_profile("full")
    reg.apply_profile("performance")
    after = [p.detach().clone() for p in svc.engine.parameters()]
    assert len(before) == len(after) and len(before) > 0
    for a, b in zip(before, after):
        assert torch.equal(a, b)


@pytest.fixture
def live_server():
    torch.manual_seed(0)
    httpd = create_server("127.0.0.1", 0, preset="small")
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    host, port = httpd.server_address
    yield f"http://{host}:{port}"
    httpd.shutdown()
    httpd.server_close()
    t.join(timeout=5)


def _post(base, path, obj):
    req = urllib.request.Request(
        base + path, data=json.dumps(obj).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())


def _get(base, path):
    try:
        with urllib.request.urlopen(base + path, timeout=30) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())


def test_post_resources_profile_switch(live_server):
    c, b = _post(live_server, "/resources/profile", {"profile": "full"})
    assert c == 200 and b["profile"] == "full"
    assert b["summary"]["visible_in_profile"] == 79
    # 切回 performance
    c, b = _post(live_server, "/resources/profile", {"profile": "performance"})
    assert c == 200 and b["profile"] == "performance"


def test_post_resources_profile_invalid_400(live_server):
    c, b = _post(live_server, "/resources/profile", {"profile": "bogus"})
    assert c == 400


def test_full_bulk_probe_fast_and_cached(live_server):
    # full 下批量探测 79 条不卡 (probe 带缓存, L3 不下载)
    import time
    t0 = time.time()
    c, b = _get(live_server, "/resources?profile=full")
    assert c == 200 and len(b["resources"]) == 79
    assert time.time() - t0 < 30      # 不卡启动


# ---------- v4.5.6 返工: profile 可机械推导, 不再手写窄白名单 ----------
def test_priority_parser_reproduces_high_count():
    """稳健 priority 解析: dict 取 priority 字段, str 取'高——'前缀。
    从 catalog 实测机械推导出的高优先级总数。"""
    from scripts.build_catalog_data import parse_priority
    import json
    rows = json.load(open("docs/opensource_catalog.json"))
    hi = sum(1 for r in rows if parse_priority(r["udos_fit"]) == "高")
    # 本字段实测为 36 (dict 高21 + str 高15); 钉死防回归解析漂移。
    assert hi == 36


def test_profile_derivation_mechanical_contract():
    perf = build_default_registry("performance")
    full = build_default_registry("full")
    pr = perf.list(); fr = full.list()
    # full = 79
    assert len(fr) == 79
    # performance ⊆ full
    pid = {r["id"] for r in pr}; fid = {r["id"] for r in fr}
    assert pid <= fid
    # performance 不含任何 L3 (重权重/GPU 模型被剔除)
    assert all(r["level"] != "L3" for r in pr)
    # performance 条目数 == "高优先级 ∩ CPU可行(L1/L2)" 推导值 (实测 25)
    assert len(pr) == 25
    # 且 performance 全部高优先级
    assert all(r["priority"] == "high" for r in pr)


def test_rssm_state_roundtrip():
    from udos.connectors.specialized import WorldModelRSSMConnector
    from udos.connectors.registry_build import build_spec
    from udos.connectors._catalog_data import CATALOG_ROWS
    row = next(r for r in CATALOG_ROWS if r["id"] == "dreamer_dreamerv3")
    c = WorldModelRSSMConnector(build_spec(row, level="L1"))
    out = c.to_udos({"stoch": [1.0, 3.0], "deter": [0.1, -0.2, 0.3]})
    assert out["schema"] == "udos/rssm-state/v1"
    assert out["stoch_categories"] == 2
    assert abs(sum(out["stoch_probs"]) - 1.0) < 1e-9
    assert out["deter_dim"] == 3

