"""v5.1.0 KV Cache 分层卸载契约测试。"""
import pytest

from udos.kvcache import (
    run_simulation, decide_action, fuse_overlap, CascadeRouter,
    compress_block, decompress_block, qat_status, gpu_available,
)
from udos.kvcache.policy import EvictPolicy
from udos.kvcache.infinity import InfinityWindow
from udos.kvcache.block_store import Block, BlockStore
from udos.kvcache.paged_cache import PagedCache


def test_lru_not_evict_pinned():
    p = EvictPolicy("lru")
    p.record("a", now=1.0); p.record("b", now=2.0)
    p.pin("a")
    cands = p.evict_candidates(set())
    assert "a" not in cands
    assert "b" in cands

def test_cold_demote_hot_promote():
    p = EvictPolicy("lfu")
    for _ in range(6): p.record("hot")
    assert p.should_promote_hot("hot", 5) is True
    assert p.should_demote_cold("cold", cold_age=10.0, now=100.0) is True

def test_higher_hit_means_less_recompute_monotone():
    # 更大窗口 -> 命中率不下降
    w8 = InfinityWindow(8)
    w2 = InfinityWindow(2)
    trace = list(range(20)) * 3
    for t in trace:
        w8.access(t); w2.access(t)
    assert w8.hit_rate >= w2.hit_rate

def test_compression_lossless_roundtrip():
    data = bytes(range(256)) * 4
    cm = compress_block(data)
    assert decompress_block(cm["compressed"], cm["method"]) == data

def test_qat_no_hardware_never_fake():
    s = qat_status()
    assert s["qat"] == "ENV_BLOCKED"
    assert "2x" not in s["message"]

def test_fuse_only_overlap():
    out = fuse_overlap([1, 2, 3, 4], [1, 2, 9, 10])
    assert out["shared_len"] == 2
    assert out["reuse"] == [1, 2]

def test_cascade_recall_floor():
    r = CascadeRouter(0.95)
    good = r.evaluate(100, 30, 0.97)
    bad = r.evaluate(100, 30, 0.80)
    assert good["passed"] is True and bad["passed"] is False

def test_ledger_pick_recompute_when_readout_too_expensive():
    # 保留昂贵(大块) + 取回慢(带宽极小) -> 重算更省
    out = decide_action(1e9, 1e6, 0.001, 10, 0.001)
    assert out["decision"] == "recompute"
    # 保留便宜 -> keep
    out2 = decide_action(1e3, 1e3, 100.0, 1000, 0.01)
    assert out2["decision"] == "keep"

def test_cpu_backend_works_gpu_unverified():
    assert gpu_available() in (True, False)
    from udos.kvcache.device import cpu_backend
    assert cpu_backend().read_cost_us(1e6) > 0

def test_blockstore_multitier():
    bs = BlockStore({"hbm": 0.0001, "ddr": 100.0})  # hbm 极小
    b = Block("x", 0, 100, 0, 100000)
    assert bs.put(b, "hbm") is False      # 超限
    assert bs.put(b, "ddr") is True

def test_paged_prefix_share():
    pc = PagedCache()
    pc.allocate("a", ["b1", "b2"])
    n = pc.share_prefix("c", "a")
    assert n == 2

def test_simulation_deterministic():
    a = run_simulation({"seed": 7, "n_tokens": 500})
    b = run_simulation({"seed": 7, "n_tokens": 500})
    assert a == b


def test_http_endpoints():
    import os, threading, json, urllib.request, urllib.error
    os.environ["UDOS_KVCACHE"] = "on"
    from udos.server import create_server
    httpd = create_server("127.0.0.1", 0, "small")
    t = threading.Thread(target=httpd.serve_forever, daemon=True); t.start()
    b = f"http://{httpd.server_address[0]}:{httpd.server_address[1]}"
    def post(p, d):
        r = urllib.request.Request(b+p, data=json.dumps(d).encode(),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(r, timeout=15) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as e:
            return e.code, None
    try:
        c, body = post("/kvcache/sim/run", {"n_tokens": 500, "seed": 1})
        assert c == 200 and "hit_rate" in body
        with urllib.request.urlopen(b + "/kvcache/state", timeout=15) as r:
            assert r.status == 200
    finally:
        httpd.shutdown(); t.join(timeout=5)
        os.environ["UDOS_KVCACHE"] = "off"


def test_sim_input_validation():
    import pytest
    for kw in ({"n_tokens": -5}, {"n_tokens": 0}, {"n_tokens": "x"},
               {"window": -1}, {"vocab": 0}):
        with pytest.raises(ValueError):
            run_simulation(kw)
    with pytest.raises(ValueError):
        run_simulation({})
    assert run_simulation({"n_tokens": 100})["n_tokens"] == 100


def test_http_sim_400_examples():
    import os, threading, json, urllib.request, urllib.error
    os.environ["UDOS_KVCACHE"] = "on"
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
        for bad in ({"n_tokens": -5}, {"n_tokens": 0}, {"n_tokens": "x"},
                    {}, {"window": -1}, {"vocab": 0}):
            assert post("/kvcache/sim/run", bad) == 400
    finally:
        httpd.shutdown(); t.join(timeout=5)
        os.environ["UDOS_KVCACHE"] = "off"


def test_extra_coverage_branches():
    from udos.kvcache.block_store import BlockStore, Block
    from udos.kvcache.paged_cache import PagedCache
    from udos.kvcache.device import ssd_backend, remote_backend, hbm_backend
    bs = BlockStore({"hbm": 100.0, "ddr": 100.0})
    b = Block("b1", 0, 10, 0, 1000)
    assert bs.put(b, "hbm")
    assert bs.move("b1", "ddr")
    assert bs.move("nope", "hbm") is False
    assert bs.evict("b1")
    assert bs.evict("b1") is False
    pc = PagedCache()
    pc.allocate("s", ["x"]); pc.access("x"); pc.free("x")
    assert pc.evict_lru(set()) == "x"
    assert ssd_backend().read_cost_us(1e6) > 0
    assert remote_backend().read_cost_us(1e6) > 0
    assert hbm_backend().tier == "hbm"
    from udos.kvcache.compression import layout_reorder
    assert layout_reorder(b"") == b""
    assert len(layout_reorder(b"abcd")) == 4
    p = EvictPolicy("lfu"); p.pin("a"); p.unpin("a")
    assert "a" not in p.pinned


# ---- v5.1.2: fuse/cascade/infinity 可运行 + cost/metrics/infrastructure ----
def test_fuse_savings():
    from udos.kvcache.fuse import fuse_savings
    out = fuse_savings([[1, 2, 3, 4], [1, 2, 5, 6], [1, 2, 7, 8]])
    assert out["recompute_units_saved"] == 4     # 公共前缀[1,2]省2*2
    assert out["recompute_without"] == 12
    assert out["recompute_with_fuse"] == 8

def test_cascade_run_shrink():
    from udos.kvcache.cascade import cascade_run
    out = cascade_run([0.1, 0.2, 0.8, 0.9])
    assert out["lite_routed"] == 2 and out["main_routed"] == 2
    assert abs(out["main_input_shrink"] - 0.5) < 1e-6

def test_infinity_run():
    from udos.kvcache.infinity import infinity_run
    out = infinity_run([1, 2, 1, 3, 1, 2], window=4, hbm_full=10)
    assert 0.0 <= out["hit_rate"] <= 1.0

def test_cost_breakeven_handcalc():
    from udos.kvcache.cost_ledger import cost_retain_vs_recompute
    # recall_cost=10, recompute_cost=4 -> breakeven=0.4; hit=0.7>0.4 -> retain
    # breakeven=0.4; hit=0.2<0.4 -> retain(取回成本2<重算4)
    out = cost_retain_vs_recompute("ddr", 0.2, 10.0, 4.0)
    assert out["breakeven_hit_rate"] == 0.4
    assert out["recommendation"] == "retain"
    # hit=0.7>0.4 -> recompute(取回7>重算4)
    out2 = cost_retain_vs_recompute("ddr", 0.7, 10.0, 4.0)
    assert out2["recommendation"] == "recompute"

def test_cost_bad_inputs():
    from udos.kvcache.cost_ledger import cost_retain_vs_recompute
    import pytest
    with pytest.raises(ValueError): cost_retain_vs_recompute("gpu", 0.5, 1, 1)
    with pytest.raises(ValueError): cost_retain_vs_recompute("ddr", 1.5, 1, 1)
    with pytest.raises(ValueError): cost_retain_vs_recompute("ddr", 0.5, -1, 1)


def test_v512_http():
    import os, threading, json, urllib.request, urllib.error
    os.environ["UDOS_KVCACHE"] = "on"
    from udos.server import create_server
    httpd = create_server("127.0.0.1", 0, "small")
    t = threading.Thread(target=httpd.serve_forever, daemon=True); t.start()
    b = f"http://{httpd.server_address[0]}:{httpd.server_address[1]}"
    def post(p, d):
        r = urllib.request.Request(b+p, data=json.dumps(d).encode(),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(r, timeout=15) as resp:
                return resp.status, resp.read().decode()
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode()
    try:
        c, _ = post("/kvcache/sim/run", {"n_tokens": 200, "seed": 1})
        assert c == 200
        c, body = post("/kvcache/cost",
                       {"tier": "ddr", "hit_rate": 0.7,
                        "recall_cost": 10.0, "recompute_cost": 4.0})
        assert c == 200 and "recommendation" in body
        assert post("/kvcache/cost", {"tier": "bad"})[0] == 400
        # metrics text/plain
        with urllib.request.urlopen(b+"/kvcache/metrics", timeout=15) as r:
            assert r.headers["Content-Type"].startswith("text/plain")
        # infra always 200
        with urllib.request.urlopen(b+"/intel/infrastructure", timeout=15) as r:
            assert r.status == 200
    finally:
        httpd.shutdown(); t.join(timeout=5)
        os.environ["UDOS_KVCACHE"] = "off"
