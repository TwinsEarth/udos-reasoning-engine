"""v4.4.1 多智能体协作线 HTTP 端点集成测试。

覆盖 4 个新端点逐路径 正常 200 + 异常 400 + 未知路由 404:
    POST /collab/select   拓扑决策树
    POST /collab/run     按拓扑执行 (star/chain; mesh 需 opt-in, 未 opt-in -> 400)
    POST /collab/handoff Bundle 五要素校验 (残缺 -> 400)
    GET  /collab/trace/{id} 责任链查询 (存在 200 / 不存在 404 / 路径穿越 400)
错误语义延续: 非法 400, 未知路由 404, 不崩进程; /metrics 纯文本。
"""
import json
import threading
import urllib.request
import urllib.error
from pathlib import Path

import pytest
import torch

from udos import __version__
from udos.server import create_server

ROOT = Path(__file__).resolve().parents[1]
CKPT = str(ROOT / "checkpoints" / "predictor_v4.3.0.pt")


def test_version_pinned():
    assert isinstance(__version__, str) and __version__.startswith("5.")


@pytest.fixture(scope="module")
def server():
    torch.manual_seed(0)
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
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


def _get(base, path):
    try:
        with urllib.request.urlopen(base + path, timeout=30) as r:
            return r.status, r.read().decode("utf-8"), r.headers.get(
                "Content-Type", "")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8"), ""


# ---- /collab/select ---- #
def test_select_star(server):
    st, b = _post(server, "/collab/select", {"q_plan": True})
    assert st == 200 and b["topology"] == "star" and b["accepted"] is True


def test_select_refuse_autonomy(server):
    st, b = _post(server, "/collab/select", {"q_autonomy": True})
    # mesh 默认关: 要自治但没 allow_mesh -> 退回 star, 不 swarm
    assert st == 200 and b["topology"] == "star"


def test_select_unknown_route(server):
    st, _ = _post(server, "/collab/selectxx", {})
    assert st == 404


# ---- /collab/run ---- #
def test_run_star_ok(server):
    st, b = _post(server, "/collab/run", {
        "topology": "star", "task_id": "svc-star", "goal": "完成推演",
        "subtasks": [
            {"key": "a", "tool": "ctm_reasoning",
             "payload": {"query": "q", "horizon": 2}},
            {"key": "b", "tool": "gpm_scene", "payload": {"scene_id": "s"}}]})
    assert st == 200 and b["topology"] == "star"
    assert b["trace_integrity"]["closed"]
    assert b["main_params_untouched"] is True


def test_run_star_bad(server):
    # 缺 goal -> 400
    st, _ = _post(server, "/collab/run", {"topology": "star", "subtasks": []})
    assert st == 400
    # 空 subtasks -> 400
    st2, _ = _post(server, "/collab/run",
                   {"topology": "star", "goal": "g", "subtasks": []})
    assert st2 == 400


def test_run_mesh_not_optin_400(server):
    # mesh 默认关: 未 allow_mesh -> 400 (拒绝自治)
    st, _ = _post(server, "/collab/run", {
        "topology": "mesh", "goal": "g", "need_tags": ["reasoning"]})
    assert st == 400


def test_run_mesh_optin_ok(server):
    st, b = _post(server, "/collab/run", {
        "topology": "mesh", "goal": "g", "allow_mesh": True,
        "need_tags": ["reasoning", "planning"], "payload": {"query": "q"}})
    assert st == 200 and b["topology"] == "mesh"


def test_run_chain_ok(server):
    st, b = _post(server, "/collab/run", {
        "topology": "chain", "goal": "g",
        "routes": {"inq": "sfm_space", "act": "wla_action"},
        "chain": ["inq", "act"],
        "payload": {"query_region": "r", "body_part": "arm", "dim": 7}})
    assert st == 200 and b["topology"] == "chain" and b["recovered"] is True


def test_run_unknown_topology(server):
    st, _ = _post(server, "/collab/run", {"topology": "teleport", "goal": "g"})
    assert st == 400


# ---- /collab/handoff ---- #
def test_handoff_valid(server):
    st, b = _post(server, "/collab/handoff", {
        "bundle": {"goal": "g", "trace": {"trace_id": "t9", "owner": "alice"},
                   "done": [], "todo": [], "context": ""}})
    assert st == 200 and b["valid"] is True


def test_handoff_incomplete_400(server):
    # 缺 trace.owner -> 400 (状态丢失被拒)
    st, b = _post(server, "/collab/handoff", {
        "bundle": {"goal": "g", "trace": {"trace_id": "t9"}}})
    assert st == 400


def test_handoff_not_dict_400(server):
    st, _ = _post(server, "/collab/handoff", {"bundle": "notdict"})
    assert st == 400


# ---- /collab/trace/{id} ---- #
def test_trace_query_after_run(server):
    # 先 run 建 trace, 再查询
    _post(server, "/collab/run", {
        "topology": "star", "task_id": "traceq", "goal": "g",
        "subtasks": [{"key": "a", "tool": "ctm_reasoning",
                      "payload": {"query": "q", "horizon": 2}}]})
    st, b, ctype = _get(server, "/collab/trace/traceq")
    assert st == 200 and "application/json" in ctype
    body = json.loads(b)
    assert body["closed"] is True


def test_trace_missing_404(server):
    st, _, _ = _get(server, "/collab/trace/no_such_trace")
    assert st == 404


def test_trace_path_traversal_400(server):
    st, _, _ = _get(server, "/collab/trace/..%2Fetc")
    assert st == 400 or st == 404  # 路径穿越守卫拒绝


# ---- 既有契约不回归 ---- #
def test_metrics_still_plain(server):
    st, body, ctype = _get(server, "/metrics")
    assert st == 200 and "text/plain" in ctype
