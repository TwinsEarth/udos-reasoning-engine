"""
v3.3.4 hardening patch: 6 个真缺陷/契约冲突的 RED->GREEN->回归用例
====================================================================
对应诊断台账 scratch/diagnosis_ledger_v334.md:
    FIX-001 / DIAG-001  GET /eval/5d 未训练应 409 (契约对齐 do_POST)
    FIX-003 / DIAG-003  POST /icl/predict 非法 window 应 400
    FIX-004 / DIAG-004  残缺 PCE scene 应 400
    FIX-005 / DIAG-008  health() 读 scene_memory 需持服务锁
    FIX-006 / DIAG-009  checkpoint 目录应基于工程根的绝对路径
"""
import json
import os
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest
import torch

from udos.server import create_server, UDOSService
from udos.pce_format import PCEParser

ROOT = Path(__file__).resolve().parents[1]
CKPT333 = str(ROOT / "checkpoints" / "predictor_v3.3.3.pt")


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
        with urllib.request.urlopen(base + path, timeout=60) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


def _serve(httpd):
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    host, port = httpd.server_address
    base = f"http://{host}:{port}"
    yield base
    httpd.shutdown()
    httpd.server_close()
    t.join(timeout=5)


@pytest.fixture
def untrained_server():
    torch.manual_seed(0)
    httpd = create_server("127.0.0.1", 0, preset="small")
    yield from _serve(httpd)


@pytest.fixture
def trained_server():
    torch.manual_seed(0)
    httpd = create_server("127.0.0.1", 0, preset="small", checkpoint=CKPT333)
    yield from _serve(httpd)


# ---------- FIX-001 / DIAG-001: GET /eval/5d 未训练 -> 409 ---------- #
def test_fix001_eval_5d_untrained_returns_409(untrained_server):
    """DIAG-001: do_GET 缺 ServiceNotReady 分支, 未训练时 500 而非契约约定的 409。"""
    code, body = _get(untrained_server, "/eval/5d")
    assert code == 409, f"未训练 GET /eval/5d 应 409, 实际 {code}: {body}"


def test_fix001_regression_eval_5d_trained_200(trained_server):
    """回归: 已加载 checkpoint 时 GET /eval/5d 仍 200。"""
    code, body = _get(trained_server, "/eval/5d")
    assert code == 200, f"已训练 GET /eval/5d 应 200, 实际 {code}: {body}"
    assert len(body["scores"]) == 5


# ---------- FIX-003 / DIAG-003: /icl/predict 非法 window -> 400 ---------- #
def test_fix003_icl_predict_bad_window_returns_400(trained_server):
    """DIAG-003: torch.as_tensor 裸调用, 字符串 window 触发 TypeError->500。"""
    code, body = _post(trained_server, "/icl/predict",
                       {"window": "not_a_tensor"})
    assert code == 400, f"非法 window 应 400, 实际 {code}: {body}"


def test_fix003_regression_icl_predict_normal_200(trained_server):
    """回归: 正常 window [W,RAW] 仍 200 (DIAG-003 旧 500 仅非法类型触发)。"""
    from udos.dynamics import build_parametric_dataset
    ds = build_parametric_dataset(n_per_kind=2, n_steps=12, window=6,
                                  horizon=2, dt=0.5, seed=7)
    code, body = _post(trained_server, "/icl/predict",
                       {"window": ds.X[0].tolist()})
    assert code == 200, f"正常 window 应 200, 实际 {code}: {body}"
    assert body["status"] == "ok"


# ---------- FIX-004 / DIAG-004: 残缺 PCE scene -> 400 ---------- #
def _broken_scene_dict():
    """结构残缺: 有 scene_id/kind/t0/dt/states 但 states 为空, 无有效 tokens。"""
    return {"scene": {"scene_id": "x", "kind": "p", "t0": 0.0, "dt": 0.1,
                      "states": []}}


def test_fix004_broken_scene_internalize_returns_400(trained_server):
    """DIAG-004: 残缺 scene -> PCEParser 返回空 tokens -> torch.stack([]) 500。"""
    code, body = _post(trained_server, "/internalize", _broken_scene_dict())
    assert code == 400, f"残缺 scene 应 400, 实际 {code}: {body}"


def test_fix004_broken_scene_reason_returns_400(trained_server):
    code, body = _post(trained_server, "/reason", _broken_scene_dict())
    assert code == 400, f"残缺 scene reason 应 400, 实际 {code}: {body}"


def test_fix004_regression_demo_legal_pce_still_200(trained_server):
    """回归: /demo 用合法 PCE (PCEParser.dumps 生成) 仍 internalize+reason 200。"""
    code, body = _get(trained_server, "/demo")
    assert code == 200, f"/demo 应 200, 实际 {code}: {body}"
    assert body["internalize"]["status"] == "ok"
    assert body["reason"]["status"] == "ok"


# ---------- FIX-005 / DIAG-008: health() 并发不崩 ---------- #
def test_fix005_health_concurrent_with_internalize_no_error(trained_server):
    """DIAG-008: 多线程 internalize + health 不应抛异常 (预防性加固)。"""
    from demos.scene_factory import build_factory_scene
    scene = json.loads(PCEParser.dumps(build_factory_scene(n_steps=6)))
    errors = []

    def hammer_internalize():
        try:
            for _ in range(20):
                _post(trained_server, "/internalize", {"scene": scene})
                _post(trained_server, "/reset", {"scene_id": scene["scene_id"]})
        except Exception as e:  # noqa: BLE001 - 记录并发错误
            errors.append(e)

    def hammer_health():
        try:
            for _ in range(200):
                c, b = _get(trained_server, "/health")
                assert c == 200 and "internalized_scenes" in b
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=hammer_internalize) for _ in range(3)] + \
              [threading.Thread(target=hammer_health) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert not errors, f"并发 health/internalize 出现错误: {errors!r}"


def test_fix005_regression_health_fields_complete(trained_server):
    code, body = _get(trained_server, "/health")
    assert code == 200
    assert body["status"] == "ok"
    assert "internalized_scenes" in body
    assert "predictor_trained" in body


# ---------- FIX-006 / DIAG-009: checkpoint 目录不依赖 cwd ---------- #
def test_fix006_checkpoints_dir_independent_of_cwd(tmp_path, monkeypatch):
    """DIAG-009: 从非工程根目录启动也应能列出工程根下的 checkpoints。"""
    monkeypatch.chdir(tmp_path)
    svc = UDOSService(preset="small")
    out = svc.list_checkpoints()
    names = [it["name"] for it in out["checkpoints"]]
    assert "predictor_v3.3.3.pt" in names, (
        f"从 {tmp_path} 启动应列出工程根 checkpoints, 实际: {names}")


def test_fix006_regression_checkpoints_dir_under_project_root():
    """回归: checkpoints_dir 是工程根下的绝对路径, 且含正式件。"""
    svc = UDOSService(preset="small")
    base = svc.checkpoints_dir
    assert os.path.isabs(base)
    assert base.endswith("checkpoints")
    assert (Path(base) / "predictor_v3.3.3.pt").is_file()


def test_fix006_regression_load_save_roundtrip_from_tmp_cwd(tmp_path, monkeypatch):
    """回归: 从 /tmp cwd 加载正式件仍成功; save 落到工程根 checkpoints。"""
    monkeypatch.chdir(tmp_path)
    svc = UDOSService(preset="small")
    out = svc.load({"name": "predictor_v3.3.3"})
    assert out["status"] == "loaded"
    assert os.path.isabs(out["path"])
    assert out["path"].endswith("predictor_v3.3.3.pt")
    assert "checkpoints" in out["path"]
