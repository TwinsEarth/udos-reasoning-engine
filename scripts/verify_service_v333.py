"""
v3.3.3 真实 HTTP 服务逐接口验证
====================================
启动服务 (--checkpoint checkpoints/predictor_v3.3.3.pt, port 18899), 逐接口验证:
- GET  /health           -> 3.3.3, predictor_trained=true
- POST /evaluate         -> 含 calibration/interval 段
- POST /loop/step        -> 200 (v2.8)
- POST /multitask/predict-> 200 (v2.8)
- POST /retarget/convert -> 200 (v2.9)
- POST /affordance/score -> 200 (v2.9)
- POST /future/predict   -> 200 (v3.0)
- GET  /eval/5d          -> 200 (v3.0)
- POST /action/tokenize  -> 200 (v3.1)
- POST /action/detokenize-> 200 (v3.1)
- POST /augment/generate -> 200 (v3.2)
- POST /icl/predict      -> 200 (v3.2)
非法输入 -> 400; 未训练端点(全新无 checkpoint 实例) -> 409。
验证完毕关闭服务进程。
"""
import json
import subprocess
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "checkpoints" / "predictor_v3.3.3.pt"
PORT = 18899
BASE = f"http://127.0.0.1:{PORT}"
EXPECT_VERSION = "3.3.3"

# 窗口: 6 步 x raw_dim 6
WIN = [[0, 0, 0, 1, 0, 0],
       [0.5, 0, 0, 1, 0, 0],
       [1, 0, 0, 1, 0, 0],
       [1.5, 0, 0, 1, 0, 0],
       [2, 0, 0, 1, 0, 0],
       [2.5, 0, 0, 1, 0, 0]]


def _get(path, base=BASE):
    try:
        with urllib.request.urlopen(base + path, timeout=20) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def _post(path, obj, base=BASE):
    data = json.dumps(obj).encode("utf-8")
    req = urllib.request.Request(base + path, data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode())
        except Exception:
            return e.code, {}


def _wait_health(base, timeout=40):
    for _ in range(timeout):
        try:
            code, _ = _get("/health", base)
            if code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def main():
    log = []
    status_table = []
    ok = True

    print("启动训练态服务 (checkpoint v3.3.3, port %d)..." % PORT)
    proc = subprocess.Popen(
        [sys.executable, "-m", "udos.server",
         "--host", "127.0.0.1", "--port", str(PORT),
         "--preset", "small", "--checkpoint", str(CKPT)],
        cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        if not _wait_health(BASE):
            print("ERROR: 训练态服务启动超时")
            return 1

        code, body = _get("/health")
        bj = json.loads(body)
        status_table.append(("GET /health", code))
        log.append(f"GET /health -> {code}, version={bj.get('version')}, "
                   f"predictor_trained={bj.get('predictor_trained')}")
        assert code == 200
        assert bj["version"] == EXPECT_VERSION, bj["version"]
        assert bj["predictor_trained"] is True

        code, body = _post("/evaluate", {"n_per_kind": 8, "horizon": 4})
        status_table.append(("POST /evaluate", code))
        log.append(f"POST /evaluate -> {code}")
        assert code == 200, body
        metrics = body.get("metrics", {})
        assert "calibration" in metrics, "/evaluate 缺 calibration"
        assert "interval" in metrics, "/evaluate 缺 interval"

        # ---- v2.8 Physical Loop / MultiTask ----
        code, body = _post("/loop/step", {"window": WIN, "horizon": 4})
        status_table.append(("POST /loop/step", code))
        log.append(f"POST /loop/step -> {code}, steps={body.get('steps')}")
        assert code == 200, body

        code, body = _post("/multitask/predict", {"window": WIN})
        status_table.append(("POST /multitask/predict", code))
        log.append(f"POST /multitask/predict -> {code}")
        assert code == 200, body

        # ---- v2.9 retargeting / affordance ----
        retarget_actions = [[0.01 * (i % 60) for _ in range(60)] for i in range(4)]
        code, body = _post("/retarget/convert", {
            "actions": retarget_actions,
            "source": "prime_u_60dof", "target": "gripper_4dof"})
        status_table.append(("POST /retarget/convert", code))
        log.append(f"POST /retarget/convert -> {code}, shape={body.get('shape')}")
        assert code == 200, body

        code, body = _post("/affordance/score", {
            "state": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            "objects": [[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]]]})
        status_table.append(("POST /affordance/score", code))
        log.append(f"POST /affordance/score -> {code}, suggestion={body.get('suggestion')}")
        assert code == 200, body

        # ---- v3.0 future multimodal / eval5d ----
        code, body = _post("/future/predict", {"window": WIN, "horizon": 4})
        status_table.append(("POST /future/predict", code))
        log.append(f"POST /future/predict -> {code}, shapes={body.get('shapes')}")
        assert code == 200, body

        code, body = _get("/eval/5d")
        body = json.loads(body)
        status_table.append(("GET /eval/5d", code))
        log.append(f"GET /eval/5d -> {code}, composite={body.get('composite')}, "
                   f"not_physbrain={body.get('not_physbrain_leaderboard')}")
        assert code == 200, body

        # ---- v3.1 action tokenize / detokenize ----
        code, body = _post("/action/tokenize", {"actions": WIN})
        status_table.append(("POST /action/tokenize", code))
        log.append(f"POST /action/tokenize -> {code}, tokens_shape={body.get('shape')}")
        assert code == 200, body

        code, body = _post("/action/detokenize", {"tokens": [0, 1, 2, 3]})
        status_table.append(("POST /action/detokenize", code))
        log.append(f"POST /action/detokenize -> {code}, shape={body.get('shape')}")
        assert code == 200, body

        # ---- v3.2 augment / icl ----
        code, body = _post("/augment/generate", {"window": WIN, "noise_sigma": 0.01})
        status_table.append(("POST /augment/generate", code))
        log.append(f"POST /augment/generate -> {code}, shape={body.get('shape')}")
        assert code == 200, body

        code, body = _post("/icl/predict", {"window": WIN})
        status_table.append(("POST /icl/predict", code))
        log.append(f"POST /icl/predict -> {code}, shape={body.get('shape')}")
        assert code == 200, body

        # ---- 非法输入 -> 400 ----
        bad = []
        code, body = _post("/loop/step", {"window": WIN, "horizon": 99})
        bad.append(("POST /loop/step(horizon越界)", code))
        assert code == 400, f"horizon 越界应 400, 实际 {code}"
        code, body = _post("/multitask/predict", {})
        bad.append(("POST /multitask/predict(缺window)", code))
        assert code == 400, f"缺 window 应 400, 实际 {code}"
        code, body = _post("/retarget/convert", {"actions": retarget_actions})
        bad.append(("POST /retarget/convert(缺source)", code))
        assert code == 400, f"缺 source 应 400, 实际 {code}"
        code, body = _post("/action/tokenize", {})
        bad.append(("POST /action/tokenize(缺actions)", code))
        assert code == 400, f"缺 actions 应 400, 实际 {code}"
        for name, c in bad:
            status_table.append((name, c))
            log.append(f"{name} -> {c}")

    except AssertionError as e:
        print(f"\nVERIFICATION FAILED: {e}")
        ok = False
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()

    # ---- 未训练 409: 全新无 checkpoint 实例 ----
    print("启动未训练服务 (port 18897) 测 409...")
    proc2 = subprocess.Popen(
        [sys.executable, "-m", "udos.server",
         "--host", "127.0.0.1", "--port", "18897", "--preset", "small"],
        cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        base2 = "http://127.0.0.1:18897"
        if not _wait_health(base2):
            print("ERROR: 未训练服务启动超时")
            return 1
        for path, obj in [("/loop/step", {"window": WIN}),
                          ("/multitask/predict", {"window": WIN}),
                          ("/icl/predict", {"window": WIN})]:
            code, body = _post(path, obj, base=base2)
            status_table.append((path + "(未训练)", code))
            log.append(f"POST {path} (未训练) -> {code}")
            assert code == 409, f"未训练应 409, {path} 实际 {code}"
    except AssertionError as e:
        print(f"\nVERIFICATION FAILED (untrained): {e}")
        ok = False
    finally:
        proc2.terminate()
        try:
            proc2.wait(timeout=10)
        except Exception:
            proc2.kill()

    print("\n===== v3.3.3 服务逐接口验证 %s =====" % ("通过" if ok else "失败"))
    print("--- 端点状态码表 ---")
    for name, c in status_table:
        print(f"  {c:>3}  {name}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
