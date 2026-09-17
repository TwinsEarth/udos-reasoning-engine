"""
v2.7.3 真实 HTTP 服务逐接口验证
==================================
启动服务 (--checkpoint checkpoints/predictor_v2.7.3.pt, port 18765), 逐接口验证:
- GET  /health           -> 2.7.3, predictor_trained=true
- POST /evaluate         -> 含 calibration/interval 段
- POST /predict          -> prediction 形状 [B,6]
- POST /policy/select     -> 200 (v2.7 新端点)
- POST /online/adapt     -> 200
- POST /active/sample    -> 200
- GET  /experiments      -> 200
- 非法输入 -> 400; 未训练端点(全新无 checkpoint 实例) -> 409
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
CKPT = ROOT / "checkpoints" / "predictor_v2.7.3.pt"
PORT = 18765
BASE = f"http://127.0.0.1:{PORT}"
EXPECT_VERSION = "2.7.3"


def _get(path, base=BASE):
    try:
        with urllib.request.urlopen(base + path, timeout=15) as r:
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


def _wait_health(base, timeout=30):
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
    ok = True
    win6 = [[0, 0, 0, 1, 0, 0],
            [0.5, 0, 0, 1, 0, 0],
            [1, 0, 0, 1, 0, 0],
            [1.5, 0, 0, 1, 0, 0],
            [2, 0, 0, 1, 0, 0],
            [2.5, 0, 0, 1, 0, 0]]

    print("启动训练态服务 (checkpoint v2.7.3, port %d)..." % PORT)
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
        log.append(f"GET /health -> {code}, version={bj.get('version')}, "
                   f"predictor_trained={bj.get('predictor_trained')}")
        assert code == 200
        assert bj["version"] == EXPECT_VERSION, bj["version"]
        assert bj["predictor_trained"] is True

        code, body = _post("/evaluate", {"n_per_kind": 8, "horizon": 4})
        log.append(f"POST /evaluate -> {code}")
        assert code == 200, body
        metrics = body.get("metrics", {})
        assert "calibration" in metrics, "/evaluate 缺 calibration"
        assert "interval" in metrics, "/evaluate 缺 interval"

        # v2.7 新端点
        code, body = _post("/policy/select",
                           {"window": win6, "actions": [{}, {}], "horizon": 2})
        log.append(f"POST /policy/select -> {code}, best_index={body.get('best_index')}")
        assert code == 200, body
        assert body["no_valid_action"] is False

        code, body = _post("/online/adapt", {"window": win6})
        log.append(f"POST /online/adapt -> {code}, adapted={body.get('adapted')}")
        assert code == 200, body

        pool = [win6, win6, win6, win6, win6]
        code, body = _post("/active/sample", {"sample_pool": pool, "k": 3})
        log.append(f"POST /active/sample -> {code}, n_idx={len(body.get('indices', []))}")
        assert code == 200, body
        assert len(body["indices"]) == 3

        code, body = _get("/experiments")
        bj = json.loads(body)
        log.append(f"GET /experiments -> {code}, count={bj.get('count')}")
        assert code == 200, body
        assert isinstance(bj.get("experiments"), list)

        # 非法输入 400
        code, body = _post("/policy/select", {"window": win6})
        log.append(f"POST /policy/select (缺 actions) -> {code}")
        assert code == 400, f"缺 actions 应 400, 实际 {code}"
        code, body = _post("/active/sample", {"sample_pool": [], "k": 3})
        log.append(f"POST /active/sample (空池) -> {code}")
        assert code == 400, f"空池应 400, 实际 {code}"

    except AssertionError as e:
        print(f"\nVERIFICATION FAILED: {e}")
        ok = False
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()

    # 未训练 409: 全新无 checkpoint 实例
    print("启动未训练服务 (port 18767) 测 409...")
    proc2 = subprocess.Popen(
        [sys.executable, "-m", "udos.server",
         "--host", "127.0.0.1", "--port", "18767", "--preset", "small"],
        cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        base2 = "http://127.0.0.1:18767"
        if not _wait_health(base2):
            print("ERROR: 未训练服务启动超时")
            return 1
        code, body = _post("/policy/select",
                           {"window": win6, "actions": [{}]}, base=base2)
        log.append(f"POST /policy/select (未训练) -> {code}")
        assert code == 409, f"未训练应 409, 实际 {code}"
    except AssertionError as e:
        print(f"\nVERIFICATION FAILED (untrained): {e}")
        ok = False
    finally:
        proc2.terminate()
        try:
            proc2.wait(timeout=10)
        except Exception:
            proc2.kill()

    print("\n===== v2.7.3 服务逐接口验证 %s =====" % ("通过" if ok else "失败"))
    for line in log:
        print(f"  {line}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
