"""
v2.6.2 真实 HTTP 服务逐接口验证
==================================
启动服务 (--checkpoint checkpoints/predictor_v2.6.2.pt, port 18765), 逐接口验证:
- GET  /health           -> 2.6.2, predictor_trained=true
- POST /evaluate         -> 含 calibration/interval 段
- POST /predict          -> prediction 形状 [B,6]
- POST /detect-ood       -> 200
- GET  /metrics          -> 200 (Prometheus 文本)
- POST /counterfactual   -> 200 (v2.6 新端点)
- POST /identify         -> 200 (v2.6 新端点)
- POST /risk             -> 200 (v2.6 新端点)
- POST /diff-checkpoints  -> 200 (v2.6 新端点, 对比 v2.5.2 vs v2.6.2)
- POST /rollback         -> 先 load 另一件再 rollback=200; 无历史=409
- 非法输入(/predict 缺 window) -> 400
- 未训练端点(全新 service 实例 /counterfactual) -> 409
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
CKPT = ROOT / "checkpoints" / "predictor_v2.6.2.pt"
PORT = 18765
BASE = f"http://127.0.0.1:{PORT}"
EXPECT_VERSION = "2.6.2"


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

    print("启动训练态服务 (checkpoint v2.6.2, port %d)..." % PORT)
    proc = subprocess.Popen(
        [sys.executable, "-m", "udos.server",
         "--host", "127.0.0.1", "--port", str(PORT),
         "--preset", "small", "--checkpoint", str(CKPT)],
        cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        if not _wait_health(BASE):
            print("ERROR: 训练态服务启动超时")
            return 1

        # 1. /health
        code, body = _get("/health")
        bj = json.loads(body)
        log.append(f"GET /health -> {code}, version={bj.get('version')}, "
                   f"predictor_trained={bj.get('predictor_trained')}")
        assert code == 200
        assert bj["version"] == EXPECT_VERSION, bj["version"]
        assert bj["predictor_trained"] is True

        # 2. /evaluate 含 calibration/interval 段
        code, body = _post("/evaluate", {"n_per_kind": 8, "horizon": 4})
        log.append(f"POST /evaluate -> {code}")
        assert code == 200, body
        metrics = body.get("metrics", {})
        assert "calibration" in metrics, "/evaluate 缺 calibration"
        assert "interval" in metrics, "/evaluate 缺 interval"

        # 3. /predict 形状正确
        import torch
        torch.manual_seed(42)
        win = torch.randn(2, 6, 6).tolist()
        code, body = _post("/predict", {"window": win})
        log.append(f"POST /predict -> {code}")
        assert code == 200, body
        pred = body.get("prediction") or body.get("pred")
        assert pred is not None, f"/predict 无 prediction: {list(body)}"
        assert len(pred) == 2 and len(pred[0]) == 6, f"prediction 形状 {len(pred)}x{len(pred[0])}"

        # 4. /detect-ood
        code, body = _post("/detect-ood", {"sequence": torch.randn(3, 6, 6).tolist()})
        log.append(f"POST /detect-ood -> {code}")
        assert code == 200, body

        # 5. /metrics
        code, body = _get("/metrics")
        log.append(f"GET /metrics -> {code} ({len(body)} bytes)")
        assert code == 200 and "udos_requests_total" in body

        # 6. /counterfactual (新端点)
        win6 = [[0, 0, 0, 1, 0, 0],
                [0.5, 0, 0, 1, 0, 0],
                [1, 0, 0, 1, 0, 0],
                [1.5, 0, 0, 1, 0, 0],
                [2, 0, 0, 1, 0, 0],
                [2.5, 0, 0, 1, 0, 0]]
        code, body = _post("/counterfactual", {"window": win6, "horizon": 2})
        log.append(f"POST /counterfactual -> {code}, ate_mean={body.get('ate_mean')}")
        assert code == 200, body
        assert "counterfactual" in body and "baseline" in body

        # 7. /identify (新端点)
        code, body = _post("/identify", {"window": win6, "horizon": 2, "grid_size": 5})
        log.append(f"POST /identify -> {code}, params={body.get('identified_params')}")
        assert code == 200, body
        assert "identified_params" in body

        # 8. /risk (新端点)
        code, body = _post("/risk", {"window": win6, "horizon": 1})
        log.append(f"POST /risk -> {code}, score={body.get('risk_score')}, "
                   f"level={body.get('risk_level')}")
        assert code == 200, body
        assert body["risk_level"] in ("low", "medium", "high")

        # 9. /diff-checkpoints (新端点, 对比 v2.5.2 vs v2.6.2)
        code, body = _post("/diff-checkpoints",
                           {"name_a": "predictor_v2.5.2.pt",
                            "name_b": "predictor_v2.6.2.pt",
                            "n_per_kind": 8})
        log.append(f"POST /diff-checkpoints -> {code}")
        assert code == 200, body

        # 10. /rollback: 无历史 -> 409
        code, body = _post("/rollback", {})
        log.append(f"POST /rollback (无历史) -> {code}")
        assert code == 409, f"无历史应 409, 实际 {code}"
        # load 另一件后 rollback -> 200
        code, body = _post("/load", {"name": "predictor_v2.5.0.pt"})
        log.append(f"POST /load v2.5.0 -> {code}")
        assert code == 200, body
        code, body = _post("/rollback", {})
        log.append(f"POST /rollback (有历史) -> {code}")
        assert code == 200, f"有历史应 200, 实际 {code}"

        # 11. 非法输入 /predict 缺 window -> 400
        code, body = _post("/predict", {})
        log.append(f"POST /predict (缺 window) -> {code}")
        assert code == 400, f"缺 window 应 400, 实际 {code}"

    except AssertionError as e:
        print(f"\nVERIFICATION FAILED: {e}")
        ok = False
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()

    # 12. 未训练端点 409: 全新 service 实例 (无 --checkpoint) 测 /counterfactual
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
        code, body = _post("/counterfactual",
                           {"window": win6, "horizon": 2}, base=base2)
        log.append(f"POST /counterfactual (未训练) -> {code}")
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

    print("\n===== v2.6.2 服务逐接口验证 %s =====" % ("通过" if ok else "失败"))
    for line in log:
        print(f"  {line}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
