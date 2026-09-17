"""
v2.5.2 真实 HTTP 服务逐接口验证
================================
启动服务 (--checkpoint checkpoints/predictor_v2.5.2.pt), 逐接口验证:
- /health = 2.5.2
- /evaluate 含 calibration/interval 段
- /detect-ood 200/409
- /metrics 200
- /rollback 200/409
- /predict guard 触发
输出验证日志。
"""
import json
import subprocess
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "checkpoints" / "predictor_v2.5.2.pt"
PORT = 18252
BASE = f"http://127.0.0.1:{PORT}"


def _get(path):
    try:
        with urllib.request.urlopen(BASE + path, timeout=10) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def _post(path, obj):
    data = json.dumps(obj).encode("utf-8")
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())


def main():
    log = []
    ok = True

    # 启动服务
    print("启动服务...")
    proc = subprocess.Popen(
        [sys.executable, "-m", "udos.server",
         "--host", "127.0.0.1", "--port", str(PORT),
         "--preset", "small", "--checkpoint", str(CKPT)],
        cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        # 等待启动
        for _ in range(30):
            time.sleep(1)
            try:
                code, _ = _get("/health")
                if code == 200:
                    break
            except Exception:
                pass
        else:
            print("ERROR: 服务启动超时")
            return 1

        # 1. /health
        code, body = _get("/health")
        body_j = json.loads(body)
        log.append(f"GET /health -> {code}, version={body_j.get('version')}")
        assert code == 200, f"/health 期望 200, 实际 {code}"
        assert body_j["version"] == "2.5.2", \
            f"/health version={body_j['version']} != 2.5.2"
        print(log[-1])

        # 2. /evaluate 含 calibration/interval 段
        code, body = _post("/evaluate", {"n_per_kind": 8, "horizon": 4})
        log.append(f"POST /evaluate -> {code}")
        assert code == 200
        metrics = body.get("metrics", {})
        assert "calibration" in metrics, "/evaluate 缺 calibration 段"
        assert "interval" in metrics, "/evaluate 缺 interval 段"
        assert "service_metrics" in body, "/evaluate 缺 service_metrics 段"
        print(f"  calibration keys: {list(metrics['calibration'].keys())[:3]}...")
        print(f"  interval keys: {list(metrics['interval'].keys())}")

        # 3. /detect-ood 200 (已挂检测器) / 409 (无检测器场景)
        # 构造一个正常 sequence
        import torch
        torch.manual_seed(42)
        seq = torch.randn(3, 6, 6).tolist()
        code, body = _post("/detect-ood", {"sequence": seq})
        log.append(f"POST /detect-ood -> {code}")
        assert code == 200, f"/detect-ood 期望 200, 实际 {code}"
        assert "ood" in body
        print(f"  ood_rate={body.get('ood_rate')}, threshold={body.get('threshold')}")

        # 4. /metrics 200 (Prometheus 文本)
        code, body = _get("/metrics")
        log.append(f"GET /metrics -> {code}")
        assert code == 200, f"/metrics 期望 200, 实际 {code}"
        assert "udos_requests_total" in body
        assert "udos_latency_seconds" in body
        print(f"  Prometheus 文本 {len(body)} 字节")

        # 5. /rollback 409 (无历史: 启动预加载占栈底, 未再 load)
        code, body = _post("/rollback", {})
        log.append(f"POST /rollback (无历史) -> {code}")
        assert code == 409, f"/rollback 无历史期望 409, 实际 {code}"
        print(f"  message: {body.get('message', '')[:60]}")

        # load 另一个 checkpoint 后 rollback 200
        code, body = _post("/load", {"name": "predictor_v2.5.0.pt"})
        log.append(f"POST /load v2.5.0 -> {code}")
        assert code == 200
        code, body = _post("/rollback", {})
        log.append(f"POST /rollback (有历史) -> {code}")
        assert code == 200, f"/rollback 有历史期望 200, 实际 {code}"
        print(f"  restored: {body.get('restored_path', '')[-40:]}")

        # 6. /predict guard 触发
        code, body = _post("/predict", {
            "window": torch.randn(2, 6, 6).tolist(),
            "guard": True})
        log.append(f"POST /predict (guard=True) -> {code}")
        assert code == 200
        assert body["guard"]["enabled"] is True
        print(f"  guard stats: {body['guard']}")

        # 7. /checkpoints
        code, body = _get("/checkpoints")
        body_j = json.loads(body)
        log.append(f"GET /checkpoints -> {code}")
        assert code == 200
        print(f"  {body_j.get('count')} checkpoints available")

        print("\n===== 全部接口验证通过 =====")
        for line in log:
            print(f"  {line}")

    except AssertionError as e:
        print(f"\nVERIFICATION FAILED: {e}")
        ok = False
    finally:
        proc.terminate()
        proc.wait(timeout=10)

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
