"""
端到端部署冒烟: 真实拉起 HTTP 服务, 逐接口打请求并断言, 最后回收进程。
用法: python scripts/smoke_test.py [--port 8011] ; 退出码非 0 即冒烟失败。
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from udos import __version__  # noqa: E402


def request(method, url, body=None, timeout=30):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())


def wait_health(base, tries=40, gap=0.5):
    for _ in range(tries):
        try:
            code, body = request("GET", base + "/health")
            if code == 200 and body["status"] == "ok":
                return True
        except Exception:
            pass
        time.sleep(gap)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8011)
    ap.add_argument("--preset", default="small")
    args = ap.parse_args()
    base = f"http://127.0.0.1:{args.port}"

    proc = subprocess.Popen(
        [sys.executable, "-m", "udos.server", "--host", "127.0.0.1",
         "--port", str(args.port), "--preset", args.preset],
        cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    checks = []
    try:
        assert wait_health(base), "服务健康检查超时"
        checks.append(("GET /health", True))

        from demos.scene_factory import build_factory_scene
        from udos.pce_format import PCEParser
        scene = json.loads(PCEParser.dumps(build_factory_scene(n_steps=8)))

        c, b = request("POST", base + "/internalize", {"scene": scene})
        checks.append(("POST /internalize", c == 200 and b["lora_params"] > 0))

        c, b = request("POST", base + "/reason",
                       {"scene": scene, "query": "smoke"})
        ok = (c == 200 and b["ticks_used"] > 0 and len(b["causal_chain"]) > 0
              and b["scene_conditioned"] is True)
        checks.append(("POST /reason (GPM场景条件化)", ok))

        # v2.2: 训练前评估/落盘 -> 409 (依赖尚未挂载的预测器)
        c, _ = request("POST", base + "/evaluate", {})
        checks.append(("POST /evaluate 未训练返回409", c == 409))

        # v2.1: /train 参数化场景条件多步训练, 返回评估体系
        c, b = request("POST", base + "/train",
                       {"epochs": 3, "n_per_kind": 6, "horizon": 4}, timeout=120)
        ev = b.get("evaluation", {})
        ok = (c == 200 and b["trained"] is True and b["horizon"] == 4
              and b["predictor_params"] > 0
              and len(b["train_loss_curve"]) == 3
              and len(b.get("phys_violation_curve", [])) == 3
              and len(ev.get("rollout_mse_curve", [])) == 4
              and "condition_gain_x" in ev.get("ablation", {}))
        checks.append(("POST /train 场景条件多步训练+评估", ok))

        # v2.2: 训练后在线评估 (新鲜测试集) 与落盘
        c, b = request("POST", base + "/evaluate",
                       {"n_per_kind": 8, "horizon": 4, "seed": 2024})
        m = b.get("metrics", {})
        ok_eval = (c == 200 and "rollout_growth_x" in m
                   and "confidence_stratification" in m)
        checks.append(("POST /evaluate 在线评估+置信分层", ok_eval))

        c, b = request("POST", base + "/calibrate",
                       {"n_per_kind": 8, "horizon": 4})
        ok_cal = (c == 200 and b.get("independent_test", {})
                  .get("calibration") is not None)
        checks.append(("POST /calibrate 保序校准+独立集对照", ok_cal))

        c, b = request("POST", base + "/save", {"name": "smoke_v22.pt"})
        saved_path = ROOT / b.get("path", "") if c == 200 else None
        checks.append(("POST /save 落盘 checkpoint",
                       c == 200 and b.get("bytes", 0) > 0
                       and saved_path is not None and saved_path.exists()))

        c, b = request("GET", base + "/checkpoints")
        checks.append(("GET /checkpoints 列出可用件",
                       c == 200 and b.get("count", 0) >= 1))
        c, b = request("POST", base + "/load", {"name": "smoke_v22.pt"})
        checks.append(("POST /load 载回(含校准器)",
                       c == 200 and b.get("calibrated") is True))
        c, _ = request("POST", base + "/load", {"name": "missing.pt"})
        checks.append(("POST /load 不存在件返回400", c == 400))

        c, b = request("POST", base + "/save", {"name": "../escape.pt"})
        # 危险名被无害化, 最终仍落在 checkpoints 内 (不报 500)
        checks.append(("POST /save 名称无害化", c == 200))
        c, h = request("GET", base + "/health")
        checks.append(("GET /health 已挂载预测器",
                       c == 200 and h["predictor_trained"] is True
                       and h["version"] == __version__ == "2.3.1"))

        c, b = request("POST", base + "/reason",
                       {"scene": scene, "query": "next", "horizon": 4})
        st = b.get("predicted_next_state") or {}
        fs = b.get("future_states") or []
        ok = (c == 200 and len(st.get("position", [])) == 3
              and len(st.get("velocity", [])) == 3 and len(fs) == 4)
        checks.append(("POST /reason 多步滚动推演(4步)", ok))

        c, _ = request("POST", base + "/reset", {"scene_id": scene["scene_id"]})
        checks.append(("POST /reset", c == 200))

        c, b = request("POST", base + "/reason", {})  # 缺字段 -> 400
        checks.append(("POST /reason 缺字段返回400", c == 400))

        c, b = request("POST", base + "/train", {"epochs": 99999})  # 越界 -> 400
        checks.append(("POST /train 越界参数返回400", c == 400))

        c, b = request("GET", base + "/demo")
        checks.append(("GET /demo 全链路", c == 200 and "reason" in b))
    finally:
        proc.terminate()
        try:
            out, _ = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            out, _ = proc.communicate()

    # 清理本脚本 /save 产生的临时 checkpoint
    for tmp in ("smoke_v22.pt", "escape.pt"):
        p = ROOT / "checkpoints" / tmp
        if p.exists():
            os.remove(p)

    # --checkpoint 预加载, 启动即带训练态 (优先当前版本, 依次回退 2.2.1 / 2.1)
    ckpt = ROOT / "checkpoints" / "predictor_v2.3.1.pt"
    if not ckpt.exists():
        ckpt = ROOT / "checkpoints" / "predictor_v2.2.1.pt"
    if not ckpt.exists():
        ckpt = ROOT / "checkpoints" / "predictor_v2.1.0.pt"
    if ckpt.exists():
        p2 = subprocess.Popen(
            [sys.executable, "-m", "udos.server", "--host", "127.0.0.1",
             "--port", str(args.port + 1), "--preset", args.preset,
             "--checkpoint", str(ckpt)],
            cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        try:
            base2 = f"http://127.0.0.1:{args.port + 1}"
            up = wait_health(base2)
            c, h = request("GET", base2 + "/health") if up else (0, {})
            checks.append(("启动 --checkpoint 即带训练态",
                           up and h.get("predictor_trained") is True))
        finally:
            p2.terminate()
            try:
                p2.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                p2.kill()

    print("\n===== 冒烟结果 =====")
    failed = 0
    for name, ok in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        failed += 0 if ok else 1
    if failed:
        print(f"\n{failed} 项失败")
        print(out.decode()[-2000:])
        sys.exit(1)
    print(f"\n全部 {len(checks)} 项冒烟通过")


if __name__ == "__main__":
    main()
