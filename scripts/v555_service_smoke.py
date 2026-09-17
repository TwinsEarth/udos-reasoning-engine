"""v5.5.5 服务实启动冒烟: 真实拉起 HTTP 服务 (带预训练 checkpoint),
打 /health、/checkpoints、POST /predict, 验证整条服务链在当前代码可启动可推理。

子进程方式启动 `python -m udos.server`, 结束后确保关闭。
用法: PYTHONPATH=. python3 scripts/v555_service_smoke.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT = int(os.environ.get("SMOKE_PORT", "8011"))
BASE = f"http://127.0.0.1:{PORT}"


def _get(path: str, timeout: float = 5.0):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        return r.status, json.loads(r.read().decode())


def _post(path: str, payload: dict, timeout: float = 10.0):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.loads(r.read().decode())


def main() -> int:
    ckpt = os.path.join(HERE, "checkpoints", "predictor_v4.3.9.pt")
    proc = subprocess.Popen(
        [sys.executable, "-m", "udos.server", "--host", "127.0.0.1",
         "--port", str(PORT), "--preset", "small", "--checkpoint", ckpt],
        cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True)
    out = {"version": "5.5.5", "port": PORT}
    try:
        # 等待 /health 就绪 (最多 40s, torch 导入+checkpoint 加载)
        health = None
        for _ in range(80):
            if proc.poll() is not None:
                break
            try:
                st, health = _get("/health")
                if st == 200:
                    break
            except Exception:
                time.sleep(0.5)
        assert health is not None and health.get("status") == "ok", health
        out["health"] = {"status": health.get("status"),
                         "predictor_trained": health.get("predictor_trained"),
                         "version": health.get("version"),
                         "internalized_scenes": health.get(
                             "internalized_scenes")}

        st, cps = _get("/checkpoints")
        out["checkpoints_status"] = st
        out["checkpoint_count"] = cps.get("count")

        # 真实单步预测: 匀速 vx=1.3, dt=0.5, 6 帧窗口, 6 维 [x,y,z,vx,vy,vz]
        v, dt = 1.3, 0.5
        window = [[0.1 + v * dt * i, 0.0, 0.0, v, 0.0, 0.0] for i in range(6)]
        # 盲预测作对照 (文档已证场景盲单步 MSE 高, 不做物理正确性断言)
        sb, blind = _post("/predict", {"window": window})
        out["blind_predict_status"] = sb
        out["blind_next_state"] = (blind["prediction"][0]
                                   if blind.get("shape") == [1, 6]
                                   else blind.get("prediction"))
        # 条件化路径: 匀速场景参数 [v0, accel, spring_omega, other_v2]
        st, pred = _post("/predict",
                         {"window": window, "scene_params": [v, 0.0, 0.0, 0.0]})
        out["predict_status"] = st
        out["predict_shape"] = pred.get("shape")
        out["predict_version"] = pred.get("version")
        raw = pred.get("prediction")
        # predict_next 保留 batch 维 -> shape [1,6]; 单条取 [0]
        nxt = raw[0] if (isinstance(raw, list) and len(raw) == 1
                         and isinstance(raw[0], list)) else raw
        out["next_state"] = nxt
        blind = out.get("blind_next_state")
        # 物理判据以"条件化相对盲预测显著修正且方向正确"为准 (5 万参数玩具
        # 模型绝对精度有限, 但场景门的修正在线可复现): 末帧 x=3.35, 期望 x=4.0。
        if isinstance(nxt, list) and len(nxt) == 6:
            last_x = window[-1][0]
            expected_x = last_x + v * dt
            cond_x_err = abs(nxt[0] - expected_x)
            cond_v_err = abs(nxt[3] - v)
            blind_x_err = abs(blind[0] - expected_x) if blind else None
            blind_v_err = abs(blind[3] - v) if blind else None
            out["physics_check"] = {
                "conditioned_x_advances": bool(nxt[0] > last_x),
                "blind_x_regresses": bool(blind and blind[0] < last_x),
                "conditioned_x_err": round(cond_x_err, 4),
                "conditioned_v_err": round(cond_v_err, 4),
                "blind_x_err": round(blind_x_err, 4) if blind_x_err is not None else None,
                "blind_v_err": round(blind_v_err, 4) if blind_v_err is not None else None,
                # 条件化位置/速度误差均显著小于盲预测, 且方向正确
                "gate_corrects": bool(
                    nxt[0] > last_x and blind and blind[0] < last_x
                    and cond_x_err < blind_x_err / 2
                    and cond_v_err < blind_v_err / 2),
            }
        pc = out.get("physics_check", {})
        ok = (out["health"]["status"] == "ok"
              and out["health"]["predictor_trained"] is True
              and st == 200 and pc.get("gate_corrects"))
        out["smoke_ok"] = bool(ok)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    path = os.path.join(HERE, "reports", "v555_service_smoke.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print("written:", path)
    return 0 if out.get("smoke_ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
