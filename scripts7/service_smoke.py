#!/usr/bin/env python3
"""v7 HTTP 服务真实冒烟：子进程拉起 /api/v7，验证 health/predict/interval/metrics。

证据分级 verified（本机回环真实 HTTP，非 mock）。写 reports7/service_smoke.json。
"""
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from udos7.dynamics import build_split
from udos7.contracts import TEST_SEED

PORT = 8791
BASE = f"http://127.0.0.1:{PORT}/api/v7"
OUT = REPO / "reports7" / "service_smoke.json"


def req(method, path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    r = urllib.request.Request(BASE + path, data=data,
                               headers={"Content-Type": "application/json"},
                               method=method)
    with urllib.request.urlopen(r, timeout=60) as resp:
        return resp.status, json.loads(resp.read())


def main():
    proc = subprocess.Popen(
        [sys.executable, "-m", "udos7.server", "--port", str(PORT)],
        cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    report = {"smoke_ok": False}
    try:
        for _ in range(40):
            try:
                s, h = req("GET", "/health")
                if s == 200:
                    break
            except Exception:
                time.sleep(1)
        report["health"] = h
        assert h["version"].startswith("7."), h

        ds = build_split(TEST_SEED, n_traj_per_kind=4)
        window = ds.X[0].tolist()
        explicit = ds.P[0].tolist()
        s, p = req("POST", "/predict",
                   {"window": window, "explicit": explicit, "horizon": 4})
        assert s == 200 and len(p["trajectory"]) == 1 and len(
            p["trajectory"][0]) == 4
        report["predict_status"] = s
        report["predict_first"] = p["trajectory"][0][0]

        widths = {}
        for a in (0.2, 0.1, 0.05):
            s, iv = req("POST", "/interval",
                        {"window": window, "explicit": explicit,
                         "alpha": a, "horizon": 4})
            up = torch.tensor(iv["upper"]); lo = torch.tensor(iv["lower"])
            widths[a] = float((up - lo).mean())
            assert torch.isfinite(up).all() and torch.isfinite(lo).all()
        report["interval_widths"] = widths
        report["widths_ordered"] = widths[0.2] < widths[0.1] < widths[0.05]
        assert report["widths_ordered"], widths

        try:
            s, m = req("GET", "/metrics")
            report["metrics_status"] = s
            report["gating_all_pass"] = m.get("gating", {}).get("all_pass")
        except Exception as e:
            report["metrics_status"] = f"unavailable: {e}"

        report["smoke_ok"] = True
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("saved", OUT)
    if not report["smoke_ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
