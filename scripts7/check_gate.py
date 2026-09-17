#!/usr/bin/env python3
"""eval 门禁：读取 reports7/v7_verification.json，gating.all_pass 非真则退出 1。

CI 用法（先跑测试与综合验证）：
  python3 -m pytest tests7 -q
  python3 scripts7/verify_v7.py
  python3 scripts7/check_gate.py
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
f = REPO / "reports7" / "v7_verification.json"
if not f.exists():
    print("GATE FAIL: 缺少 v7_verification.json，请先运行 scripts7/verify_v7.py")
    sys.exit(1)
r = json.loads(f.read_text())
g = r.get("gating", {})
ok = bool(g.get("all_pass"))
print(f"gating.all_pass={ok} checked={g.get('checked')}")
for k, v in r.get("coverage", {}).items():
    print(f"  {k:22s} nominal={v['nominal']} empirical={v['marginal']} pass={v['pass']}")
pf = r.get("param_fan", {})
print(f"  param_fan              nominal={pf.get('nominal')} empirical={pf.get('empirical_coverage')} pass={pf.get('pass')}")
sys.exit(0 if ok else 1)
