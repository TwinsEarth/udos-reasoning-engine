#!/usr/bin/env python3
"""构建期: 把 docs/opensource_catalog.json 编译为运行时内嵌的 Python 模块。
运行时不读 JSON 文件; 本脚本只在发版前手动/CI 跑一次。"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "opensource_catalog.json"
OUT = ROOT / "udos" / "connectors" / "_catalog_data.py"

def slugify(name: str) -> str:
    s = name.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s


def parse_priority(fit) -> str:
    """稳健解析 udoS_fit 优先级: 兼容 dict 与 str 两种形态。
    dict: 取 udoS_fit['priority']; str: 取 '高——/中——/低...' 前缀首字。"""
    if isinstance(fit, dict):
        return (fit.get("priority") or "").strip()
    m = re.match(r"^\s*(高|中|低)", str(fit))
    return m.group(1) if m else ""


raw = json.loads(SRC.read_text(encoding="utf-8"))
seen, rows = set(), []
for i, e in enumerate(raw):
    sl = slugify(e["name"]) or f"res_{i}"
    base, k = sl, 2
    while sl in seen:
        sl = f"{base}_{k}"; k += 1
    seen.add(sl)
    pri = parse_priority(e["udos_fit"])
    rows.append({
        "id": sl,
        "name": e["name"],
        "org": e["org"],
        "category": e["category"],
        "source_class": e["source_class"],   # 模型库/数据库/动作库 -> kind
        "license": e["license"],
        "scale": e["scale"],
        "oss_triangle": e["oss_triangle"],
        "udos_fit": e["udos_fit"],
        "cpu_feasibility": e["cpu_feasibility"],
        "source_url": e["source_url"],
        "evidence": e["evidence"],
        "priority": "high" if pri == "高" else "normal",
        "priority_raw": pri,
    })

header = ('# 本文件由 scripts/build_catalog_data.py 从 docs/opensource_catalog.json 自动生成。\n'
          '# 运行时不读取 JSON; 改 catalog 后请重跑该脚本。禁止手改。\n'
          'CATALOG_ROWS = ')
OUT.write_text(header + repr(rows) + "\n", encoding="utf-8")
print(f"wrote {OUT} with {len(rows)} rows")
from collections import Counter
print(Counter(r["priority"] for r in rows), Counter(r["source_class"] for r in rows))
