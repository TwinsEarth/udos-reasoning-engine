"""v7.4.0 第一人称经验数据飞轮演示（CPU，零第三方依赖）。

产物：reports7/egodata_flywheel_demo.json
用法：python scripts7/egodata_flywheel_demo.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from udos7.egodata.benchmark import run_flywheel_benchmark  # noqa: E402
from udos7.observability import Tracer  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "reports7")


def main() -> int:
    tr = Tracer()
    with tr.span("egodata.flywheel", version="v7.4.0"):
        rep = run_flywheel_benchmark(pool_size=160, budget=48, K=12)
    p = rep["passive_collection"]
    a = rep["coverage_aware_collection"]
    ab = rep["ablation_greedy_on_skewed_pool"]
    print("== UDOS v7.4.0 第一人称经验数据飞轮（证据 cpu-proto）==")
    print(f"候选池 {rep['config']['pool_size']} 片段，预算 "
          f"{rep['config']['budget_raw_minutes']} 原始分钟，注入缺陷率 25%")
    print(f"\n                 Yield良率  状态单元  占oracle  冗余率  经验密度")
    print(f"被动偏态采集      {p['yield']:.3f}     {p['state_cells']:>4}    "
          f"{p['oracle_fraction']:.3f}    {p['redundancy_rate']:.3f}   {p['mean_density']}")
    print(f"Coverage-aware    {a['yield']:.3f}     {a['state_cells']:>4}    "
          f"{a['oracle_fraction']:.3f}    {a['redundancy_rate']:.3f}   {a['mean_density']}")
    print(f"消融(同偏态池贪心)  {ab['yield']:.3f}     {ab['state_cells']:>4}    "
          f"{ab['oracle_fraction']:.3f}")
    print(f"\noracle 状态单元（全池合格片段）：{rep['oracle_state_cells']}")
    print("被动拒绝原因：", p["reject_reasons"])
    print("主动拒绝原因：", a["reject_reasons"])
    late_p = sum(p["marginal_curve"][-12:]) / 12
    late_a = sum(a["marginal_curve"][-12:]) / 12
    print(f"末段平均边际新状态：被动 {late_p:.1f} / 主动 {late_a:.1f}")
    print("\n外部参照（unverified，不与本仓互证）：")
    for c in rep["external_reference"]["claims_unverified"][:3]:
        print("  -", c)
    print("\n已知限制：", "；".join(rep["known_limits"]))
    print("闸门：", "；".join(rep["gates"]))
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "egodata_flywheel_demo.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    tr.export_jsonl(os.path.join(OUT, "egodata_flywheel_demo.traces.jsonl"))
    print(f"\n已写出：{os.path.abspath(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
