"""v7.5.0 百万级 Agent 矩阵演示（CPU，零第三方依赖）。

产物：reports7/matrix_scale_demo.json
用法：python scripts7/matrix_demo.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from udos7.topology.matrix import run_mission, MatrixConfig          # noqa: E402
from udos7.topology.base import standard_workload                     # noqa: E402
from udos7.observability import Tracer                                # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "reports7")


def main() -> int:
    tr = Tracer()
    with tr.span("topology.matrix", version="v7.5.0"):
        clean = run_mission(standard_workload(24, seed=11),
                            MatrixConfig(), seed=1)
        faults = {"spec-kinematics-0": "crash",
                  "spec-spatial-1": "drop_context",
                  "spec-dataflywheel-2": "byzantine"}
        stressed = run_mission(standard_workload(24, seed=11),
                               MatrixConfig(), faulty_workers=faults,
                               seed=1)
    scale = clean["scale"]

    print("== UDOS v7.5.0 多智能体矩阵（证据 cpu-proto）==")
    print(f"干净舰队：验收 {clean['accepted']}/{clean['n_orders']}，"
          f"治理 ok={clean['governance']['ok']}，"
          f"市场守恒={clean['market']['conserved']}，"
          f"Trace 完整={clean['trace_verified']}")
    print(f"故障注入：验收 {stressed['accepted']}/{stressed['n_orders']}，"
          f"隔离 {stressed['isolated_agents']}，改派返工 "
          f"{stressed['retries']} 次，治理 ok={stressed['governance']['ok']}")
    print("\n规模基准（b=8；显式/闭式外推分级）：")
    print("  目标Agent   实际Agent   深度  星型中心扇入  分层最大扇入  并行轮次(星/分)  证据")
    for r in scale:
        print(f"  {r['target_agents']:>9}  {r['actual_agents']:>9}  "
              f"{r['depth']:>3}  {r['star_center_fanin']:>11}  "
              f"{r['layered_max_fanin']:>9}  "
              f"{r['star_rounds']:>7}/{r['layered_rounds']:<5}  {r['grade']}")
    million = scale[-1]
    print(f"\n百万级：星型中心承接 {million['star_center_fanin']:,} 条消息、"
          f"串行 {million['star_rounds']:,} 轮；分层每节点扇入恒 "
          f"{million['layered_max_fanin']}、{million['layered_rounds']} 轮收敛。")
    print("\n已知限制：Agent 为确定性处理器模拟，非真实 LLM 进程；"
          "analytical 行由小规模显式模拟验证过的闭式计数外推；"
          "真实分布式部署、网络故障与安全边界属未达闸门（unverified）。")

    rep = {"evidence_grade": "cpu-proto", "clean": clean, "stressed": stressed}
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "matrix_scale_demo.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2, default=str)
    tr.export_jsonl(os.path.join(OUT, "matrix_scale_demo.traces.jsonl"))
    print(f"\n已写出：{os.path.abspath(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
