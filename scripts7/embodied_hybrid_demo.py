"""v7.3.7 具身混合控制端到端演示（零第三方依赖，CPU 可跑）。

三模式闭环对比：direct（语义层直控）/ motion（仅动作先验）/ hybrid（先验 TopK
候选 + 语义裁决/接管）。产物写入 reports7/embodied_hybrid_demo.json。

用法：python scripts7/embodied_hybrid_demo.py [每任务回合数，默认10]
"""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from udos7.embodied import run_suite  # noqa: E402
from udos7.observability import Tracer  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "reports7")


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    tr = Tracer()
    rep = run_suite(episodes_per_task=n, tracer=tr)

    print("== UDOS v7.3.7 具身混合控制基准（CPU 合成动力学，证据等级 cpu-proto）==")
    print(f"每任务 {n} 回合 × 3 模式 × 4 任务；K={rep['config']['K']}，"
          f"段长 {rep['config']['seg_len']}，接管段长 {rep['config']['override_len']}\n")
    print(f"{'任务':<16}{'模式':<8}{'成功率':>8}{'平均分':>9}{'接管率':>8}{'碰撞':>6}")
    for task, mm in rep["by_task"].items():
        for mode in ("direct", "motion", "hybrid"):
            x = mm[mode]
            print(f"{task:<16}{mode:<8}{x['success_rate']:>8.2f}"
                  f"{x['mean_score']:>9.1f}{x['intervention_rate']:>8.2f}"
                  f"{'-':>6}")
    print("\n== 汇总 ==")
    for mode, a in rep["aggregate"].items():
        print(f"{mode:<8} 成功率 {a['success_rate']:.2f}  平均分 {a['mean_score']:.1f}"
              f"  均碰撞 {a['mean_collisions']:.2f}  接管率 {a['intervention_rate']:.2f}"
              f"  Token代理 in/out {a['tokens_in_proxy_total']}/"
              f"{a['tokens_out_proxy_total']}")
    ext = rep["external_reference"]["roboDojo"]
    print("\n== 外部参照（UNVERIFIED，非本仓实测，不可直接比较）==")
    print(f"报告口径：direct {ext['direct']['success_rate']*100:.0f}%/"
          f"{ext['direct']['mean_score']} 分；hybrid {ext['hybrid']['success_rate']*100:.0f}%/"
          f"{ext['hybrid']['mean_score']} 分、接管 {ext['hybrid']['intervention_share']*100:.1f}%")

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "embodied_hybrid_demo.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    tr.export_jsonl(os.path.join(OUT, "embodied_hybrid_demo.traces.jsonl"))
    print(f"\n已写出：{os.path.abspath(path)}（含 raw_rows、external_reference、gates）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
