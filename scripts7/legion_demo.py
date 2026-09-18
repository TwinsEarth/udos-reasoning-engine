"""Agent 军团演示（v7.2）：组织层级 + 协调器异步 fan-out。

PYTHONPATH=. python3 scripts7/legion_demo.py
展示：
1) build_org 以管理幅度 10 表示百万级军团的精确编制（对象有界、瞬间建完）；
2) CoordinatorAgent 把一个高阶预测目标异步分发给多位专家，验证/讨论/投票选优。
不实例化百万个在线 Agent（那需要 LLM key+云预算，属 AL4 门禁）。
"""
import asyncio
import json

from udos7.agents import CoordinatorAgent
from udos7.agents.automation import baseline_assessment
from udos7.agents.legion import build_org, level_headcounts
from udos7.dynamics import build_split
from udos7.persistence import load_worldmodel


async def main():
    org = build_org(1_000_000, span=10)
    print("== 军团精确编制（headcount=1,000,000, span=10）==")
    print(json.dumps(level_headcounts(org), ensure_ascii=False, indent=2))
    print("物化预览节点数:", org.node_count())

    model, _ = load_worldmodel("checkpoints7/worldmodel_v7.0.3.pt")
    ds = build_split(2026, n_traj_per_kind=1)
    window, truth = ds.X[0].tolist(), ds.Y[0].tolist()

    coord = CoordinatorAgent(project="legion-demo", model=model, concurrency=8)
    report = await coord.run("预测下一阶段轨迹", {
        "window": window, "truth": truth,
        "expert_modes": ["blind", "analytic"]})
    await coord.aclose()
    print("\n== 协调器异步选优 ==")
    print("winner:", report.winner)
    print("products:", [(p["author"], round(p["score"], 4))
                        for p in report.products])

    print("\n== 自动化分级（诚实口径）==")
    for a in baseline_assessment():
        print(f"  AL{int(a.al)}  {a.feature}  [{a.evidence}]")


if __name__ == "__main__":
    asyncio.run(main())
