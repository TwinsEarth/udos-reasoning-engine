"""v7.3.9 高斯泼溅 + Real-to-Sim/Sim-to-Real 演示（CPU，零第三方依赖）。

产物：reports7/splat_transfer_demo.json
用法：python scripts7/splat_transfer_demo.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from udos7.spatial.transfer import run_transfer_benchmark  # noqa: E402
from udos7.observability import Tracer  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "reports7")


def main() -> int:
    tr = Tracer()
    with tr.span("spatial.splat_transfer", version="v7.3.9"):
        rep = run_transfer_benchmark()
    print("== UDOS v7.3.9 高斯泼溅 + Real/Sim-to-Real（证据 cpu-proto）==")
    gs = rep["gaussian_splat"]["novel_view"]
    print(f"\n[3DGS-lite] {rep['gaussian_splat']['n_gaussians']} 高斯 × "
          f"{rep['gaussian_splat']['iters']} 步，新视角："
          f"IoU {gs['silhouette_iou']:.3f}，覆盖率 {gs['coverage']:.3f}，"
          f"深度MAE {gs['depth_mae']:.3f}")
    r2s = rep["real_to_sim"]
    print(f"\n[Real-to-Sim] 带噪多视角重建 → 平面障碍簇 "
          f"{r2s['plane_intersecting_clusters']}（真值 {r2s['true_plane_obstacles']}），"
          f"规划障碍 {r2s['planning_obstacle']}")
    print(f"             重建仿真中混合控制成功率：{r2s['success_rate']*100:.0f}%")
    s2r = rep["sim_to_real"]
    print(f"\n[Sim-to-Real] 有偏估计 + 扰动真实代理世界：")
    print(f"  朴素规划        成功率 {s2r['naive']['success_rate']:.2f}，"
          f"均碰撞 {s2r['naive']['mean_collisions']:.2f}")
    print(f"  裕量 {s2r['margin']} 鲁棒规划 成功率 {s2r['robust']['success_rate']:.2f}，"
          f"均碰撞 {s2r['robust']['mean_collisions']:.2f}")
    print("\n已知限制：", "；".join(rep["known_limits"]))
    print("闸门：", "；".join(rep["gates"]))
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "splat_transfer_demo.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    tr.export_jsonl(os.path.join(OUT, "splat_transfer_demo.traces.jsonl"))
    print(f"\n已写出：{os.path.abspath(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
