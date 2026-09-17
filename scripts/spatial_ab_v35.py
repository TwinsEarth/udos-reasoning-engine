"""
SFM 空间 A/B (v3.5.0.dev6) — 空间距离查询 vs affordance 打分 + 占据分辨率扫描
================================================================================
analogy, not reproduction。在合成场景上对比两条"找近处物体"路线:
    A. SpatialQueryEngine.range_search  —— 精确解析欧氏距离 (SFM 新路线);
    B. AffordanceScorer.score          —— 距离/可达性 softmax 打分 (既有路线)。

同时扫描 OccupancyGrid 分辨率 8/16/32 的占用耗时与体素数。
结论落 benchmarks/results/spatial_ab_v3.5.0.json; 收益不稳 => SFM 默认 opt-in 关。

用法: python3 scripts/spatial_ab_v35.py
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from udos.spatial import SpatialObject, SpatialScene  # noqa: E402
from udos.spatial_query import SpatialQueryEngine  # noqa: E402
from udos.occupancy import OccupancyGrid  # noqa: E402
from udos.affordance import AffordanceScorer  # noqa: E402

torch.set_num_threads(2)


def build_scene(n: int = 40, seed: int = 0) -> SpatialScene:
    rng = np.random.default_rng(seed)
    sc = SpatialScene()
    for i in range(n):
        sc.add(SpatialObject(f"o{i}", rng.uniform(-5, 5, 3),
                             radius=float(rng.uniform(0.1, 0.5))))
    return sc


def main():
    scene = build_scene()
    agent = np.array([0.0, 0.0, 0.0])

    # ---- A: SFM 精确距离查询 ---- #
    q = SpatialQueryEngine(scene)
    t0 = time.perf_counter()
    for _ in range(50):
        near = q.range_search(agent, radius=3.0)
    a_ms = (time.perf_counter() - t0) / 50 * 1000.0

    # ---- B: affordance 打分 (同 agent/物体代理) ---- #
    objs = scene.objects()
    # object_proxy [1, N, 6] = [pos3, vel3]
    proxy = np.zeros((1, len(objs), 6), dtype=np.float32)
    for i, o in enumerate(objs):
        proxy[0, i, :3] = o.position
    scorer = AffordanceScorer(reach_radius=3.0)
    state = np.zeros((1, 6), dtype=np.float32)
    state[0, :3] = agent
    t0 = time.perf_counter()
    for _ in range(50):
        res = scorer.score(state, proxy)
    b_ms = (time.perf_counter() - t0) / 50 * 1000.0

    # 语义一致性: SFM 认为的近处集合 vs affordance 给分最高的 k 个
    sfm_near = {e["object_id"] for e in near}
    topk = int((res["scores"][0] > 0).sum().item())
    aff_idx = res["reachable"][0].nonzero()[0].tolist()
    aff_ids = {objs[i].object_id for i in aff_idx}
    overlap = len(sfm_near & aff_ids)
    union = len(sfm_near | aff_ids)
    iou = overlap / union if union else 1.0

    # ---- 占据分辨率扫描 ---- #
    res_sweep = {}
    for res_n in (8, 16, 32):
        g = OccupancyGrid(bounds=((-5, -5, -5), (5, 5, 5)), resolution=res_n)
        t0 = time.perf_counter()
        for o in objs:
            g.occupy_object(o)
        dt = (time.perf_counter() - t0) * 1000.0
        res_sweep[str(res_n)] = {"occupied_voxels": g.n_occupied,
                                "total_voxels": len(g),
                                "occupy_ms": round(dt, 3)}

    summary = {
        "version": "3.5.0.dev6",
        "n_objects": len(scene),
        "route_A_spatial_range_search_ms": round(a_ms, 4),
        "route_B_affordance_score_ms": round(b_ms, 4),
        "semantic_iou_A_vs_B": round(iou, 4),
        "sfm_near_count": len(sfm_near),
        "affordance_reachable_count": len(aff_ids),
        "occupancy_resolution_sweep": res_sweep,
        "decision": "opt-in default off",
        "rationale": "SFM range_search 语义=中心距<=R; affordance=softmax 可达性打分, "
                     "二者语义不同 (IoU 非 1.0), 不互相替代; SFM 作为独立 opt-in 查询能力, "
                     "默认不接管 affordance 路径, 保持旧输出逐位等价。",
        "rejected_candidates": [
            {"candidate": "用 SFM range_search 替换 affordance_score",
             "reason": "语义不同 (精确距离 vs 归一化可达性打分), 替换会改变既有服务输出"},
            {"candidate": "占据网格分辨率默认 32",
             "reason": "32^3 占用与 SDF 成本约为 8^3 的 64 倍, 合成代理场景 16 已足够"},
        ],
    }
    os.makedirs("benchmarks/results", exist_ok=True)
    out = "benchmarks/results/spatial_ab_v3.5.0.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: summary[k] for k in
                      ["route_A_spatial_range_search_ms",
                       "route_B_affordance_score_ms",
                       "semantic_iou_A_vs_B", "decision"]},
                     ensure_ascii=False, indent=2))
    print("written:", out)


if __name__ == "__main__":
    main()
