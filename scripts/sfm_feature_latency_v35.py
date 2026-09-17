"""
SFM 特征延迟基准 (v3.5.2) — 落 benchmarks/results/feature_latency_v3.5.0.json
========================================================================
在合成场景上测各 SFM 模块单次查询延迟 (CPU, torch 2 线程), 与 v3.4 线同口径。
analogy, not reproduction。

用法: python3 scripts/sfm_feature_latency_v35.py
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from udos.spatial import (SpatialObject, SpatialScene, SpatialTransform,  # noqa: E402
                          OrthographicView)
from udos.scene_graph import SceneGraph  # noqa: E402
from udos.occupancy import OccupancyGrid, DistanceField  # noqa: E402
from udos.collision import CollisionDetector, NearestNeighbor  # noqa: E402
from udos.spatial_query import SpatialQueryEngine  # noqa: E402


def build_scene(n: int = 50):
    rng = np.random.default_rng(7)
    sc = SpatialScene()
    for i in range(n):
        sc.add(SpatialObject(f"o{i}", rng.uniform(-5, 5, 3),
                             radius=float(rng.uniform(0.1, 0.5))))
    return sc


def bench(fn, iters=200):
    fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return round((time.perf_counter() - t0) / iters * 1000.0, 4)


def main():
    sc = build_scene()
    q = SpatialQueryEngine(sc)
    cd = CollisionDetector()
    nn = NearestNeighbor(sc)
    sg = SceneGraph(sc, near_dist=1.5)
    occ = OccupancyGrid(bounds=((-6, -6, -6), (6, 6, 6)), resolution=16)
    for o in sc.objects()[:10]:
        occ.occupy_object(o)
    df = DistanceField(occ)
    view = OrthographicView(eye=[8.0, 0.0, 3.0], look_at=[0, 0, 0])
    pts = np.array([[0.0, 0.0, 0.0], [1.0, -1.0, 0.5], [-2.0, 2.0, 0.0]])

    lat = {
        "spatial_pairwise_distances_ms": bench(lambda: sc.pairwise_distances()),
        "scene_graph_relate_ms": bench(lambda: sg.relate("o0", "o1")),
        "occupancy_query_ms": bench(lambda: occ.is_occupied([0.1, 0.1, 0.1])),
        "distance_field_at_ms": bench(lambda: df.at([0.1, 0.1, 0.1])),
        "collision_detect_ms": bench(lambda: cd.detect_contacts(sc)),
        "nearest_neighbor_ms": bench(lambda: nn.query([0, 0, 0], k=3)),
        "spatial_range_search_ms": bench(lambda: q.range_search([0, 0, 0], 2.0)),
        "spatial_raycast_ms": bench(lambda: q.raycast([-8, 0, 0], [1, 0, 0])),
        "orthographic_project_ms": bench(lambda: view.project(pts)),
        "transform_apply_ms": bench(
            lambda: SpatialTransform.rotation_from_axis_angle(
                [0, 0, 1], 0.3).apply(pts)),
    }
    summary = {
        "feature": "sfm_component_latency",
        "version": "3.5.2",
        "n_objects": len(sc),
        "cpu_threads": 2,
        "latency_ms": lat,
    }
    os.makedirs("benchmarks/results", exist_ok=True)
    out = "benchmarks/results/feature_latency_v3.5.0.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("written:", out)


if __name__ == "__main__":
    main()
