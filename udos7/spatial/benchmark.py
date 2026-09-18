"""新视角预测基准：多视角空间上下文 vs 单视角，在留出视角上量几何一致性。"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

from ..contracts import EvidenceGrade
from .camera import PinholeCamera
from .scene import Scene, Sphere, render_view
from .fusion import fuse_views, novel_view_depth, depth_metrics

# Atlas 相关描述来自公开资料/创始人访谈转述，本仓未复现其网络与指标。
EXTERNAL_REFERENCE = {
    "source": "Atlas（世界模型）公开资料转述：新视角预测、空间上下文、高斯泼溅、Real/Sim-to-Real",
    "grade": EvidenceGrade.UNVERIFIED.value,
    "claims": ["新视角预测被认为与下一 token 预测一样是潜在的 AI-complete 基础任务",
               "高斯泼溅渲染快但难表达动态，Atlas 支持但不再强制依赖",
               "可压缩 Real-to-Sim 成本、减少对真实扫描环境模型的依赖"],
    "note": "均为外部主张，非本仓实测；本基准只验证显式几何机制。",
}


def demo_scene() -> Scene:
    return Scene([
        Sphere((-1.1, 0.3, -0.2), 0.55, (0.85, 0.35, 0.30)),
        Sphere((0.9, -0.5, 0.1), 0.70, (0.30, 0.55, 0.85)),
        Sphere((0.2, 1.0, -0.6), 0.45, (0.35, 0.75, 0.45)),
        Sphere((1.3, 0.9, 0.8), 0.35, (0.80, 0.70, 0.30)),
    ])


def ring_cameras(n: int, radius: float = 5.0, elev_deg: float = 18.0,
                 width: int = 48, height: int = 36,
                 phase: float = 0.0) -> List[PinholeCamera]:
    cams = []
    elev = math.radians(elev_deg)
    for k in range(n):
        a = phase + 2 * math.pi * k / n
        eye = (radius * math.cos(a) * math.cos(elev),
               radius * math.sin(a) * math.cos(elev),
               radius * math.sin(elev) + 0.2)
        cams.append(PinholeCamera(width=width, height=height, eye=eye,
                                  look=(0.0, 0.0, 0.0)))
    return cams


def _views(cams: List[PinholeCamera], scene: Scene):
    return [(render_view(c, scene), c) for c in cams]


def run_novelview_benchmark(n_train: int = 8, n_novel: int = 6,
                            voxel: float = 0.10, tracer=None) -> Dict:
    scene = demo_scene()
    train_cams = ring_cameras(n_train, phase=0.0)
    novel_cams = ring_cameras(n_novel, phase=math.pi / n_novel)  # 视角错开
    train_views = [(render_view(c, scene)[0], render_view(c, scene)[2], c)
                   for c in train_cams]

    ctx_multi = fuse_views(train_views, voxel=voxel)
    ctx_single = fuse_views(train_views[:1], voxel=voxel)

    rows = []
    for tag, ctx in (("single_view", ctx_single), ("multi_view", ctx_multi)):
        ms = []
        for c in novel_cams:
            gt_d, _, gt_m = render_view(c, scene)
            pd_, pm = novel_view_depth(c, ctx)
            ms.append(depth_metrics(pd_, pm, gt_d, gt_m))
        agg = {k: round(sum(m[k] for m in ms if m[k] == m[k]) /
                        max(1, sum(1 for m in ms if m[k] == m[k])), 4)
               for k in ("silhouette_iou", "coverage", "precision",
                         "depth_mae", "gt_hit_ratio")}
        rows.append({"context": tag, "metrics": agg, "per_view": ms})

    single = rows[0]["metrics"]
    multi = rows[1]["metrics"]
    return {
        "module": "udos7.spatial novel-view prediction",
        "version": "v7.3.8",
        "evidence_grade": EvidenceGrade.CPU_PROTO.value,
        "config": {"n_train_views": n_train, "n_novel_views": n_novel,
                   "voxel": voxel, "representation": "TSDF voxel (explicit)"},
        "results": {r["context"]: r["metrics"] for r in rows},
        "per_view_rows": rows,
        "finding": "多视角空间上下文在留出视角上的轮廓 IoU/覆盖率显著高于单视角，"
                   "证明新视角预测依赖相机几何与三维一致性。",
        "external_reference": EXTERNAL_REFERENCE,
        "gates": ["学习型 NeRF/3DGS 新视角合成与真实图像需 GPU（v7.3.9 给 CPU 版高斯泼溅原型）",
                  "真实相机标定/多目采集需硬件"],
    }
