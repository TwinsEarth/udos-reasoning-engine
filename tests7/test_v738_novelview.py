"""v7.3.8 新视角预测 / 空间上下文契约测试。"""
import math
import torch

from udos7.contracts import EvidenceGrade
from udos7.spatial import (PinholeCamera, Scene, Sphere, render_view,
                           fuse_views, novel_view_depth, depth_metrics,
                           project_points, unproject, run_novelview_benchmark)
from udos7.spatial.benchmark import demo_scene, ring_cameras


def test_camera_project_unproject_roundtrip():
    cam = PinholeCamera(eye=(3.0, -2.0, 1.0), look=(0.0, 0.0, 0.0))
    P = torch.tensor([[0.5, -0.3, 0.2], [-1.0, 0.8, 0.6]])
    pix, z = project_points(P, cam)
    assert (z > 0).all()
    back = unproject(pix[:, 0], pix[:, 1], z, cam)
    assert torch.allclose(back, P, atol=1e-4)


def test_render_spheres_visible():
    scene = demo_scene()
    cam = PinholeCamera(eye=(0.0, -4.0, 0.5), look=(0.0, 0.0, 0.0))
    depth, color, hit = render_view(cam, scene)
    assert hit.any() and depth[hit].min() > 0
    assert color[hit].max() > 0


def test_fusion_occupies_near_surface():
    scene = demo_scene()
    cams = ring_cameras(8)
    views = []
    for c in cams:
        d, _, m = render_view(c, scene)
        views.append((d, m, c))
    ctx = fuse_views(views, voxel=0.10)
    occ = ctx.occupied()
    assert occ.sum() > 50
    centers = ctx.centers[occ]
    # 至少一个真球心附近存在占据体素
    d = torch.cdist(centers, scene.centers).min(1).values
    assert d.min().item() < 0.25


def test_novel_view_at_training_pose_consistent():
    scene = demo_scene()
    cams = ring_cameras(8)
    views = []
    for c2 in cams:
        d2, _, m2 = render_view(c2, scene)
        views.append((d2, m2, c2))
    cam = cams[0]
    d, _, m = render_view(cam, scene)
    ctx = fuse_views(views, voxel=0.10)
    pd_, pm = novel_view_depth(cam, ctx)
    met = depth_metrics(pd_, pm, d, m)
    assert met["silhouette_iou"] > 0.8


def test_multiview_context_beats_single_view():
    r = run_novelview_benchmark()
    assert r["evidence_grade"] == EvidenceGrade.CPU_PROTO.value
    single, multi = r["results"]["single_view"], r["results"]["multi_view"]
    assert multi["silhouette_iou"] > single["silhouette_iou"]
    assert multi["coverage"] > 0.95 > single["coverage"]
    assert multi["depth_mae"] < single["depth_mae"]
    assert r["external_reference"]["grade"] == EvidenceGrade.UNVERIFIED.value


def test_metrics_perfect_match():
    d = torch.rand(8, 8) + 1.0
    m = torch.rand(8, 8) > 0.3
    met = depth_metrics(d, m, d, m)
    assert met["silhouette_iou"] == 1.0 and met["depth_mae"] == 0.0


def test_benchmark_determinism():
    a = run_novelview_benchmark()
    b = run_novelview_benchmark()
    assert a["results"] == b["results"]
