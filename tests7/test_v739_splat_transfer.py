"""v7.3.9 高斯泼溅 + Real-to-Sim/Sim-to-Real 契约测试。"""
import math
import torch

from udos7.contracts import EvidenceGrade
from udos7.spatial.camera import PinholeCamera, project_points
from udos7.spatial.scene import Scene, Sphere, render_view
from udos7.spatial.splat import GaussianSplat, render_splat, fit_splat, splat_loss
from udos7.spatial.benchmark import demo_scene, ring_cameras
from udos7.spatial.transfer import (noisy_observation, real2sim_task,
                                    sim2real_gap, run_transfer_benchmark,
                                    run_split_episode)
from udos7.embodied.env import EnvTask


def test_single_gaussian_projects_near_center():
    cam = PinholeCamera(width=32, height=24, eye=(0.0, -3.0, 0.0),
                        look=(0.0, 0.0, 0.0))
    sp = GaussianSplat(means=torch.tensor([[0.0, 0.0, 0.0]]),
                       scales=torch.tensor([[0.4, 0.4, 0.4]]),
                       colors=torch.tensor([[1.0, 1.0, 1.0]]),
                       opacity=torch.tensor([0.99]))
    d, color, mask = render_splat(cam, sp)
    pix, z = project_points(sp.means, cam)
    px, py = int(round(pix[0, 0].item())), int(round(pix[0, 1].item()))
    assert mask[py, px] and color[py, px].sum() > 1.5


def test_fit_reduces_training_loss():
    scene = demo_scene()
    cams = ring_cameras(4, width=36, height=28)
    sp0 = fit_splat(scene, cams, n_gaussians=80, iters=1, seed=0)[0]
    sp1, _ = fit_splat(scene, cams, n_gaussians=80, iters=20, seed=0)
    l0 = float(splat_loss(sp0.clone(), cams, scene)[0])
    l1 = float(splat_loss(sp1.clone(), cams, scene)[0])
    assert l1 < l0


def test_splat_novelview_carries_geometry():
    scene = demo_scene()
    train = ring_cameras(6, width=36, height=28)
    novel = ring_cameras(3, width=36, height=28, phase=math.pi / 4)
    sp, _ = fit_splat(scene, train, n_gaussians=150, iters=40, seed=0)
    from udos7.spatial.splat import splat_novelview_report
    rep = splat_novelview_report(sp, novel, scene)["aggregate"]
    assert rep["silhouette_iou"] > 0.55 and rep["coverage"] > 0.9


def test_noisy_depth_deterministic_and_lossy():
    scene = demo_scene()
    d, _, m = render_view(ring_cameras(6)[0], scene)
    n1, m1 = noisy_observation(d, m, depth_sigma=0.03, seed=42)
    n2, m2 = noisy_observation(d, m, depth_sigma=0.03, seed=42)
    assert torch.equal(n1, n2) and torch.equal(m1, m2)
    assert (m1.sum() <= m.sum())
    assert (n1[m1] - d[m1]).abs().mean() > 0.01


def test_real2sim_recovers_plane_obstacles():
    scene = demo_scene()
    task, clusters = real2sim_task(scene, ring_cameras(10, width=48, height=36))
    # 真值中与 z=0 平面相交的球为 2 个（另两个悬在平面外）
    assert len(clusters) == 2
    assert len(task.obstacles) == 1 and task.contact_terminal


def test_real2sim_controller_succeeds():
    from udos7.embodied.hybrid import run_episode
    scene = demo_scene()
    task, _ = real2sim_task(scene, ring_cameras(10, width=48, height=36))
    succ = sum(run_episode(task, "hybrid", seed=1000 + k)["success"]
               for k in range(8))
    assert succ >= 7


def test_sim2real_robust_margin_reduces_collisions():
    g = sim2real_gap((([0.0, 0.0, 0.0], 0.55),), n_worlds=8)
    assert g["robust"]["mean_collisions"] < g["naive"]["mean_collisions"]
    assert g["robust"]["success_rate"] >= g["naive"]["success_rate"]
    assert "cpu-proto" in g["evidence"]


def test_transfer_benchmark_deterministic_and_graded():
    a = run_transfer_benchmark(episodes=4)
    b = run_transfer_benchmark(episodes=4)
    assert a["evidence_grade"] == EvidenceGrade.CPU_PROTO.value
    assert a["real_to_sim"]["success_rate"] == b["real_to_sim"]["success_rate"]
    assert a["sim_to_real"]["robust"]["success_rate"] >= \
        a["sim_to_real"]["naive"]["success_rate"]
