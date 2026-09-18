"""Real-to-Sim / Sim-to-Real 闭环（CPU 代理）。

- Real-to-Sim：给“真实观测”加深度噪声/丢点与位姿抖动，TSDF 重建后抽取障碍球，
  生成 v7.3.7 具身控制器可用的导航任务；
- Sim-to-Real：规划器只拿**有偏估计**，在带扰动的“真实代理”世界执行，比较
  朴素规划与域随机化安全裕量（半径膨胀）下的碰撞/成功率差距。
无真机、无 MuJoCo：真实世界用固定 seed 的扰动合成世界代理，证据 cpu-proto。
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import replace
from typing import Dict, List, Sequence, Tuple

import torch

from .camera import PinholeCamera
from .scene import Scene, render_view
from .fusion import fuse_views
from ..embodied.env import EnvTask, PointMassEnv
from ..embodied.hybrid import Candidate, MotionPrior, SemanticCritic


def noisy_observation(depth: torch.Tensor, hitmask: torch.Tensor,
                      depth_sigma: float = 0.02, dropout: float = 0.05,
                      seed: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """模拟真实深度相机：高斯噪声 + 随机丢点。"""
    g = torch.Generator().manual_seed(seed)
    d = depth.clone()
    m = hitmask.clone()
    n = d[m]
    d[m] = (n + torch.randn(n.shape, generator=g) * depth_sigma).clamp_min(0.05)
    drop = torch.rand(m.shape, generator=g) < dropout
    m = m & ~drop
    return d, m


def jitter_camera(cam: PinholeCamera, sigma: float = 0.03, seed: int = 0
                  ) -> PinholeCamera:
    g = torch.Generator().manual_seed(seed)
    eye = tuple(torch.tensor(cam.eye) + torch.randn(3, generator=g) * sigma)
    return PinholeCamera(width=cam.width, height=cam.height, fov_deg=cam.fov_deg,
                         eye=eye, look=cam.look, up=cam.up)


def extract_obstacle_spheres(ctx, min_voxels: int = 12) -> List[Dict]:
    """对占据体素做 6-连通 BFS，每簇给包围球（中心、半径、体素数）。"""
    occ3d = ctx.occupied().view(ctx.nx, ctx.ny, ctx.nz).numpy()
    seen = torch.zeros(occ3d.shape, dtype=torch.bool)
    clusters = []
    for ix in range(ctx.nx):
        for iy in range(ctx.ny):
            for iz in range(ctx.nz):
                if not occ3d[ix, iy, iz] or seen[ix, iy, iz]:
                    continue
                q = deque([(ix, iy, iz)]); seen[ix, iy, iz] = True
                vox = []
                while q:
                    x, y, z = q.popleft(); vox.append((x, y, z))
                    for dx, dy, dz in ((1, 0, 0), (-1, 0, 0), (0, 1, 0),
                                       (0, -1, 0), (0, 0, 1), (0, 0, -1)):
                        a, b, c = x + dx, y + dy, z + dz
                        if 0 <= a < ctx.nx and 0 <= b < ctx.ny and 0 <= c < ctx.nz \
                                and occ3d[a, b, c] and not seen[a, b, c]:
                            seen[a, b, c] = True; q.append((a, b, c))
                if len(vox) >= min_voxels:
                    clusters.append(vox)
    out = []
    for vox in clusters:
        idx = [v[0] * ctx.ny * ctx.nz + v[1] * ctx.nz + v[2] for v in vox]
        pts = ctx.centers[torch.tensor(idx)]
        c = pts.mean(0)
        # TSDF 占据带比真值表面厚约 1 个体素：在合成真值上标定后减去（cpu-proto
        # 校准量；真实传感器需按相机重新标定，见 docs 闸门）。
        r3d = torch.sqrt(((pts - c) ** 2).sum(1)).max().item() - ctx.voxel
        # 机器人在 z=0 平面：只保留与该平面相交的簇，半径取球面在 z=0 的截圆
        if r3d < abs(c[2].item()):
            continue
        r_xy = math.sqrt(max(r3d ** 2 - c[2].item() ** 2, 0.0))
        out.append({"center": c.tolist(), "radius": round(max(r_xy, 0.25), 3),
                    "radius3d": round(r3d, 3), "voxels": len(vox)})
    return out


def real2sim_task(scene: Scene, cams: Sequence[PinholeCamera],
                  depth_sigma: float = 0.005, pose_sigma: float = 0.0,
                  voxel: float = 0.10, seed: int = 0,
                  name: str = "reconstructed_gate") -> Tuple[EnvTask, List[Dict]]:
    """带噪“真实观测”→ TSDF 重建 → 障碍球 → 具身导航任务。

    当前全局规划器只覆盖**单主障碍**过门（取平面截圆最大者）；多障碍连续绕行
    需全局航点规划，列为已知限制（见 docs）。
    """
    views = []
    for k, c in enumerate(cams):
        d, _, m = render_view(c, scene)
        d, m = noisy_observation(d, m, depth_sigma=depth_sigma, seed=seed + k)
        cj = jitter_camera(c, sigma=pose_sigma, seed=seed + 100 + k)
        views.append((d, m, cj))
    ctx = fuse_views(views, voxel=voxel)
    spheres = extract_obstacle_spheres(ctx)
    corridor = [s for s in spheres if abs(s["center"][0]) < 1.8]
    obstacles = ()
    if corridor:
        big = max(corridor, key=lambda s: s["radius"])
        obstacles = (([big["center"][0], big["center"][1], 0.0],
                      max(big["radius"], 0.25)),)
    task = EnvTask(name=name, start=(-2.2, 0.0, 0.0),
                   waypoints=[(2.2, 0.0, 0.0)],
                   obstacles=obstacles, contact_terminal=True, max_steps=80)
    return task, spheres


def run_split_episode(plan_task: EnvTask, exec_task: EnvTask, seed: int = 0,
                      K: int = 32) -> Dict:
    """规划用 plan_task（有偏估计），执行在 exec_task（真实代理），模型失配闭环。"""
    plan = PointMassEnv(plan_task); plan.reset()
    exe = PointMassEnv(exec_task); exe.reset()
    prior = MotionPrior(K=K, seed=seed)
    critic = SemanticCritic()
    decisions = 0
    while not exe.done:
        tgt = critic._active_target(plan)
        cands = prior.propose(plan, tgt)
        cands.append(Candidate("override", critic.override_program(plan, tgt)))
        chosen = critic.decide(plan, cands)
        decisions += 1
        for acc in chosen.accs:
            exe.step(acc)                      # 真实世界执行（含失配）
            if not plan.done:
                plan.step(acc)                 # 规划模型同步前推
            if exe.done:
                break
    return {"success": bool(exe.success), "collisions": exe.collisions,
            "steps": exe.steps, "decisions": decisions}


def sim2real_gap(true_obstacles: Sequence, n_worlds: int = 8, seed0: int = 2000,
                 est_pos_sigma: float = 0.12, real_pos_sigma: float = 0.10,
                 margin: float = 0.22, K: int = 32) -> Dict:
    """比较朴素规划 vs 半径膨胀（域随机化安全裕量）在真实代理世界的表现。"""
    def make_task(obs, name, terminal=True):
        return EnvTask(name=name, start=(-2.2, 0.0, 0.0),
                       waypoints=[(2.2, 0.0, 0.0)],
                       obstacles=tuple((o[0], o[1]) for o in obs),
                       contact_terminal=terminal, max_steps=80)

    g = torch.Generator().manual_seed(77)
    est = [([c[0] + (torch.randn(1, generator=g) * est_pos_sigma).item(),
             c[1] + (torch.randn(1, generator=g) * est_pos_sigma).item(), 0.0],
            r * 0.9) for c, r in true_obstacles]
    # 规划任务不因“估计碰撞”提前终止（估计本就有偏），只在真实代理世界判终止
    naive_plan = make_task(est, "naive_plan", terminal=False)
    robust_est = [(c, r + margin) for c, r in est]
    robust_plan = make_task(robust_est, "robust_plan", terminal=False)

    rows = {"naive": [], "robust": []}
    for k in range(n_worlds):
        gg = torch.Generator().manual_seed(seed0 + k)
        real = [([c[0] + (torch.randn(1, generator=gg) * real_pos_sigma).item(),
                  c[1] + (torch.randn(1, generator=gg) * real_pos_sigma).item(), 0.0],
                 r + (torch.randn(1, generator=gg) * 0.04).item())
                for c, r in true_obstacles]
        real_task = make_task(real, f"real_{k}")
        rows["naive"].append(run_split_episode(naive_plan, real_task, seed=seed0 + k, K=K))
        rows["robust"].append(run_split_episode(robust_plan, real_task,
                                                seed=seed0 + k, K=K))

    def agg(rs):
        return {"success_rate": round(sum(r["success"] for r in rs) / len(rs), 3),
                "mean_collisions": round(sum(r["collisions"] for r in rs) / len(rs), 3)}
    return {"naive": agg(rows["naive"]), "robust": agg(rows["robust"]),
            "raw": rows, "margin": margin,
            "evidence": "cpu-proto（真实世界=固定 seed 扰动合成代理，非真机）"}


def run_transfer_benchmark(episodes: int = 8, seed: int = 1000) -> Dict:
    """v7.3.9 综合：高斯泼溅新视角 + Real-to-Sim 重建控制 + Sim-to-Real 裕量。"""
    import math
    from .benchmark import demo_scene, ring_cameras
    from .splat import fit_splat, splat_novelview_report
    from ..embodied.hybrid import run_episode
    from ..contracts import EvidenceGrade

    scene = demo_scene()
    # 1) 高斯泼溅：训练视角拟合，留出视角评估
    train_cams = ring_cameras(6, width=36, height=28)
    novel_cams = ring_cameras(4, width=36, height=28, phase=math.pi / 4)
    splat, history = fit_splat(scene, train_cams, n_gaussians=150, iters=40,
                               voxel=0.12, seed=0, verbose=True)
    splat_rep = splat_novelview_report(splat, novel_cams, scene)

    # 2) Real-to-Sim：带噪观测重建 → 单主障碍过门任务 → 混合控制成功率
    obs_cams = ring_cameras(10, width=48, height=36)
    task, clusters = real2sim_task(scene, obs_cams)
    r2s_rows = [run_episode(task, "hybrid", seed=seed + k)
                for k in range(episodes)]
    planning_obstacle = None
    if task.obstacles:
        planning_obstacle = [[round(v, 3) for v in task.obstacles[0][0]],
                             round(task.obstacles[0][1], 3)]
    r2s = {"success_rate": round(sum(r["success"] for r in r2s_rows) / episodes, 3),
           "plane_intersecting_clusters": len(clusters),
           "planning_obstacle": planning_obstacle,
           "true_plane_obstacles": 2}

    # 3) Sim-to-Real：朴素规划 vs 半径膨胀（域随机化安全裕量）
    s2r = sim2real_gap((([0.0, 0.0, 0.0], 0.55),), n_worlds=episodes)
    return {
        "module": "udos7.spatial gaussian splatting + real/sim transfer",
        "version": "v7.3.9",
        "evidence_grade": EvidenceGrade.CPU_PROTO.value,
        "gaussian_splat": {"n_gaussians": 150, "iters": 40,
                           "novel_view": splat_rep["aggregate"],
                           "loss_history": history},
        "real_to_sim": r2s,
        "sim_to_real": {"naive": s2r["naive"], "robust": s2r["robust"],
                        "margin": s2r["margin"]},
        "known_limits": ["轴对齐尺度、无旋转协方差/EWA，静态场景",
                         "Real-to-Sim 全局规划仅覆盖单主障碍过门",
                         "真实世界用固定 seed 扰动合成代理，非真机/MuJoCo"],
        "gates": ["完整 3DGS 训练与实时渲染需 GPU",
                  "真实深度相机标定与多目采集需硬件",
                  "MuJoCo(-MJX) 接触与真机 Sim-to-Real 需 GPU/HPC"],
    }
