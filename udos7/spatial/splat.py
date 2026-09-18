"""最小 CPU 3D 高斯泼溅（3DGS-lite）：可微高斯点元 + alpha 合成 + 拟合。

与完整 3DGS 的差距（诚实标注，cpu-proto）：
- 仅轴对齐各向异性尺度（无旋转四元数/协方差雅可比 EWA 投影）；
-  footprint 用针孔一阶近似 sigma = f * scale / depth；
- 小分辨率、少量高斯、Adam 几十步；无运动/动态表达（静态场景）。
完整训练与实时渲染需 GPU（闸门）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch

from .camera import PinholeCamera, project_points
from .scene import Scene, render_view
from .fusion import VoxelContext, fuse_views, depth_metrics


@dataclass
class GaussianSplat:
    means: torch.Tensor          # (N,3) 可优化
    scales: torch.Tensor         # (N,3) 轴对齐半尺度
    colors: torch.Tensor         # (N,3) 0..1
    opacity: torch.Tensor        # (N,) 0..1

    def parameters(self):
        return [self.means, self.scales, self.colors, self.opacity]

    def clone(self):
        return GaussianSplat(self.means.detach().clone(),
                             self.scales.detach().clone(),
                             self.colors.detach().clone(),
                             self.opacity.detach().clone())


def render_splat(cam: PinholeCamera, splat: GaussianSplat
                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """前到后 alpha 合成，返回深度[H,W]、颜色[H,W,3]、掩膜[H,W]。"""
    pix, z = project_points(splat.means, cam)
    visible = (z > 0.1) & (pix[:, 0] > -8) & (pix[:, 0] < cam.width + 8) \
        & (pix[:, 1] > -8) & (pix[:, 1] < cam.height + 8)
    idx = visible.nonzero(as_tuple=False).flatten()
    idx = idx[torch.argsort(z[idx])]            # 前到后
    H, W = cam.height, cam.width
    jy, ix = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    px, py = ix.float().reshape(-1), jy.float().reshape(-1)
    T = torch.ones(H * W)
    color = torch.zeros(H * W, 3)
    depth = torch.zeros(H * W)
    for i in idx.tolist():
        sx = max(cam.fx * splat.scales[i, 0].item() / z[i].item(), 1e-3)
        sy = max(cam.fy * splat.scales[i, 1].item() / z[i].item(), 1e-3)
        d2 = ((px - pix[i, 0]) / sx) ** 2 + ((py - pix[i, 1]) / sy) ** 2
        alpha = splat.opacity[i] * torch.exp(-0.5 * d2)
        alpha = alpha.clamp(0.0, 0.999)
        w = T * alpha
        color += w.unsqueeze(1) * splat.colors[i].unsqueeze(0)
        depth += w * z[i]
        T = T * (1.0 - alpha)
    mask = (1.0 - T) > 0.30
    return depth.view(H, W), color.view(H, W, 3), mask.view(H, W)


def init_splat_from_context(ctx: VoxelContext, scene: Scene,
                            n_gaussians: int, seed: int = 0) -> GaussianSplat:
    """用 TSDF 占据体素初始化高斯中心（Real-to-Sim 友好的冷启动）。"""
    g = torch.Generator().manual_seed(seed)
    occ = ctx.occupied()
    pts = ctx.centers[occ]
    if pts.shape[0] == 0:
        pts = (torch.rand(n_gaussians, 3, generator=g) - 0.5) * 2.0
    sel = torch.randint(0, pts.shape[0], (n_gaussians,), generator=g)
    means = pts[sel].clone()
    d = torch.cdist(means, scene.centers)
    nearest = d.argmin(1)
    colors = scene.colors[nearest].clone()
    scales = torch.full((n_gaussians, 3), max(ctx.voxel * 0.9, 0.05))
    opacity = torch.full((n_gaussians,), 0.85)
    return GaussianSplat(means.clone().requires_grad_(True),
                         scales.clone().requires_grad_(True),
                         colors.clone().requires_grad_(True),
                         opacity.clone().requires_grad_(True))


def splat_loss(splat: GaussianSplat, cams: Sequence[PinholeCamera],
               scene: Scene) -> Tuple[torch.Tensor, dict]:
    l_color = torch.tensor(0.0)
    l_depth = torch.tensor(0.0)
    n_pix = 0
    with torch.no_grad():
        pass
    for cam in cams:
        gd, gc, gm = render_view(cam, scene)
        pd, pc, pm = render_splat(cam, splat)
        l_color = l_color + (pc - gc).abs().mean()
        both = pm & gm
        if both.any():
            l_depth = l_depth + (pd[both] - gd[both]).abs().mean()
        n_pix += 1
    return l_color + 0.5 * l_depth, {"color_l1": float((l_color / max(1, n_pix)).detach()),
                                    "depth_l1": float((l_depth / max(1, n_pix)).detach())}


def fit_splat(scene: Scene, train_cams: Sequence[PinholeCamera],
              n_gaussians: int = 160, iters: int = 60, lr: float = 0.03,
              voxel: float = 0.12, seed: int = 0, verbose: bool = False
              ) -> Tuple[GaussianSplat, List[dict]]:
    views = []
    for c in train_cams:
        d, _, m = render_view(c, scene)
        views.append((d, m, c))
    ctx = fuse_views(views, voxel=voxel)
    splat = init_splat_from_context(ctx, scene, n_gaussians, seed)
    opt = torch.optim.Adam(splat.parameters(), lr=lr)
    history = []
    for it in range(iters):
        opt.zero_grad()
        loss, parts = splat_loss(splat, train_cams, scene)
        # 轻微 opacity 正则，避免全不透明堆叠
        loss = loss + 1e-3 * splat.opacity.abs().mean()
        loss.backward()
        opt.step()
        with torch.no_grad():
            splat.colors.clamp_(0.0, 1.0)
            splat.opacity.clamp_(0.05, 0.98)
            splat.scales.clamp_(0.02, 0.35)
        if verbose and it % 10 == 0:
            history.append({"iter": it, "loss": float(loss), **parts})
    return splat, history


def splat_novelview_report(splat: GaussianSplat,
                           novel_cams: Sequence[PinholeCamera],
                           scene: Scene) -> dict:
    rows = []
    for c in novel_cams:
        gd, _, gm = render_view(c, scene)
        pd, _, pm = render_splat(c, splat)
        rows.append(depth_metrics(pd, pm, gd, gm))
    agg = {k: round(sum(r[k] for r in rows if r[k] == r[k]) /
                    max(1, sum(1 for r in rows if r[k] == r[k])), 4)
           for k in ("silhouette_iou", "coverage", "precision", "depth_mae")}
    return {"aggregate": agg, "per_view": rows}
