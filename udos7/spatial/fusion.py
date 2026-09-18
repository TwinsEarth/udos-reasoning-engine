"""SpatialContext：多视角深度 TSDF 体素融合，并对任意新视角做深度预测。

这是“把多张图及其三维位姿共同编码成场景表征”的显式几何实现（无学习参数）：
观测越多、视角覆盖越全，新视角的几何一致性越好——可直接量化测量。
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import torch

from .camera import PinholeCamera, project_points


class VoxelContext:
    def __init__(self, bounds: Tuple[Tuple[float, float], ...], voxel: float = 0.12,
                 trunc_voxels: float = 3.0, occ_threshold: float = 0.6):
        self.bounds = bounds
        self.voxel = voxel
        self.trunc = trunc_voxels * voxel
        self.occ_threshold = occ_threshold      # tsdf < -0.6*voxel 判为内部
        ns = [int(torch.ceil(torch.tensor((b[1] - b[0]) / voxel)).item())
              for b in bounds]
        self.nx, self.ny, self.nz = ns
        xs = torch.arange(self.nx) * voxel + bounds[0][0] + voxel / 2
        ys = torch.arange(self.ny) * voxel + bounds[1][0] + voxel / 2
        zs = torch.arange(self.nz) * voxel + bounds[2][0] + voxel / 2
        gx, gy, gz = torch.meshgrid(xs, ys, zs, indexing="ij")
        self.centers = torch.stack([gx.reshape(-1), gy.reshape(-1),
                                    gz.reshape(-1)], dim=1)
        self.tsdf = torch.ones(self.centers.shape[0])
        self.weight = torch.zeros(self.centers.shape[0])

    def integrate(self, depth: torch.Tensor, hitmask: torch.Tensor,
                  cam: PinholeCamera) -> None:
        pix, z = project_points(self.centers, cam)
        px = torch.round(pix[:, 0]).long()
        py = torch.round(pix[:, 1]).long()
        in_img = (px >= 0) & (px < cam.width) & (py >= 0) & (py < cam.height) & (z > 0)
        idx = py.clamp(0, cam.height - 1) * cam.width + px.clamp(0, cam.width - 1)
        dflat = depth.reshape(-1)
        hflat = hitmask.reshape(-1)
        dirs, origin = cam.rays()
        t_ray = ((self.centers - origin) * dirs[idx]).sum(1)
        surface = dflat[idx]
        visible = in_img & hflat[idx] & (t_ray > 0) & (t_ray < surface + self.trunc)
        sdf = (surface - t_ray).clamp(-self.trunc, self.trunc)
        # 与已有值加权平均
        old_w = self.weight
        new_w = old_w + visible.float()
        upd = visible & (new_w > 0)
        self.tsdf[upd] = (self.tsdf[upd] * old_w[upd] + sdf[upd]) / new_w[upd]
        self.weight[upd] = new_w[upd]

    def occupied(self) -> torch.Tensor:
        return (self.weight > 0) & (self.tsdf < -self.occ_threshold * self.voxel)

    def _index_at(self, points: torch.Tensor):
        """世界坐标 → 体素索引（越界返回 -1）。"""
        ix = torch.floor((points[:, 0] - self.bounds[0][0]) / self.voxel).long()
        iy = torch.floor((points[:, 1] - self.bounds[1][0]) / self.voxel).long()
        iz = torch.floor((points[:, 2] - self.bounds[2][0]) / self.voxel).long()
        ok = ((ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
              & (iz >= 0) & (iz < self.nz))
        flat = torch.full((points.shape[0],), -1, dtype=torch.long)
        flat[ok] = (ix[ok] * self.ny * self.nz + iy[ok] * self.nz + iz[ok])
        return flat


def fuse_views(views: Sequence[Tuple[torch.Tensor, torch.Tensor, PinholeCamera]],
               bounds: Tuple[Tuple[float, float], ...] = ((-2.5, 2.5),) * 3,
               voxel: float = 0.12) -> VoxelContext:
    ctx = VoxelContext(bounds, voxel=voxel)
    for depth, hitmask, cam in views:
        ctx.integrate(depth, hitmask, cam)
    return ctx


def novel_view_depth(cam: PinholeCamera, ctx: VoxelContext,
                     near: float = 0.2, far: float = 9.0
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
    """沿视线步进，命中首个占据体素即作为预测表面。"""
    dirs, origin = cam.rays()
    M = dirs.shape[0]
    occ = ctx.occupied()
    steps = int((far - near) / (ctx.voxel * 0.8)) + 1
    ts = near + ctx.voxel * 0.8 * torch.arange(steps)
    depth = torch.zeros(M)
    found = torch.zeros(M, dtype=torch.bool)
    for t in ts:
        if found.all():
            break
        pts = origin.unsqueeze(0) + t * dirs
        flat = ctx._index_at(pts)
        hit = (~found) & (flat >= 0) & occ[flat.clamp_min(0)]
        depth[hit] = t
        found |= hit
    H, W = cam.height, cam.width
    return depth.view(H, W), found.view(H, W)


def depth_metrics(pred_depth: torch.Tensor, pred_mask: torch.Tensor,
                  gt_depth: torch.Tensor, gt_mask: torch.Tensor) -> dict:
    inter = pred_mask & gt_mask
    union = pred_mask | gt_mask
    iou = float(inter.sum() / union.sum().clamp_min(1))
    coverage = float((inter.sum() / gt_mask.sum().clamp_min(1)))
    precision = float((inter.sum() / pred_mask.sum().clamp_min(1)))
    mae = float((pred_depth[inter] - gt_depth[inter]).abs().mean()) \
        if inter.any() else float("nan")
    return {"silhouette_iou": round(iou, 4), "coverage": round(coverage, 4),
            "precision": round(precision, 4),
            "depth_mae": round(mae, 4),
            "gt_hit_ratio": round(float(gt_mask.float().mean()), 4)}
