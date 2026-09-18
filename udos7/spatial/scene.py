"""球体基元合成三维场景：解析光线求交，渲染深度图、颜色、命中掩膜（真值源）。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch

from .camera import PinholeCamera


@dataclass(frozen=True)
class Sphere:
    center: Tuple[float, float, float]
    radius: float
    color: Tuple[float, float, float] = (0.7, 0.7, 0.7)


class Scene:
    def __init__(self, spheres: Sequence[Sphere]):
        self.spheres = list(spheres)
        self.centers = torch.tensor([s.center for s in self.spheres],
                                    dtype=torch.float32)
        self.radii = torch.tensor([s.radius for s in self.spheres],
                                  dtype=torch.float32)
        self.colors = torch.tensor([s.color for s in self.spheres],
                                   dtype=torch.float32)


def render_view(cam: PinholeCamera, scene: Scene
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """渲染 (深度[H,W], 颜色[H,W,3], 命中掩膜[H,W])。深度为沿视线距离。"""
    dirs, origin = cam.rays()                       # (M,3)
    n = dirs.shape[0]
    best_t = torch.full((n,), float("inf"))
    best_idx = torch.full((n,), -1, dtype=torch.long)
    for k in range(len(scene.spheres)):
        oc = origin - scene.centers[k]
        b = (dirs * oc).sum(1)
        c = (oc * oc).sum() - scene.radii[k] ** 2
        disc = b * b - c
        hit = disc > 0
        t = -b - torch.sqrt(disc.clamp_min(0.0))    # 近交点
        valid = hit & (t > 1e-4) & (t < best_t)
        best_t = torch.where(valid, t, best_t)
        best_idx = torch.where(valid, torch.full_like(best_idx, k), best_idx)
    hitmask = best_idx >= 0
    depth = torch.where(hitmask, best_t, torch.zeros_like(best_t))
    color = torch.zeros(n, 3)
    color[hitmask] = scene.colors[best_idx[hitmask]]
    H, W = cam.height, cam.width
    return (depth.view(H, W), color.view(H, W, 3), hitmask.view(H, W))
