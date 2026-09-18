"""针孔相机：OpenCV 约定（x 右、y 下、z 前向），位姿用 eye/look/up 给定。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import torch


def look_at(eye: Sequence[float], look: Sequence[float],
            up: Sequence[float] = (0.0, 0.0, 1.0)) -> Tuple[torch.Tensor, torch.Tensor]:
    """返回世界→相机旋转 R(3x3) 与相机世界坐标 eye(3,)。"""
    eye = torch.as_tensor(eye, dtype=torch.float32)
    look = torch.as_tensor(look, dtype=torch.float32)
    up = torch.as_tensor(up, dtype=torch.float32)
    f = look - eye
    f = f / f.norm()
    s = torch.cross(f, up, dim=0)
    s = s / s.norm()
    u = torch.cross(s, f, dim=0)          # 已单位化
    R = torch.stack([s, -u, f], dim=0)    # OpenCV: x 右, y 下, z 前
    return R, eye


@dataclass
class PinholeCamera:
    width: int = 48
    height: int = 36
    fov_deg: float = 60.0
    eye: Tuple[float, float, float] = (0.0, -4.0, 1.0)
    look: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    up: Tuple[float, float, float] = (0.0, 0.0, 1.0)

    def __post_init__(self):
        self.fx = self.fy = (self.width / 2.0) / torch.tensor(
            self.fov_deg / 2.0).deg2rad().item()
        self.cx = self.width / 2.0
        self.cy = self.height / 2.0
        self.R, self.eye_t = look_at(self.eye, self.look, self.up)

    def rays(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回每像素世界系单位视线 (H*W,3) 与原点 (3,)。"""
        j, i = torch.meshgrid(torch.arange(self.height),
                              torch.arange(self.width), indexing="ij")
        x = (i.float() - self.cx) / self.fx
        y = (j.float() - self.cy) / self.fy
        dirs_cam = torch.stack([x.reshape(-1), y.reshape(-1),
                                torch.ones_like(x.reshape(-1))], dim=1)
        dirs_world = dirs_cam @ self.R
        dirs_world = dirs_world / dirs_world.norm(dim=1, keepdim=True)
        return dirs_world, self.eye_t.clone()


def project_points(P: torch.Tensor, cam: PinholeCamera) -> Tuple[torch.Tensor, torch.Tensor]:
    """世界点 (N,3) → 像素坐标 (N,2) 与相机深度 z (N,)；z<=0 表示在身后。"""
    Pc = (P - cam.eye_t) @ cam.R.T
    z = Pc[:, 2]
    x = cam.fx * Pc[:, 0] / z.clamp_min(1e-6) + cam.cx
    y = cam.fy * Pc[:, 1] / z.clamp_min(1e-6) + cam.cy
    return torch.stack([x, y], dim=1), z


def unproject(pix_x: torch.Tensor, pix_y: torch.Tensor, depth: torch.Tensor,
              cam: PinholeCamera) -> torch.Tensor:
    """像素 + 深度 → 世界坐标点 (N,3)。"""
    p_cam = torch.stack([depth * (pix_x - cam.cx) / cam.fx,
                         depth * (pix_y - cam.cy) / cam.fy,
                         depth], dim=1)
    return cam.eye_t + p_cam @ cam.R
