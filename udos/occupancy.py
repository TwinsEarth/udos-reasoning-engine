"""
占据网格 (OccupancyGrid) + 有符号距离场 (DistanceField) — v3.5.0.dev2
========================================================================
analogy, not reproduction —— 合成 3D 体素代理, 不使用真实 LiDAR/TSDF/点云。

* OccupancyGrid: 规则体素网格, 用球体 (SpatialObject.radius) 填充占据;
  CPU 可算, 分辨率可配 (8/16/32)。
* DistanceField: 到最近占据体素的**有符号距离场** (占据为负, 自由为正),
  小规模网格暴力计算, 确定性。

设计纪律: 纯前向/零梯度/不改主权重; 空网格、越界点显式 ValueError。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .spatial import SpatialObject

logger = logging.getLogger("udos.occupancy")


class OccupancyGrid:
    """规则 3D 体素占据网格。

    Parameters
    ----------
    bounds:
        ((xmin, ymin, zmin), (xmax, ymax, zmax)) 世界坐标包围盒。
    resolution:
        每个轴的体素数 (整数元组或单值各向同)。
    """

    def __init__(self, bounds: Sequence[Sequence[float]] = ((-1, -1, -1),
                                                          (1, 1, 1)),
                 resolution: int | Sequence[int] = 16) -> None:
        lo = np.asarray(bounds[0], dtype=np.float64)
        hi = np.asarray(bounds[1], dtype=np.float64)
        if lo.shape != (3,) or hi.shape != (3,):
            raise ValueError("bounds 须为 ((3,),(3,))")
        if not (np.isfinite(lo).all() and np.isfinite(hi).all()):
            raise ValueError("bounds 含非有限值")
        if (hi <= lo).any():
            raise ValueError("bounds 须满足 max > min")
        if isinstance(resolution, int):
            res = (resolution, resolution, resolution)
        else:
            res = tuple(int(x) for x in resolution)
        if len(res) != 3 or any(r < 1 for r in res):
            raise ValueError("resolution 须为 3 个 >=1 的整数")
        self.lo, self.hi = lo, hi
        self.resolution = res
        self.voxel_size = ((self.hi - self.lo) / np.asarray(res)).tolist()
        # 0=自由, 1=占据
        self.grid = np.zeros(res, dtype=np.float32)

    # -- 坐标 <-> 体素索引 --------------------------------------------- #
    def _index(self, point: Sequence[float]) -> Tuple[int, int, int]:
        p = np.asarray(point, dtype=np.float64).reshape(-1)
        if p.size != 3 or not np.isfinite(p).all():
            raise ValueError("point 须为长度3的有限向量")
        if (p < self.lo).any() or (p > self.hi).any():
            raise ValueError(f"点 {p.tolist()} 越界 (bounds {self.lo.tolist()}"
                             f"~{self.hi.tolist()})")
        rel = (p - self.lo) / (self.hi - self.lo)
        idx = np.floor(rel * np.asarray(self.resolution)).astype(int)
        idx = np.clip(idx, 0, np.asarray(self.resolution) - 1)
        return int(idx[0]), int(idx[1]), int(idx[2])

    def voxel_center(self, ijk: Tuple[int, int, int]) -> np.ndarray:
        """体素中心世界坐标。"""
        i, j, k = ijk
        if not (0 <= i < self.resolution[0] and 0 <= j < self.resolution[1]
                and 0 <= k < self.resolution[2]):
            raise ValueError(f"体素索引越界: {ijk}")
        vc = (np.asarray([i, j, k]) + 0.5) * np.asarray(self.voxel_size) + self.lo
        return vc

    # -- 占据操作 ------------------------------------------------------- #
    def occupy_object(self, obj: SpatialObject) -> int:
        """用球体半径填充占据体素; 返回占据体素数。"""
        if not isinstance(obj, SpatialObject):
            raise ValueError("obj 须为 SpatialObject")
        r2 = obj.radius ** 2
        n = 0
        # 粗筛包围盒体素范围, 再精判距离
        for i in range(self.resolution[0]):
            for j in range(self.resolution[1]):
                for k in range(self.resolution[2]):
                    c = self.voxel_center((i, j, k))
                    if float(np.sum((c - obj.position) ** 2)) <= r2:
                        if self.grid[i, j, k] < 0.5:
                            n += 1
                        self.grid[i, j, k] = 1.0
        return n

    def is_occupied(self, point: Sequence[float]) -> bool:
        i, j, k = self._index(point)
        return bool(self.grid[i, j, k] > 0.5)

    @property
    def n_occupied(self) -> int:
        return int((self.grid > 0.5).sum())

    @property
    def occupied_indices(self) -> np.ndarray:
        return np.argwhere(self.grid > 0.5)

    def __len__(self) -> int:
        return int(self.grid.size)

    def config_dict(self) -> Dict[str, Any]:
        return {"kind": "OccupancyGrid", "bounds": [self.lo.tolist(),
                self.hi.tolist()], "resolution": self.resolution,
                "n_occupied": self.n_occupied}


class DistanceField:
    """有符号距离场 (基于 OccupancyGrid)。

    * 自由体素: 到最近占据体素中心的距离 (正);
    * 占据体素: 到最近自由体素中心的距离 (负)。
    小规模网格暴力计算, 确定性。
    """

    def __init__(self, occ: OccupancyGrid) -> None:
        if not isinstance(occ, OccupancyGrid):
            raise ValueError("occ 须为 OccupancyGrid")
        self.occ = occ
        self.sdf = self._compute()

    def _compute(self) -> np.ndarray:
        g = self.occ.grid
        occ_idx = np.argwhere(g > 0.5)
        free_idx = np.argwhere(g <= 0.5)
        out = np.full(g.shape, np.inf, dtype=np.float64)
        # 预计算所有体素中心坐标
        ii, jj, kk = np.meshgrid(
            np.arange(g.shape[0]), np.arange(g.shape[1]),
            np.arange(g.shape[2]), indexing="ij")
        centers = np.stack([ii.ravel(), jj.ravel(), kk.ravel()], axis=1)
        centers_world = (centers + 0.5) * np.asarray(self.occ.voxel_size) + self.occ.lo

        def nearest(own: np.ndarray) -> np.ndarray:
            """对所有体素中心, 求到 own 点集的最近距离 (分块, 不爆内存)。"""
            Q = centers_world.shape[0]
            best = np.full(Q, np.inf, dtype=np.float64)
            if own.size == 0:
                return best
            own_world = ((own + 0.5) * np.asarray(self.occ.voxel_size)
                         + self.occ.lo)
            chunk = 512
            for s in range(0, own_world.shape[0], chunk):
                ow = own_world[s:s + chunk]
                d = np.sqrt(((centers_world[:, None, :] - ow[None, :, :]) ** 2
                             ).sum(-1)).min(axis=1)
                best = np.minimum(best, d)
            return best

        d_free = nearest(occ_idx)     # 每个体素到最近占据
        d_occ = nearest(free_idx)     # 每个体素到最近自由
        occ_mask = (g > 0.5).ravel()
        sdf = np.where(occ_mask, -d_occ, d_free)
        return sdf.reshape(g.shape)

    def at(self, point: Sequence[float]) -> float:
        """查询世界点的 SDF (会先体素化; 越界/空占据显式报错)。"""
        if self.occ.n_occupied == 0:
            raise ValueError("空占据网格无法计算距离场")
        i, j, k = self.occ._index(point)
        return float(self.sdf[i, j, k])

    def near_surface(self, tol: float = 1e-6) -> int:
        """|sdf|<=tol 的体素数 (近似等值面体素)。"""
        return int((np.abs(self.sdf) <= tol).sum())
