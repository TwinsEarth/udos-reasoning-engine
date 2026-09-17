"""
空间查询引擎 (SpatialQueryEngine) — v3.5.0.dev5
================================================================
analogy, not reproduction —— 合成低维查询代理, 非真实射线追踪/光线投射引擎。

提供四类纯解析查询 (零梯度、确定性):
    * ray_sphere: 射线-球相交 (解析二次方程);
    * line_of_sight: 两点视线遮挡 (段-球相交);
    * range_search: 球形/盒形区域包含检索;
    * box_query: 轴对齐盒内物体检索。

复用 spatial / collision 既有几何, 不改主权重。空场景 / 非法输入显式 ValueError。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .spatial import SpatialObject, SpatialScene

logger = logging.getLogger("udos.spatial_query")


class SpatialQueryEngine:
    """基于 SpatialScene 的解析空间查询引擎。"""

    def __init__(self, scene: SpatialScene) -> None:
        if not isinstance(scene, SpatialScene):
            raise ValueError("scene 须为 SpatialScene")
        self.scene = scene

    # -- 基本图元 ------------------------------------------------------- #
    @staticmethod
    def ray_sphere(origin: Sequence[float], direction: Sequence[float],
                   center: Sequence[float], radius: float) -> Dict[str, Any]:
        """射线 o + t*d 与球 (center, radius) 相交; 返回 {hit, t_near, point}。"""
        o = np.asarray(origin, dtype=np.float64).reshape(-1)
        d = np.asarray(direction, dtype=np.float64).reshape(-1)
        c = np.asarray(center, dtype=np.float64).reshape(-1)
        if o.size != 3 or d.size != 3 or c.size != 3:
            raise ValueError("origin/direction/center 须长度 3")
        if not (np.isfinite(o).all() and np.isfinite(d).all()
                and np.isfinite(c).all()):
            raise ValueError("含非有限值")
        dn = float(np.linalg.norm(d))
        if dn == 0.0:
            raise ValueError("direction 不能为零向量")
        d = d / dn
        oc = o - c
        b = 2.0 * float(oc @ d)
        cc = float(oc @ oc) - radius ** 2
        disc = b ** 2 - 4.0 * cc
        if disc < 0:
            return {"hit": False, "t_near": None, "point": None}
        sq = float(np.sqrt(disc))
        t1 = (-b - sq) / 2.0
        t2 = (-b + sq) / 2.0
        t_near = t1 if t1 >= 0 else (t2 if t2 >= 0 else None)
        if t_near is None:
            return {"hit": False, "t_near": None, "point": None}
        pt = o + t_near * d
        return {"hit": True, "t_near": float(t_near), "point": pt.tolist()}

    # -- 场景级查询 ----------------------------------------------------- #
    def raycast(self, origin: Sequence[float], direction: Sequence[float]
                ) -> Dict[str, Any]:
        """对场景所有物体做射线投射, 返回最近命中 (无命中则 hit=False)。"""
        self._require_nonempty()
        best: Optional[Dict[str, Any]] = None
        for o in self.scene.objects():
            r = self.ray_sphere(origin, direction, o.position, o.radius)
            if r["hit"] and (best is None or r["t_near"] < best["t_near"]):
                best = {"object_id": o.object_id, "t_near": r["t_near"],
                        "point": r["point"]}
        if best is None:
            return {"hit": False, "object_id": None, "t_near": None,
                    "point": None}
        return {"hit": True, **best}

    def line_of_sight(self, a: str, b: str,
                      ignore_endpoints: bool = True) -> Dict[str, Any]:
        """物体 a 到物体 b 中心的视线是否被场景中其他球遮挡。"""
        self._require_nonempty()
        oa = self.scene.get(a)
        ob = self.scene.get(b)
        if a == b:
            raise ValueError("a/b 不能为同一物体")
        start, end = oa.position, ob.position
        seg = end - start
        length = float(np.linalg.norm(seg))
        if length == 0.0:
            return {"visible": True, "blocker": None}
        d = seg / length
        for o in self.scene.objects():
            if ignore_endpoints and o.object_id in (a, b):
                continue
            # 段-球: 射线相交且 t 在 (0, length) 内
            r = self.ray_sphere(start, d, o.position, o.radius)
            if r["hit"] and 0.0 < r["t_near"] < length:
                return {"visible": False, "blocker": o.object_id,
                        "t": r["t_near"]}
        return {"visible": True, "blocker": None}

    def range_search(self, center: Sequence[float], radius: float
                     ) -> List[Dict[str, Any]]:
        """球形区域检索: 中心距 <= radius 的物体 (含距离)。"""
        self._require_nonempty()
        c = np.asarray(center, dtype=np.float64).reshape(-1)
        if c.size != 3 or not np.isfinite(c).all():
            raise ValueError("center 须长度3有限向量")
        if radius < 0:
            raise ValueError("radius 须 >= 0")
        out = []
        for o in self.scene.objects():
            d = float(np.linalg.norm(o.position - c))
            if d <= radius:
                out.append({"object_id": o.object_id, "distance": d})
        return sorted(out, key=lambda e: e["distance"])

    def box_query(self, box_min: Sequence[float], box_max: Sequence[float]
                  ) -> List[str]:
        """轴对齐盒内物体中心检索, 返回 object_id 列表。"""
        self._require_nonempty()
        bmin = np.asarray(box_min, dtype=np.float64).reshape(-1)
        bmax = np.asarray(box_max, dtype=np.float64).reshape(-1)
        if bmin.size != 3 or bmax.size != 3 or (bmax < bmin).any():
            raise ValueError("box_min/box_max 须长度3且 max>=min")
        ids = []
        for o in self.scene.objects():
            p = o.position
            if (p >= bmin).all() and (p <= bmax).all():
                ids.append(o.object_id)
        return ids

    def _require_nonempty(self) -> None:
        if len(self.scene) == 0:
            raise ValueError("空场景不支持空间查询")
