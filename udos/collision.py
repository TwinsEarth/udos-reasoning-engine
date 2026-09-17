"""
碰撞检测 + 最近邻 (CollisionDetector / NearestNeighbor) — v3.5.0.dev3
========================================================================
analogy, not reproduction —— 球体/盒体代理碰撞, 非真实物理引擎 (无连续冲量/
摩擦/解算)。所有判定为解析几何, 零梯度、确定性。

* CollisionDetector:
    - 球-球代理碰撞: 中心距 < r1 + r2;
    - 球-轴对齐盒 (AABB) 代理碰撞;
    - 连续帧接触检测: 跨两帧判定"新进入接触"的物体对。
* NearestNeighbor:
    - 给定点/物体, 暴力查最近 k 个物体 (可剔除自身);
    - 小规模场景直接暴力; 网格加速为可选占位 (v3.5 线合成场景无需)。

设计纪律: 空场景 / 非法输入显式 ValueError; 不改主权重。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .spatial import SpatialObject, SpatialScene

logger = logging.getLogger("udos.collision")


class CollisionDetector:
    """球体 / 球-AABB 代理碰撞检测器 (纯解析)。"""

    def __init__(self, contact_tol: float = 1e-9) -> None:
        self.contact_tol = float(contact_tol)

    # -- 基本图元 ------------------------------------------------------- #
    def ball_ball(self, a: SpatialObject, b: SpatialObject) -> Dict[str, Any]:
        """球-球: 中心距 <= r_a + r_b 视为接触; 返回接触标志 + 穿透深度。"""
        if not isinstance(a, SpatialObject) or not isinstance(b, SpatialObject):
            raise ValueError("a/b 须为 SpatialObject")
        d = a.distance_to(b)
        threshold = a.radius + b.radius
        penetrating = d <= threshold + self.contact_tol
        return {"contact": bool(penetrating), "distance": float(d),
                "threshold": float(threshold),
                "penetration": float(max(0.0, threshold - d))}

    def ball_box(self, center: Sequence[float], radius: float,
                 box_min: Sequence[float], box_max: Sequence[float]
                 ) -> Dict[str, Any]:
        """球-轴对齐盒代理碰撞: 盒上最近点到球心距离 <= r。"""
        c = np.asarray(center, dtype=np.float64).reshape(-1)
        bmin = np.asarray(box_min, dtype=np.float64).reshape(-1)
        bmax = np.asarray(box_max, dtype=np.float64).reshape(-1)
        if c.size != 3 or bmin.size != 3 or bmax.size != 3:
            raise ValueError("center/box_min/box_max 须长度 3")
        if not np.isfinite(c).all() or not np.isfinite(bmin).all() \
                or not np.isfinite(bmax).all():
            raise ValueError("含非有限值")
        if (bmax < bmin).any():
            raise ValueError("box_max 须 >= box_min")
        closest = np.clip(c, bmin, bmax)
        d = float(np.linalg.norm(c - closest))
        return {"contact": bool(d <= radius + self.contact_tol),
                "distance": d, "radius": float(radius)}

    # -- 场景级 --------------------------------------------------------- #
    def detect_contacts(self, scene: SpatialScene) -> List[Dict[str, Any]]:
        """枚举场景中所有接触的物体对 (无向去重)。"""
        if len(scene) < 2:
            raise ValueError("碰撞检测需至少 2 个物体")
        ids = scene.ids()
        out: List[Dict[str, Any]] = []
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = scene.get(ids[i]), scene.get(ids[j])
                r = self.ball_ball(a, b)
                if r["contact"]:
                    out.append({"a": ids[i], "b": ids[j],
                                "penetration": r["penetration"],
                                "distance": r["distance"]})
        return out

    def continuous_contact(self, prev: SpatialScene, cur: SpatialScene
                           ) -> List[Dict[str, Any]]:
        """跨帧接触事件: cur 接触但 prev 未接触的物体对 (新进入接触)。"""
        if len(prev) < 2 or len(cur) < 2:
            raise ValueError("连续帧检测需两帧均 >=2 物体")
        prev_set = {(e["a"], e["b"]) for e in self.detect_contacts(prev)}
        events = []
        for e in self.detect_contacts(cur):
            pair = (e["a"], e["b"])
            if pair not in prev_set:
                events.append(e)
        return events


class NearestNeighbor:
    """最近邻查询 (暴力, 小规模合成场景)。"""

    def __init__(self, scene: SpatialScene) -> None:
        if not isinstance(scene, SpatialScene):
            raise ValueError("scene 须为 SpatialScene")
        self.scene = scene

    def query(self, point: Sequence[float], k: int = 1,
              exclude: Optional[str] = None) -> List[Dict[str, Any]]:
        """返回距 point 最近的 k 个物体 (含距离)。"""
        self._require_nonempty()
        if k < 1:
            raise ValueError("k 须 >= 1")
        p = np.asarray(point, dtype=np.float64).reshape(-1)
        if p.size != 3 or not np.isfinite(p).all():
            raise ValueError("point 须为长度3的有限向量")
        objs = [o for o in self.scene.objects() if o.object_id != exclude]
        if len(objs) == 0:
            raise ValueError("剔除后无可查询物体")
        dists = [float(np.linalg.norm(o.position - p)) for o in objs]
        order = np.argsort(dists)[:min(k, len(objs))]
        return [{"object_id": objs[i].object_id, "distance": float(dists[i])}
                for i in order]

    def nearest_to_object(self, object_id: str, k: int = 1
                          ) -> List[Dict[str, Any]]:
        """以某物体自身为查询点, 找最近的其他物体 (自动剔除自身)。"""
        self._require_nonempty()
        self.scene.get(object_id)  # 存在性校验
        p = self.scene.get(object_id).position
        return self.query(p, k=k, exclude=object_id)

    def _require_nonempty(self) -> None:
        if len(self.scene) == 0:
            raise ValueError("空场景不支持最近邻查询")
