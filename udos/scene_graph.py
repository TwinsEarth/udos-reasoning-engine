"""
场景图 (Scene Graph) — v3.5.0.dev1
================================================================
在 SpatialScene 之上构建物体节点 + 空间关系边。所有关系均由**解析几何计算,
非学习、零梯度**: 给定两个物体的位置/半径, 直接判定 above/below/left/right/
near/far/inside。

analogy, not reproduction —— 这是合成低维代理关系, 不使用真实 CAD / 点云 /
3D 检测。关系语义采用可配置的参考轴 (默认 up=z, right=x), 阈值显式可调。

设计纪律:
    * 纯前向、确定性、不改主权重;
    * 空场景 / 缺失节点显式 ValueError;
    * 边是有向的 (subject -> object); 对称关系由调用方反向查询。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .spatial import SpatialScene, SpatialObject

logger = logging.getLogger("udos.scene_graph")

# 支持的空间关系谓词
RELATIONS = ("above", "below", "left", "right", "near", "far", "inside")


class SceneGraph:
    """物体节点 + 空间关系边的图。

    Parameters
    ----------
    scene:
        SpatialScene 实例 (节点来源)。
    near_dist:
        near 判定阈值 (中心距离 <= near_dist 视为 near)。
    far_dist:
        far 判定阈值 (中心距离 >= far_dist 视为 far)。
    up_axis, right_axis:
        上方向 / 右方向的轴索引 (默认 2=z, 0=x)。
    vert_gap:
        above/below 所需的垂直间距阈值 (避免共面抖动)。
    """

    def __init__(self, scene: SpatialScene, near_dist: float = 1.0,
                 far_dist: Optional[float] = None, up_axis: int = 2,
                 right_axis: int = 0, vert_gap: float = 1e-6) -> None:
        if not isinstance(scene, SpatialScene):
            raise ValueError("scene 须为 SpatialScene")
        if near_dist <= 0:
            raise ValueError("near_dist 须 > 0")
        self.scene = scene
        self.near_dist = float(near_dist)
        self.far_dist = float(far_dist if far_dist is not None
                              else 2.0 * near_dist)
        if self.far_dist < self.near_dist:
            raise ValueError("far_dist 须 >= near_dist")
        if up_axis == right_axis or up_axis not in (0, 1, 2) \
                or right_axis not in (0, 1, 2):
            raise ValueError("up_axis/right_axis 须为 0/1/2 且互不相同")
        self.up_axis = int(up_axis)
        self.right_axis = int(right_axis)
        self.vert_gap = float(vert_gap)

    # -- 节点 ----------------------------------------------------------- #
    def nodes(self) -> List[str]:
        return self.scene.ids()

    def __len__(self) -> int:
        return len(self.scene)

    def _get(self, oid: str) -> SpatialObject:
        if oid not in self.scene:
            raise ValueError(f"图中不存在节点: {oid}")
        return self.scene.get(oid)

    # -- 单条关系判定 --------------------------------------------------- #
    def relate(self, subject: str, obj: str) -> Dict[str, Any]:
        """判定 subject -> obj 的全部空间谓词 + 几何量。"""
        if subject == obj:
            raise ValueError("不能对同一物体判定关系")
        a = self._get(subject)
        b = self._get(obj)
        d = a.distance_to(b)
        rel = a.relative_position(b)  # b - a
        vert = -rel[self.up_axis]       # a 相对 b 的垂直偏移
        # 水平 (垂直于 up 轴) 距离
        horiz = float(np.sqrt(sum(
            (rel[i] ** 2) for i in range(3) if i != self.up_axis)))
        side = rel[self.right_axis]     # obj - subj 在右轴方向
        # 谓词描述 subject 相对 object 的方位:
        # subject 在 object 左侧 <=> subj.x < obj.x <=> (obj-subj).x > 0
        above = vert > self.vert_gap and horiz < self.near_dist
        below = vert < -self.vert_gap and horiz < self.near_dist
        right = side < -self.vert_gap
        left = side > self.vert_gap
        near = d <= self.near_dist
        far = d >= self.far_dist
        inside = (b.radius > a.radius
                  and d + a.radius <= b.radius + self.vert_gap)
        return {
            "subject": subject, "object": obj,
            "distance": float(d), "vertical": float(vert),
            "horizontal": horiz,
            "above": bool(above), "below": bool(below),
            "left": bool(left), "right": bool(right),
            "near": bool(near), "far": bool(far),
            "inside": bool(inside),
        }

    # -- 图遍历 --------------------------------------------------------- #
    def edges(self, relation: Optional[str] = None) -> List[Dict[str, Any]]:
        """枚举所有有向边; 可按单个 relation 过滤 (该谓词为真的边)。"""
        self._require_nonempty()
        if relation is not None and relation not in RELATIONS:
            raise ValueError(f"未知关系 {relation!r}; 支持 {RELATIONS}")
        ids = self.scene.ids()
        out: List[Dict[str, Any]] = []
        for i, s in enumerate(ids):
            for j, o in enumerate(ids):
                if i == j:
                    continue
                r = self.relate(s, o)
                if relation is None or r[relation]:
                    out.append(r)
        return out

    def neighbors(self, obj: str, relation: str) -> List[str]:
        """与 obj 满足 relation 的其他节点 id 列表。"""
        if relation not in RELATIONS:
            raise ValueError(f"未知关系 {relation!r}; 支持 {RELATIONS}")
        self._get(obj)  # 存在性校验
        return [e["object"] for e in self.edges(relation=relation)
                if e["subject"] == obj]

    def query(self, relation: str) -> List[Dict[str, str]]:
        """查询所有满足 relation 的 (subject, object) 有向对。"""
        if relation not in RELATIONS:
            raise ValueError(f"未知关系 {relation!r}; 支持 {RELATIONS}")
        return [{"subject": e["subject"], "object": e["object"]}
                for e in self.edges(relation=relation)]

    def _require_nonempty(self) -> None:
        if len(self.scene) == 0:
            raise ValueError("空场景图不支持遍历/查询")

    def config_dict(self) -> Dict[str, Any]:
        return {"kind": "SceneGraph", "near_dist": self.near_dist,
                "far_dist": self.far_dist, "up_axis": self.up_axis,
                "right_axis": self.right_axis, "n_nodes": len(self.scene)}
