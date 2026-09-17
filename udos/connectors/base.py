"""connector 基类与 catalog 行 -> ResourceSpec 的装配。"""
from __future__ import annotations

from typing import Any, Dict, Optional

from ..resource_registry import (ResourceConnector, ResourceSpec,
                                 normalize_trajectory)

_KIND_MAP = {"模型库": "model", "数据库": "dataset", "动作库": "action"}


def build_spec(row: Dict[str, Any], *, level: str,
               requires_gpu: bool = False, requires_weights: bool = False,
               requires_pkg: Optional[str] = None,
               profile_tags=()) -> ResourceSpec:
    return ResourceSpec(
        id=row["id"], name=row["name"],
        kind=_KIND_MAP.get(row["source_class"], "action"),
        license=row.get("license") or "UNKNOWN",
        level=level, priority=row.get("priority", "normal"),
        requires_gpu=requires_gpu, requires_weights=requires_weights,
        requires_pkg=requires_pkg, profile_tags=tuple(profile_tags),
        category=row.get("category", ""), org=row.get("org", ""),
        source_url=row.get("source_url", ""),
        cpu_feasibility=row.get("cpu_feasibility", ""),
        udos_fit=row.get("udos_fit", ""),
        oss_triangle=row.get("oss_triangle", ""),
        scale=row.get("scale", ""))


class GenericConnector(ResourceConnector):
    """默认 connector: 元数据 L0 + 通用轨迹归一化 L1。

    绝大多数 catalog 条目用它; 专门格式 (动作分块/VLA token/SMPL/BVH/URDF)
    在 specialized.py 里覆写 to_udos 做更贴近真实格式的解析。"""

    def __init__(self, spec: ResourceSpec):
        super().__init__(spec)

    def to_udos(self, payload: Any) -> Dict[str, Any]:
        return normalize_trajectory(payload)
