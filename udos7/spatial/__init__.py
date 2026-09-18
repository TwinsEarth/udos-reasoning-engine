"""v7.3.8 新视角预测与空间上下文（New View Prediction / Spatial Context）。

Atlas 类世界模型把“给定若干带位姿的观察，预测任意新视角下的场景”当作基础
任务。本包在 CPU 合成三维场景上给出**可真跑的几何原型**：

- camera.py  针孔相机模型（位姿、投影、反投影、视线生成）
- scene.py   球体基元合成场景 + 光线求交，渲染深度图/颜色/掩膜（已知真值）
- fusion.py  SpatialContext：多视角深度 TSDF 体素融合 → 新视角深度预测
- benchmark.py  多视角上下文 vs 单视角的新视角几何指标对比

证据等级 cpu-proto：无真实图像、无可学习 NeRF/3DGS（后者在 v7.3.9），
TSDF 是经典显式几何；它验证的是“新视角预测必须携带相机几何与空间一致性”
这一机制，而非 Atlas 的网络指标。
"""
from __future__ import annotations

from .camera import PinholeCamera, look_at, project_points, unproject
from .scene import Sphere, Scene, render_view
from .fusion import VoxelContext, fuse_views, novel_view_depth, depth_metrics
from .benchmark import run_novelview_benchmark, EXTERNAL_REFERENCE

__all__ = ["PinholeCamera", "look_at", "project_points", "unproject",
           "Sphere", "Scene", "render_view", "VoxelContext", "fuse_views",
           "novel_view_depth", "depth_metrics", "run_novelview_benchmark",
           "EXTERNAL_REFERENCE"]
