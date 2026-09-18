"""v7.4.0 第一人称经验数据基础设施（Egocentric Experience Data Flywheel）。

对标 egocentric data boom 中“把人类真实活动变成 robot training-ready 数据”的
工业管线（Maxinsights/Dyna-2 等公开报道，数字均为 unverified 外部参照）：

- episodes.py    合成第一人称片段采集（任务/参数变体、手部轨迹代理、缺陷注入）
- coverage.py    Task Coverage → 细粒度 State Coverage 的状态覆盖图
- processing.py  自动质检（静止/镜头漂移/重复摆拍）、经验密度、Yield 良率
- collection.py  被动采集 vs Coverage-aware 主动采集（子模贪心补缺口）
- benchmark.py   数据飞轮基准：覆盖率曲线、边际新状态、良率、密度

证据等级 cpu-proto：没有真实第一视角视频/手部追踪/MaxVLM，世界来自引擎既有的
点质量具身环境；它验证的是“主动补缺口 + 良率管线”相对“盲目上量”的机制优势。
"""
from __future__ import annotations

from .episodes import EpisodeRecord, make_task, collect_episode, generate_pool
from .coverage import CoverageMap, state_cells
from .processing import quality_control, experience_density, process_batch
from .collection import passive_select, active_select, flywheel_curve
from .benchmark import run_flywheel_benchmark, EXTERNAL_REFERENCE

__all__ = ["EpisodeRecord", "make_task", "collect_episode", "generate_pool",
           "CoverageMap", "state_cells", "quality_control",
           "experience_density", "process_batch", "passive_select",
           "active_select", "flywheel_curve", "run_flywheel_benchmark",
           "EXTERNAL_REFERENCE"]
