"""v7.3.7 具身混合控制原型：语义层 × 动作先验（TopK 候选挑选 + 闭环再观察）。

对标公开仿真报告中「通用大模型负责语义判断/修正、VLA 策略（π0.5）提供物理
动作先验」的混合架构，在引擎既有的 6 维状态契约 [px,py,pz,vx,vy,vz] 上做
**CPU 合成动力学原型**：

- env.py        点质量双臂末端的最小闭环环境（有序目标、障碍接触、扰动恢复）
- hybrid.py     MotionPrior（TopK 候选动作段）+ SemanticCritic（挑选/接管修正）
- benchmark.py  Direct / Motion-only / Hybrid 三模式对比与证据分级

证据等级：cpu-proto。外部报告中的 62.6 分、48% 成功率、14.4% 接管率、
6.25 亿 Token 等数字属于匿名第三方仿真报告，本包只作 UNVERIFIED 参照，
不是本仓实测；LLM 语义裁判与真机/MuJoCo 为资源闸门。
"""
from __future__ import annotations

from .env import EnvTask, PointMassEnv
from .hybrid import (Candidate, MotionPrior, SemanticCritic, run_episode)
from .benchmark import EXTERNAL_REFERENCE, run_suite

__all__ = ["EnvTask", "PointMassEnv", "Candidate", "MotionPrior",
           "SemanticCritic", "run_episode", "run_suite", "EXTERNAL_REFERENCE"]
