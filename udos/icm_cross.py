"""
跨本体演示归一化后 ICL (v3.4.0.dev2)
================================================================
analogy, not reproduction —— 借鉴 Human-as-Humanoid "源本体动作 -> 目标本体动作"
思想的**轻量化类比实现**, 非复现:

v3.4.0 的 ICM 演示记忆默认假设演示与查询**同本体** (同为 6 维 RAW)。真实 ICM 里,
演示常由**不同本体**采集 (DOF 数不同、控制频率不同)。本模块复用 `udos/retargeting.py`
的成熟件, 把源本体演示**归一化到目标本体**后再注册进 ICM:

    源本体演示轨迹 [T_src, src_dof]
        --(时间重采样)--> [T_tgt, src_dof]      (ActionRetargeter.resample)
        --(DOF 映射+限幅)--> [T_tgt, tgt_dof=6] (ActionRetargeter.retarget)
        --(切窗+注册)--> DemonstrationEpisode 入 DemonstrationMemory

设计纪律:
    * **复用不重造**: DOF 映射 / 时间重采样 / 关节限幅全部委托 retargeting.py;
    * 纯推理外挂、零梯度、不改主模型权重;
    * DOF 不匹配 / 空序列 / dof=0 源本体显式 ValueError (沿 retargeting 契约);
    * 第二引擎一律称 GPM; analogy, not reproduction。
"""

from __future__ import annotations

import logging
from typing import List, Optional

import torch

from .dynamics import RAW_DIM
from .retargeting import MorphologyConfig, ActionRetargeter
from .icm import DemonstrationEpisode, DemonstrationMemory, ICMAggregator

logger = logging.getLogger("udos.icm_cross")


class CrossEmbodimentICM:
    """跨本体演示归一化 + ICM 条件化预测。

    Parameters
    ----------
    predictor : 已训练 PhysicsPredictor (只读外挂)。
    source : 源本体 MorphologyConfig (演示采集本体, dof>=1)。
    target : 目标本体 MorphologyConfig (推理本体; dof 必须 == RAW_DIM=6)。
    """

    TGT_DOF = RAW_DIM

    def __init__(self, predictor, source: MorphologyConfig,
                 target: MorphologyConfig) -> None:
        if target.dof != self.TGT_DOF:
            raise ValueError(
                f"目标本体 dof 须 == {self.TGT_DOF} (RAW_DIM), 收到 {target.dof}")
        self.predictor = predictor
        self.source = source
        self.target = target
        self.retargeter = ActionRetargeter(source, target)
        self.memory = DemonstrationMemory()
        self.aggregator = ICMAggregator(predictor)

    # ------------------------------------------------------------------ #
    # 源本体演示 -> 目标本体归一化 -> 注册
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def register_trajectory(self, src_traj: torch.Tensor, window: int = 6,
                            src_freq: Optional[float] = None,
                            dst_freq: Optional[float] = None,
                            kind: str = "linear",
                            scene_params: Optional[torch.Tensor] = None,
                            label: str = "cross_embodiment") -> int:
        """把一条源本体状态轨迹 [T_src, src_dof] 归一化后注册为演示事件。

        步骤: 时间重采样 (可选) -> DOF 映射+限幅到 6 维 -> 滑窗切 (window, result)
        对 -> 逐个注册 DemonstrationEpisode 并缓存残差。返回注册条数。
        """
        traj = torch.as_tensor(src_traj, dtype=torch.float32)
        if traj.dim() != 2 or traj.size(-1) != self.source.dof:
            raise ValueError(
                f"源轨迹需为 [T,{self.source.dof}], 收到 {tuple(traj.shape)}")
        if traj.size(0) < 3:
            raise ValueError("源轨迹至少需 3 帧才能切窗")
        # 时间重采样 (频率不同时对齐)
        if src_freq is not None or dst_freq is not None:
            traj = self.retargeter.resample(traj, src_freq=src_freq,
                                            dst_freq=dst_freq, kind=kind)
        # DOF 映射 + 限幅到目标本体 (6 维)
        tgt = self.retargeter.retarget(traj)               # [T_tgt, 6]
        n = 0
        for s in range(tgt.size(0) - window):
            in_win = tgt[s:s + window]
            result = tgt[s + window]
            ep = DemonstrationEpisode(in_win, result, kind=label,
                                      scene_params=scene_params)
            self.memory.register(ep)
            self.aggregator.cache_residual(ep, scene_params=scene_params)
            n += 1
        return n

    # ------------------------------------------------------------------ #
    # 预测 (委托 ICMAggregator)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def predict(self, query_window: torch.Tensor, k: int = 3,
                scene_params: Optional[torch.Tensor] = None) -> torch.Tensor:
        """在归一化后的跨本体演示库上做演示条件化预测。"""
        return self.aggregator.predict(query_window, memory=self.memory,
                                       k=k, scene_params=scene_params)

    def __len__(self) -> int:
        return self.memory.size


__all__ = ["CrossEmbodimentICM"]
