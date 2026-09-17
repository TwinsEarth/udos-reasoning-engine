"""
ICM 事件级切分与三流对齐 (v3.4.0.dev1)
================================================================
analogy, not reproduction —— 受 "WALL/WM 世界模型: 把连续轨迹按事件边界切成
可变长度因果片段, 再在事件粒度对齐 物理token/动作/结果三流" 思想启发的
**轻量化类比实现**, 非复现:

UDOS 的演示记忆 (v3.4.0) 默认按固定 6 帧窗口注册。真实物理轨迹里, 一次
"事件" 往往对应一段**动力学一致的片段** (匀速段、碰撞瞬间、弹簧半个周期)。
本模块提供:

    1. EventSegmenter   : 在连续轨迹上做变点检测, 找出事件边界索引;
    2. ThreeStreamAligner: 沿事件边界把 (物理token 状态流 / 动作流 / 结果流)
                           切成对齐的事件片段, 供 ICM 按事件粒度注册演示。

设计纪律:
    * 纯函数式、确定性、零参数、不调主模型;
    * 变点信号 = 速度增量幅度 (|v_t − v_{t-1}|), 自适应阈值 (中位数 + k·MAD);
    * 匀速/平滑段不产生边界 (空边界合法); 短序列/空序列显式守卫;
    * 三流对齐严格等长: 一个事件片段内 states/actions/results 长度一致;
    * 第二引擎一律称 GPM; analogy, not reproduction。
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

import torch

from .dynamics import RAW_DIM

logger = logging.getLogger("udos.icm_events")


def _as_traj(traj: torch.Tensor) -> torch.Tensor:
    t = torch.as_tensor(traj, dtype=torch.float32)
    if t.dim() != 2 or t.size(-1) != RAW_DIM:
        raise ValueError(f"轨迹需为 [T,{RAW_DIM}], 收到 shape={tuple(t.shape)}")
    if not bool(torch.isfinite(t).all()):
        raise ValueError("轨迹含 NaN/inf")
    return t


class EventSegmenter:
    """变点检测器: 在状态轨迹上找动力学突变的事件边界。

    变点信号 = 相邻帧速度差的 L2 范数 a_t = ||v_t − v_{t-1}|| (t=1..T-1)。
    阈值自适应: thresh = median(a) + k_mad * MAD(a), MAD = median(|a − median|)。
    匀速段 a≈0 => 不触发; 碰撞/剧烈变速段 a 尖峰 => 触发边界。

    Parameters
    ----------
    k_mad : 阈值倍数 (默认 3.0; 越大越保守、边界越少)。
    min_gap : 相邻边界最小间隔 (帧), 防止单点抖动出多个边界。
    fixed_thresh : 给定则用固定阈值, 否则用自适应中位数+MAD。
    """

    def __init__(self, k_mad: float = 3.0, min_gap: int = 1,
                 fixed_thresh: Optional[float] = None) -> None:
        if k_mad <= 0:
            raise ValueError("k_mad 须 > 0")
        if min_gap < 1:
            raise ValueError("min_gap 须 >= 1")
        if fixed_thresh is not None and fixed_thresh < 0:
            raise ValueError("fixed_thresh 须 >= 0")
        self.k_mad = float(k_mad)
        self.min_gap = int(min_gap)
        self.fixed_thresh = fixed_thresh

    # ------------------------------------------------------------------ #
    # 变点信号
    # ------------------------------------------------------------------ #
    def change_signal(self, traj: torch.Tensor) -> torch.Tensor:
        """返回速度变化幅度 [T-1]。"""
        t = _as_traj(traj)
        vel = t[:, 3:6]
        dv = vel[1:] - vel[:-1]              # [T-1, 3]
        return dv.norm(dim=-1)               # [T-1]

    def _threshold(self, signal: torch.Tensor) -> float:
        if self.fixed_thresh is not None:
            return float(self.fixed_thresh)
        if signal.numel() == 0:
            return float("inf")
        med = float(signal.median())
        mad = float((signal - med).abs().median())
        return med + self.k_mad * mad

    # ------------------------------------------------------------------ #
    # 检测
    # ------------------------------------------------------------------ #
    def detect(self, traj: torch.Tensor) -> List[int]:
        """返回事件边界索引列表 (升序)。

        边界 b 表示 "在第 b 帧发生了动力学突变", 即信号 a_b 超阈值。
        信号长度 T-1 对应索引 t=1..T-1; 边界取信号超阈值处的 (t+0) 帧索引。
        无突变时返回空列表 (合法, 表示整条轨迹是单一事件)。
        """
        t = _as_traj(traj)
        if t.size(0) < 3:
            return []
        sig = self.change_signal(t)           # [T-1], 索引 i 对应帧 i+1
        thresh = self._threshold(sig)
        above = (sig > thresh).nonzero(as_tuple=False).flatten().tolist()
        # min_gap 抑制相邻抖动
        boundaries: List[int] = []
        for i in above:
            frame_idx = i + 1                 # 信号第 i 个差对应帧 (i+1)
            if boundaries and frame_idx - boundaries[-1] < self.min_gap:
                continue
            boundaries.append(frame_idx)
        return boundaries


class ThreeStreamAligner:
    """沿事件边界把三流切成对齐事件片段。

    三流 (由同一条 [T,6] 状态轨迹派生, 长度严格对齐):
        * 物理 token 状态流 states : [T, 6]
        * 动作流 actions            : [T-1, 6] = states[1:] − states[:-1]
        * 结果流 results            : [T-1, 6] = states[1:]

    给定 EventSegmenter 检出的边界, 把轨迹切成连续片段
    [0, b1), [b1, b2), ..., [b_{m-1}, T); 对每片段抽取三流且长度一致。
    """

    def __init__(self, segmenter: Optional[EventSegmenter] = None) -> None:
        self.segmenter = segmenter or EventSegmenter()

    @staticmethod
    def derive_streams(states: torch.Tensor
                       ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """由状态流派生动作流/结果流 (纯函数, 长度自洽)。"""
        s = _as_traj(states)
        if s.size(0) < 2:
            raise ValueError("状态流至少需 2 帧才能派生动作/结果流")
        actions = s[1:] - s[:-1]
        results = s[1:].clone()
        return s, actions, results

    def split(self, states: torch.Tensor) -> List[dict]:
        """把一条状态轨迹切成事件片段列表, 每片段含三流对齐张量。

        返回 list[dict], 键: start/end/kind/states/actions/results。
        无边界时返回单片段 (整条轨迹)。空状态流显式守卫。
        """
        s = _as_traj(states)
        if s.size(0) < 2:
            raise ValueError("状态流至少需 2 帧")
        boundaries = self.segmenter.detect(s)
        cuts = [0] + list(boundaries) + [s.size(0)]
        segments: List[dict] = []
        for a, b in zip(cuts[:-1], cuts[1:]):
            if b - a < 2:
                # 片段不足 2 帧无法形成 (动作/结果) 一对, 跳过 (合法退化)
                continue
            seg_states = s[a:b]
            seg_actions = seg_states[1:] - seg_states[:-1]
            seg_results = seg_states[1:].clone()
            segments.append({
                "start": int(a), "end": int(b), "kind": "event",
                "states": seg_states.contiguous(),
                "actions": seg_actions.contiguous(),
                "results": seg_results.contiguous(),
            })
        if not segments:
            # 全程过短/全部被 min_gap 抑制: 退化为整条单片段
            seg_actions = s[1:] - s[:-1]
            segments.append({
                "start": 0, "end": int(s.size(0)), "kind": "whole",
                "states": s.contiguous(), "actions": seg_actions.contiguous(),
                "results": s[1:].contiguous()})
        return segments

    def align_episodes(self, states: torch.Tensor):
        """把事件片段转成 (window, result) 对齐对, 供 ICM DemonstrationEpisode 注册。

        每片段用其前 window 帧作输入、下一帧作结果 (window=2 默认, 可按片段
        实际长度截断); 返回 list[(window_tensor[w,6], result[6], start)]。
        """
        segs = self.split(states)
        pairs = []
        for seg in segs:
            st = seg["states"]
            if st.size(0) < 3:
                continue
            w = min(6, st.size(0) - 1)
            pairs.append((st[:w], st[w], seg["start"]))
        return pairs


__all__ = ["EventSegmenter", "ThreeStreamAligner"]
