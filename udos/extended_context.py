"""
扩展上下文窗口 (v3.2.0.dev2)
==========================================
analogy, not reproduction —— 受长上下文建模思想启发的**轻量化类比实现**, 非复现。

UDOS 的 obs_encoder 是逐时间步 Linear, CTM 对历史 token 做全注意力, 因此模型
结构上**本就接受任意窗口长度 S**; 本模块在其之上提供:

    1. 位置编码扩展 (正弦 positional encoding, opt-in 默认关): 给 obs_encoder 输出
       的 token 序列加位置信号, 让注意力能区分 token 的先后;
    2. 历史截断 / 填充策略: 窗口超过 max_len 时保留最近 max_len 帧; 不足 pad_to
       时用首帧重复填充 (对齐位置索引);
    3. **W=6 默认逐位等价锚点**: use_pe=False 且窗口恰为默认 6 帧时, 直接委托
       predictor.predict_next, 与旧路径逐位一致 (不改默认 window=6)。

设计纪律: 纯推理外挂、确定性、不修改主模型权重; 默认 window=6 输出不变。
第二引擎一律称 GPM。
"""



from __future__ import annotations

import logging

logger = logging.getLogger("udos.extended_context")


import math
from typing import Optional

import torch

from .dynamics import RAW_DIM


def sinusoidal_pe(length: int, dim: int) -> torch.Tensor:
    """标准正弦位置编码 [length, dim] (偶数维 sin, 奇数维 cos)。"""
    if length < 1:
        raise ValueError("pe length 必须 >= 1")
    if dim % 2 != 0:
        raise ValueError("pe dim 需为偶数")
    pos = torch.arange(length, dtype=torch.float32).unsqueeze(1)       # [L,1]
    div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32)
                    * (-math.log(10000.0) / dim))                      # [dim/2]
    pe = torch.zeros(length, dim)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class ExtendedContextWindow:
    """在已训练 predictor 上支持更长历史窗口的推理外挂。

    Parameters
    ----------
    predictor:
        已训练 PhysicsPredictor (只读外挂)。
    max_len:
        最长保留历史帧数 (窗口超过时截断为最近 max_len)。
    default_window:
        默认窗口长度 (=6); use_pe=False 且输入恰为该长度时逐位委托旧路径。
    pe_max_len / pe_dim:
        位置编码表大小与维度 (= obs_encoder 输出 d_input)。
    """

    def __init__(self, predictor, max_len: int = 24,
                 default_window: int = 6, pe_max_len: int = 64) -> None:
        if max_len < 1:
            raise ValueError("max_len 必须 >= 1")
        if default_window < 1:
            raise ValueError("default_window 必须 >= 1")
        self.predictor = predictor
        self.max_len = int(max_len)
        self.default_window = int(default_window)
        # obs_encoder 输出维度 = CTM d_input (从模型结构探测)
        d_input = self._probe_d_input(predictor)
        self.pe_dim = d_input
        pe = sinusoidal_pe(pe_max_len, d_input)
        self.register_pe = pe

    @staticmethod
    def _probe_d_input(predictor) -> int:
        """从 obs_encoder 末层 Linear.out_features 探测 token 维度。"""
        for layer in reversed(list(predictor.obs_encoder)):
            if isinstance(layer, torch.nn.Linear):
                return int(layer.out_features)
        raise RuntimeError("无法从 obs_encoder 探测 token 维度")

    # ------------------------------------------------------------------ #
    # 历史整理: 截断 / 填充
    # ------------------------------------------------------------------ #
    def truncate(self, window: torch.Tensor) -> torch.Tensor:
        """窗口超过 max_len 时保留最近 max_len 帧; 否则原样返回。"""
        w = self._as_window(window)
        if w.size(1) > self.max_len:
            w = w[:, -self.max_len:, :]
        return w.contiguous()

    def pad(self, window: torch.Tensor, pad_to: int) -> torch.Tensor:
        """窗口不足 pad_to 时用首帧重复填充到 pad_to 帧 (仅在 pad_to>当前长度时)。"""
        w = self._as_window(window)
        cur = w.size(1)
        if pad_to <= cur:
            return w
        pad = w[:, 0:1, :].expand(w.size(0), pad_to - cur, w.size(2))
        return torch.cat([pad, w], dim=1).contiguous()

    @staticmethod
    def _as_window(window: torch.Tensor) -> torch.Tensor:
        w = torch.as_tensor(window, dtype=torch.float32)
        if w.dim() == 2:
            w = w.unsqueeze(0)
        if w.dim() != 3:
            raise ValueError("window 需为 [W,RAW] 或 [B,W,RAW]")
        if w.size(1) == 0:
            raise ValueError("window 时间维为空 (W=0)")
        if w.size(-1) != RAW_DIM:
            raise ValueError(f"window 最后一维 {w.size(-1)} 应为 {RAW_DIM}")
        return w

    # ------------------------------------------------------------------ #
    # 前向: 复用 predictor 的 obs_encoder + ctm + state_decoder
    # ------------------------------------------------------------------ #
    def _forward_tokens(self, tokens: torch.Tensor,
                        scene_context: Optional[torch.Tensor]) -> torch.Tensor:
        preds, certs, sync_out, info = self.predictor.ctm(
            tokens, scene_context=scene_context)
        B, out, T = preds.shape
        per_tick = preds.transpose(1, 2).reshape(B * T, out)
        decoded = self.predictor.state_decoder(per_tick).view(
            B, T, self.predictor.raw_dim)
        return decoded[..., -1, :]   # [B, raw] = 最终 tick 预测

    @torch.no_grad()
    def predict_next(self, window: torch.Tensor,
                     scene_params: Optional[torch.Tensor] = None,
                     use_pe: bool = False, pad_to: Optional[int] = None
                     ) -> torch.Tensor:
        """扩展窗口下一状态预测。

        use_pe=False 且窗口恰为 default_window(=6) 时 => 逐位委托旧路径;
        否则先截断 (max_len), 可选填充 (pad_to), 编码后可选加位置编码再前向。
        """
        w = self._as_window(window)
        self.predictor.eval()
        # ---- W=6 默认逐位等价锚点 ----
        if not use_pe and w.size(1) == self.default_window and pad_to is None:
            return self.predictor.predict_next(w, scene_params=scene_params)
        # ---- 长路径: 截断 -> (可选) 填充 -> 编码 ----
        w = self.truncate(w)
        if pad_to is not None:
            w = self.pad(w, int(pad_to))
        h = self.predictor.obs_encoder(w)                       # [B,S,d]
        if use_pe:
            S = h.size(1)
            h = h + self.register_pe[:S].to(h.device).unsqueeze(0)
        ctx = self.predictor._resolve_context(None, scene_params)
        return self._forward_tokens(h, ctx)


__all__ = ["ExtendedContextWindow", "sinusoidal_pe"]
