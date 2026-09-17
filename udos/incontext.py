"""
In-Context Learning (v3.2.0.dev5)
==========================================
analogy, not reproduction —— 受 PhysBrain 262K 长上下文 / in-context learning 启发的
**轻量化类比实现**, 非复现:

UDOS 是 ~52k 参数合成动力学小模型, 不预训练大模型。这里把 few-shot 示例窗口 +
任务描述向量 + 查询窗口沿时间维拼接成一条更长的上下文序列, 送入已训练 predictor
(经 ExtendedContextWindow 的位置编码), 验证以下工程事实:

    1. 上下文拼接形状正确 (n_shots 个示例窗口 + 查询窗口);
    2. 任务描述向量经 scene_params 通道注入;
    3. 0/1/3-shot 推理可运行 (精度如实测量, 不保证提升; 收益不稳 opt-in);
    4. 空示例守卫、与 extended_context 接口一致。

设计纪律: 纯推理外挂、确定性、不修改主模型权重; 默认 0-shot 与旧路径一致。
第二引擎一律称 GPM。
"""



from __future__ import annotations

import logging

logger = logging.getLogger("udos.incontext")


from typing import List, Optional, Sequence

import torch

from .dynamics import RAW_DIM
from .extended_context import ExtendedContextWindow


class InContextLearner:
    """few-shot 示例 + 任务描述 + 查询 -> 扩展上下文预测。

    Parameters
    ----------
    predictor:
        已训练 PhysicsPredictor (只读外挂)。
    max_len:
        上下文最大长度 (= 示例总帧数 + 查询帧数 上限)。
    use_pe:
        拼接上下文是否加位置编码 (默认 True; 跨示例边界需要位置信号区分)。
    """

    def __init__(self, predictor, max_len: int = 64, use_pe: bool = True) -> None:
        self.predictor = predictor
        self.ecw = ExtendedContextWindow(predictor, max_len=max_len)
        self.use_pe = bool(use_pe)

    # ------------------------------------------------------------------ #
    # 上下文拼接
    # ------------------------------------------------------------------ #
    @staticmethod
    def _as_win(w: torch.Tensor) -> torch.Tensor:
        t = torch.as_tensor(w, dtype=torch.float32)
        if t.dim() != 2:
            raise ValueError("示例/查询窗口需为 [W, RAW_DIM]")
        if t.size(-1) != RAW_DIM:
            raise ValueError(f"窗口最后一维 {t.size(-1)} 应为 {RAW_DIM}")
        return t

    def build_context(self, query_window: torch.Tensor,
                      examples: Optional[Sequence[torch.Tensor]] = None
                      ) -> torch.Tensor:
        """拼接 few-shot 示例窗口 + 查询窗口 -> [1, (k+1)*W, RAW_DIM]。"""
        q = self._as_win(query_window)
        if examples:
            ex = [self._as_win(e) for e in examples]
            if any(e.shape != q.shape for e in ex):
                raise ValueError("所有示例窗口须与查询窗口同形状")
            ctx = torch.cat(ex + [q], dim=0)
        else:
            ctx = q
        return ctx.unsqueeze(0).contiguous()

    # ------------------------------------------------------------------ #
    # 推理
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def predict(self, query_window: torch.Tensor,
                examples: Optional[Sequence[torch.Tensor]] = None,
                task_desc: Optional[torch.Tensor] = None,
                scene_params: Optional[torch.Tensor] = None) -> torch.Tensor:
        """拼接上下文并预测下一状态 [RAW_DIM]。

        task_desc: 任务描述向量 (>=4 维, 取前 4 维映射到 scene_params 通道)。
        示例为空时退化为 0-shot (仅查询窗口)。
        """
        ctx = self.build_context(query_window, examples)   # [1, S, 6]
        sp = scene_params
        if task_desc is not None:
            td = torch.as_tensor(task_desc, dtype=torch.float32).reshape(-1)
            if td.numel() < 4:
                raise ValueError("task_desc 至少需 4 维 (映射到 scene_params)")
            td4 = td[:4].unsqueeze(0)
            sp = td4 if sp is None else sp
        out = self.ecw.predict_next(ctx, scene_params=sp, use_pe=self.use_pe)
        return out[0]

    # ------------------------------------------------------------------ #
    # 0/1/3-shot 精度测量 (合成查询, 如实记录)
    # ------------------------------------------------------------------ #
    def shot_mse(self, query_windows: torch.Tensor, query_targets: torch.Tensor,
                 example_pool: Sequence[torch.Tensor], n_shots: int,
                 task_desc: Optional[torch.Tensor] = None,
                 scene_params: Optional[torch.Tensor] = None) -> float:
        """对一批查询测 n-shot 下一状态 MSE。

        query_windows [N,W,6], query_targets [N,6]; example_pool 为示例窗口列表。
        n_shots=0 => 纯查询 (0-shot)。
        """
        if n_shots < 0:
            raise ValueError("n_shots 必须 >= 0")
        if len(example_pool) == 0 and n_shots > 0:
            raise ValueError("n_shots>0 但示例池为空")
        total = n = 0.0
        for i in range(query_windows.size(0)):
            examples = None
            if n_shots > 0:
                examples = [example_pool[(i + j) % len(example_pool)]
                            for j in range(n_shots)]
            sp_i = scene_params[i:i+1] if scene_params is not None else None
            pred = self.predict(query_windows[i], examples=examples,
                                task_desc=task_desc, scene_params=sp_i)
            total += float(((pred - query_targets[i]) ** 2).sum())
            n += query_targets.size(1)
        return total / max(n, 1)


__all__ = ["InContextLearner"]
