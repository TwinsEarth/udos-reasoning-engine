"""
批量推理引擎 (v2.5.0)
======================
BatchPredictor 包装 PhysicsPredictor, 支持:
1. 变长序列批量推理: 输入为 List[Tensor(W_i, RAW_DIM)] 时, 按窗口长度分组,
   同长序列拼 batch 一次前向, 不同长序列分别前向, 结果严格与逐笔 predict_next
   逐位一致 (atol=1e-5)。
2. 等长批量直接前向: 输入已为 [B, W, RAW_DIM] 时直接走模型前向 (CTM 本身
   已支持真批量, test_ctm_batch_independence 锁死批量==逐笔)。
3. 大 batch 分片: 单组超过 max_shard 时拆成多个分片依次前向再拼接, 控制内存。
4. 可选推理缓存: 传入 InferenceCache 时, 命中直接返回缓存张量, miss 计算后写回。

设计纪律:
- 不修改 CTM/模型前向逻辑, 不引入 attention mask (避免改变数值)。
- 变长序列通过"按长度分组"实现等价 padding: 同长组内无 padding, 跨组分别前向。
- 空 batch (空列表 / batch 维 0) 返回 [0, RAW_DIM] 空张量 (v2.6.1 加固, 不再报错);
  非法输入类型仍显式报错 (ValueError)。
- 纯 torch + 标准库, 零重依赖。
"""



from __future__ import annotations

import logging

logger = logging.getLogger("udos.batch")


from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from .training import PhysicsPredictor


class BatchPredictor:
    """包装 PhysicsPredictor 的批量推理器 (opt-in, 不改变 predict_next 行为)。"""

    def __init__(self, predictor: PhysicsPredictor, max_shard: int = 64):
        if not isinstance(predictor, PhysicsPredictor):
            raise ValueError("BatchPredictor 需要 PhysicsPredictor 实例")
        self.predictor = predictor
        self.max_shard = int(max_shard)
        if self.max_shard < 1:
            raise ValueError("max_shard 必须 >= 1")

    # ------------------------------------------------------------------ #
    # 内部工具
    # ------------------------------------------------------------------ #
    @staticmethod
    def _is_tensor_batch(x: Any) -> bool:
        return isinstance(x, torch.Tensor) and x.dim() == 3

    @staticmethod
    def _validate_single(seq: torch.Tensor, raw_dim: int) -> None:
        if not isinstance(seq, torch.Tensor):
            raise ValueError("序列元素必须是 torch.Tensor")
        if seq.dim() != 2:
            raise ValueError(
                f"单条序列需为 [W, RAW_DIM], 实际 dim={seq.dim()}")
        if seq.size(-1) != raw_dim:
            raise ValueError(
                f"序列最后一维 {seq.size(-1)} 应为 {raw_dim}")

    def _shard_forward(self, batch: torch.Tensor,
                       scene_params: Optional[torch.Tensor],
                       guard: bool) -> torch.Tensor:
        """对一个等长 batch 做前向, 必要时分片。返回 [N, RAW_DIM]。"""
        n = batch.size(0)
        if n <= self.max_shard:
            return self.predictor.predict_next(
                batch, scene_params=scene_params, guard=guard)
        outs: List[torch.Tensor] = []
        for start in range(0, n, self.max_shard):
            end = min(start + self.max_shard, n)
            sp = None
            if scene_params is not None:
                sp = scene_params[start:end]
            outs.append(self.predictor.predict_next(
                batch[start:end], scene_params=sp, guard=guard))
        return torch.cat(outs, dim=0)

    # ------------------------------------------------------------------ #
    # 公开入口
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def predict(self,
                sequences,
                scene_params: Optional[torch.Tensor] = None,
                guard: bool = False,
                cache: Optional[Any] = None
                ) -> torch.Tensor:
        """
        批量下一状态预测。

        参数:
            sequences:
                - List[Tensor(W_i, RAW_DIM)]: 变长序列列表 (可不同 W);
                - Tensor [B, W, RAW_DIM]: 等长批量 (直接前向)。
            scene_params: 可选 [B, P] 或 List[Tensor(P,)]; 与 sequences 对齐。
            guard: 是否过 PredictionGuard (与 predict_next 同语义)。
            cache: 可选 InferenceCache; 命中返回缓存, miss 计算后写回。

        返回: Tensor [N, RAW_DIM], N = len(sequences) 或 B。
        """
        pred = self.predictor
        raw_dim = pred.raw_dim

        # ---- 空 batch 守卫 (v2.6.1: 返回 [0, RAW_DIM] 空张量, 不再报错) ----
        if isinstance(sequences, (list, tuple)):
            if len(sequences) == 0:
                return torch.empty(0, raw_dim, dtype=torch.float32)
        elif self._is_tensor_batch(sequences):
            if sequences.size(0) == 0:
                return torch.empty(0, sequences.size(-1), dtype=torch.float32)
        else:
            raise ValueError(
                "sequences 需为 List[Tensor(W,RAW_DIM)] 或 Tensor [B,W,RAW_DIM]")

        # ---- 解析 scene_params ----
        sp_list: Optional[List[torch.Tensor]] = None
        sp_tensor: Optional[torch.Tensor] = None
        if scene_params is not None:
            if isinstance(scene_params, torch.Tensor):
                sp_tensor = scene_params
                if sp_tensor.dim() == 1:
                    sp_tensor = sp_tensor.unsqueeze(0)
            else:
                sp_list = [torch.as_tensor(s, dtype=torch.float32)
                           for s in scene_params]

        # ---- 情况 A: 已经是等长 [B,W,RAW] 张量 ----
        if self._is_tensor_batch(sequences):
            batch = sequences.float()
            if batch.size(-1) != raw_dim:
                raise ValueError(
                    f"batch 最后一维 {batch.size(-1)} 应为 {raw_dim}")
            # 缓存: 等长张量逐行缓存
            if cache is not None and cache.enabled:
                results = self._predict_with_cache(
                    batch, sp_tensor, guard, cache)
            else:
                results = self._shard_forward(batch, sp_tensor, guard)
            return results

        # ---- 情况 B: List[Tensor(W_i, RAW)] 变长 ----
        seqs: List[torch.Tensor] = [s.float() for s in sequences]
        for s in seqs:
            self._validate_single(s, raw_dim)
        n = len(seqs)
        # 解析对齐的 scene_params
        if sp_list is not None:
            if len(sp_list) != n:
                raise ValueError(
                    f"scene_params 数量 {len(sp_list)} 与序列数 {n} 不一致")
            sp_tensor = torch.stack(sp_list, dim=0)
        elif sp_tensor is not None and sp_tensor.size(0) != n:
            raise ValueError(
                f"scene_params 数量 {sp_tensor.size(0)} 与序列数 {n} 不一致")

        # 按窗口长度分组, 记录原始索引
        groups: Dict[int, List[int]] = OrderedDict()
        for i, s in enumerate(seqs):
            w = s.size(0)
            groups.setdefault(w, []).append(i)

        out = torch.empty(n, raw_dim, dtype=torch.float32)
        for w, idxs in groups.items():
            batch_w = torch.stack([seqs[i] for i in idxs], dim=0)
            sp_w = (sp_tensor[idxs] if sp_tensor is not None else None)
            if cache is not None and cache.enabled:
                res_w = self._predict_group_with_cache(
                    batch_w, sp_w, guard, cache)
            else:
                res_w = self._shard_forward(batch_w, sp_w, guard)
            for slot, row in zip(idxs, range(res_w.size(0))):
                out[slot] = res_w[row]
        return out

    # ------------------------------------------------------------------ #
    # 缓存集成 (opt-in, 仅 cache.enabled 时激活)
    # ------------------------------------------------------------------ #
    def _predict_with_cache(self, batch: torch.Tensor,
                            sp: Optional[torch.Tensor],
                            guard: bool,
                            cache) -> torch.Tensor:
        """等长 [B,W,RAW] 张量逐行缓存。"""
        n = batch.size(0)
        out = torch.empty(n, batch.size(-1), dtype=torch.float32)
        miss_idx: List[int] = []
        miss_sp: List[torch.Tensor] = []
        for i in range(n):
            s = batch[i:i + 1]
            si = sp[i:i + 1] if sp is not None else None
            key = cache.make_key(s, si)
            hit = cache.get(key)
            if hit is not None:
                out[i] = hit[0]
            else:
                miss_idx.append(i)
                miss_sp.append(si)
        if miss_idx:
            mb = batch[miss_idx]
            msp = (torch.cat(miss_sp, dim=0) if sp is not None else None)
            res = self._shard_forward(mb, msp, guard)
            for slot, row in zip(miss_idx, range(res.size(0))):
                s = batch[slot:slot + 1]
                si = sp[slot:slot + 1] if sp is not None else None
                cache.put(cache.make_key(s, si), res[row:row + 1].clone())
                out[slot] = res[row]
        return out

    def _predict_group_with_cache(self, batch_w: torch.Tensor,
                                  sp_w: Optional[torch.Tensor],
                                  guard: bool,
                                  cache) -> torch.Tensor:
        return self._predict_with_cache(batch_w, sp_w, guard, cache)


def predict_batch(predictor: PhysicsPredictor,
                  sequences,
                  scene_params: Optional[torch.Tensor] = None,
                  guard: bool = False,
                  cache: Optional[Any] = None,
                  max_shard: int = 64) -> torch.Tensor:
    """便捷函数: 一次性构造 BatchPredictor 并批量推理。"""
    return BatchPredictor(predictor, max_shard=max_shard).predict(
        sequences, scene_params=scene_params, guard=guard, cache=cache)
