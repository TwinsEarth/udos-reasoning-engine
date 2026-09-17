"""
推理缓存 (v2.5.0)
=================
InferenceCache: 基于输入张量哈希 + 模型参数哈希的 LRU 结果缓存。
默认关闭 (enabled=False), opt-in 激活。开启后命中必须返回与未命中完全相同输出。

缓存 key 构成:
    - 输入张量内容哈希 (raw_seq 字节 + shape)
    - 可选 scene_params 哈希
    - 模型参数哈希 (state_dict 所有张量字节的 SHA-256)
      模型更新/换件后自动失效 (旧 key 不会命中新模型)

LRU 淘汰: 超过 maxsize 时弹出最久未访问项。线程安全 (标准库 threading.Lock)。

零重依赖: 仅 hashlib + collections + threading + torch。
"""



from __future__ import annotations

import logging

logger = logging.getLogger("udos.cache")


import hashlib
import threading
from collections import OrderedDict
from typing import Any, Optional, Tuple

import torch


def _tensor_digest(t: Optional[torch.Tensor]) -> str:
    """对张量内容做确定性 SHA-256 (detach + contiguous + bytes)。"""
    if t is None:
        return "none"
    t = t.detach().to(torch.float32).contiguous().cpu()
    h = hashlib.sha256()
    h.update(t.numpy().tobytes())
    h.update(str(tuple(t.shape)).encode())
    return h.hexdigest()


def _params_digest(model: torch.nn.Module) -> str:
    """对模型 state_dict 做确定性哈希; 模型权重变化 -> digest 变化。"""
    h = hashlib.sha256()
    sd = model.state_dict()
    for name in sorted(sd.keys()):
        t = sd[name].detach().to(torch.float32).contiguous().cpu()
        h.update(name.encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


class InferenceCache:
    """
    LRU 推理结果缓存 (默认关闭, opt-in)。

    用法:
        cache = InferenceCache(predictor, maxsize=128)
        cache.enabled = True          # opt-in 激活
        out = BatchPredictor(predictor).predict(seqs, cache=cache)
    """

    def __init__(self, model: Optional[torch.nn.Module] = None,
                 maxsize: int = 128):
        self._model = model
        self.maxsize = int(maxsize)
        if self.maxsize < 1:
            raise ValueError("maxsize 必须 >= 1")
        self._store: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        self._lock = threading.Lock()
        self.enabled = False           # 默认关闭, opt-in
        self._hits = 0
        self._misses = 0

    # ------------------------------------------------------------------ #
    # 模型绑定 (可在运行中更新; 旧 key 自动失效因为 digest 变了)
    # ------------------------------------------------------------------ #
    def bind_model(self, model: torch.nn.Module) -> None:
        self._model = model
        # 模型更换后清空旧缓存 (参数哈希必然变, 但清空更安全)
        with self._lock:
            self._store.clear()

    def params_digest(self) -> str:
        if self._model is None:
            return "no-model"
        return _params_digest(self._model)

    # ------------------------------------------------------------------ #
    # Key 构造
    # ------------------------------------------------------------------ #
    def make_key(self, raw_seq: torch.Tensor,
                 scene_params: Optional[torch.Tensor] = None) -> str:
        """构造缓存 key: 输入哈希 + 模型参数哈希。"""
        parts = [
            "v1",
            _tensor_digest(raw_seq),
            _tensor_digest(scene_params if scene_params is not None else None),
            self.params_digest(),
        ]
        return "|".join(parts)

    # ------------------------------------------------------------------ #
    # 读写
    # ------------------------------------------------------------------ #
    def get(self, key: str) -> Optional[torch.Tensor]:
        """命中返回克隆张量 (避免外部修改污染缓存), 未命中返回 None。"""
        if not self.enabled:
            return None
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
                self._hits += 1
                return self._store[key].clone()
            self._misses += 1
            return None

    def put(self, key: str, value: torch.Tensor) -> None:
        """写入缓存 (存克隆); 超容淘汰最久未用项。"""
        if not self.enabled:
            return
        with self._lock:
            self._store[key] = value.detach().clone().cpu()
            self._store.move_to_end(key)
            while len(self._store) > self.maxsize:
                self._store.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def __len__(self) -> int:
        return len(self._store)

    @property
    def size(self) -> int:
        return len(self._store)

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def misses(self) -> int:
        return self._misses

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return (self._hits / total) if total > 0 else 0.0

    def reset_stats(self) -> None:
        self._hits = 0
        self._misses = 0
