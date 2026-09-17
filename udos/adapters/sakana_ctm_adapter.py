"""
SakanaAI/continuous-thought-machines 真实代码库适配器
=====================================================
直接加载 third_party/ctm 中的上游实现 ContinuousThoughtMachine,
用于与本仓库机制对齐版 CTMPhysicsEngine 做交叉验证。

上游 backbone='none' 分支的 compute_features 按 4D 张量处理:
    [B, F, S, 1] --flatten(2)--> [B, F, S] --transpose--> [B, S, F] -> kv_proj
因此本适配器把序列输入 [B, S, F] 转置并补一维为伪图像 4D。

依赖: torch, numpy, huggingface_hub (上游 ctm.py 顶部导入)。
若依赖缺失, available() 返回 False, 调用方应优雅降级。
"""



from __future__ import annotations

import logging

logger = logging.getLogger("udos.sakana_ctm_adapter")


import importlib
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

_THIRD_PARTY = (Path(__file__).resolve().parents[2] / "third_party" / "ctm")


class SakanaCTMAdapter:
    def __init__(self,
                 iterations: int = 8,
                 d_model: int = 128,
                 d_input: int = 128,
                 heads: int = 4,
                 n_synch_out: int = 32,
                 n_synch_action: int = 32,
                 memory_length: int = 8,
                 out_dims: int = 64,
                 repo_path: Optional[str] = None):
        self.repo_path = Path(repo_path) if repo_path else _THIRD_PARTY
        self.cfg = dict(
            iterations=iterations, d_model=d_model, d_input=d_input,
            heads=heads, n_synch_out=n_synch_out,
            n_synch_action=n_synch_action, synapse_depth=1,
            memory_length=memory_length, deep_nlms=True,
            memory_hidden_dims=16, do_layernorm_nlm=False,
            backbone_type="none", positional_embedding_type="none",
            out_dims=out_dims, neuron_select_type="random-pairing",
            n_random_pairing_self=2,
        )
        self.model: Optional[Any] = None

    @staticmethod
    def available(repo_path: Optional[str] = None) -> bool:
        rp = Path(repo_path) if repo_path else _THIRD_PARTY
        if not (rp / "models" / "ctm.py").exists():
            return False
        try:
            import huggingface_hub  # noqa: F401
            import numpy  # noqa: F401
        except ImportError:
            return False
        return True

    def load(self) -> Any:
        if not str(self.repo_path) in sys.path:
            sys.path.insert(0, str(self.repo_path))
        # 清掉可能的旧缓存以保证加载的是目标仓库
        for mod in [m for m in list(sys.modules) if m.startswith("models.")]:
            sys.modules.pop(mod, None)
        ctm_mod = importlib.import_module("models.ctm")
        self.model = ctm_mod.ContinuousThoughtMachine(**self.cfg)
        self.model.eval()
        # 上游含 LazyLinear, 用 dummy 前向完成延迟初始化
        with torch.no_grad():
            dummy = torch.zeros(1, self.cfg["d_input"], 4, 1)
            _ = self.model(dummy)
        return self.model

    def _to_pseudo_image(self, seq: torch.Tensor) -> torch.Tensor:
        # [B, S, F] -> [B, F, S, 1]
        assert seq.dim() == 3
        return seq.transpose(1, 2).unsqueeze(-1)

    @torch.no_grad()
    def forward(self, seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """输入 [B,S,F], 返回上游原生 (predictions, certainties, sync_out)。"""
        if self.model is None:
            self.load()
        x = self._to_pseudo_image(seq)
        predictions, certainties, sync_out = self.model(x)
        return predictions, certainties, sync_out

    def num_params(self) -> int:
        if self.model is None:
            self.load()
        return sum(p.numel() for p in self.model.parameters())
