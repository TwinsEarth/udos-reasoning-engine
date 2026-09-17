"""
GPM 生成式物理推演引擎 (机制对齐 SakanaAI/doc-to-lora)
======================================================
上游链路 (src/ctx_to_lora/modeling/):
    ctx_encoder (冻结LLM编码文档)
      -> aggregator.Perceiver / Idefics2Perceiver (latent-query 瓶颈)
      -> HyperLoRA.head 生成各层各模块的 LoRA A/B
      -> lora_forward 以前向补丁方式注入 base model (不改权重)
      -> internalize() 缓存 LoRA / reset() 无损移除
      -> 多 chunk 经 combine_lora 做平均聚合 (chunking + mean-pooling)

本模块把"文档"替换为"物理场景 (PCE-Format)", 用轻量编码器替代冻结 LLM,
完整保留上述机制, 可在 CPU 上端到端跑通; 接入真实 LLM 时仅需替换
PhysicsContextEncoder 与 base model。

相对用户初版 Demo 的关键修复:
    1) 初版直接改写 module.weight.data 再靠减法回滚, 存在浮点累积误差且
       与 D2L 真实实现不符; 这里改为 D2L 同款 *前向补丁*:
       out = Linear(x) + B(Ax)*scaling, 原始权重永不修改, reset 零误差。
    2) 初版场景向量维度与超网络输入维度错配; 由编码器统一投影解决。
    3) 补齐 A/B 初始化约定 (scaler_B=0 => 初始扰动为 0, 对齐 LoRA 惯例)。
"""



from __future__ import annotations

import logging

logger = logging.getLogger("udos.gpm_engine")


import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pce_format import PhysicsScene, PhysicsSceneEncoder


# ---------------------------------------------------------------------------
# 1) 物理场景编码器 (替代 D2L 的冻结 LLM Context Encoder)
# ---------------------------------------------------------------------------
class PhysicsContextEncoder(nn.Module):
    """物理 Token 序列 -> 上下文特征 [B, S, feature_dim]。"""

    def __init__(self, d_model: int, feature_dim: int, attr_keys: Optional[Sequence[str]] = None):
        super().__init__()
        self.token_encoder = PhysicsSceneEncoder(d_model, attr_keys=attr_keys)
        self.refine = nn.Sequential(
            nn.Linear(d_model, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
        )
        self.feature_dim = feature_dim

    def forward(self, scenes: List[PhysicsScene]) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回 features [B,S,F] 与 mask [B,S] (按各自 token 数, 此处等长场景)。"""
        seqs = [self.token_encoder(s.tokens) for s in scenes]
        max_s = max(s.size(0) for s in seqs)
        B, Fdim = len(seqs), self.feature_dim
        feats = torch.zeros(B, max_s, self.token_encoder.projector[-1].normalized_shape[0])
        mask = torch.zeros(B, max_s)
        for i, s in enumerate(seqs):
            feats[i, : s.size(0)] = s
            mask[i, : s.size(0)] = 1.0
        return self.refine(feats), mask


# ---------------------------------------------------------------------------
# 2) Perceiver 瓶颈 (对齐 aggregator.Perceiver / Idefics2Perceiver)
# ---------------------------------------------------------------------------
class _CrossAttnBlock(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.q_norm = nn.LayerNorm(dim)
        self.kv_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)

    def forward(self, q: torch.Tensor, kv: torch.Tensor,
                kv_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        key_padding = None if kv_mask is None else kv_mask == 0
        a, _ = self.attn(self.q_norm(q), self.kv_norm(kv), self.kv_norm(kv),
                         key_padding_mask=key_padding, need_weights=False)
        return q + a


class _SelfAttnBlock(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ff = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 4 * dim),
                                nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        x = x + self.ff(x)
        return x


class PerceiverBottleneck(nn.Module):
    """
    latent queries 交叉注意力压缩任意长度上下文, 再用输出 queries
    解码出 n_layers*n_modules*r 个 LoRA 槽位向量。
    """

    def __init__(self, feature_dim: int, latent_size: int, n_latents: int,
                 n_output_queries: int, heads: int = 4, num_blocks: int = 2):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(n_latents, latent_size) * 0.02)
        self.in_proj = (nn.Linear(feature_dim, latent_size)
                        if feature_dim != latent_size else nn.Identity())
        self.cross_in = _CrossAttnBlock(latent_size, heads)
        self.self_blocks = nn.ModuleList(
            _SelfAttnBlock(latent_size, heads) for _ in range(num_blocks))
        # decoder: 输出 query 从瓶颈 latent 中取信息
        self.out_queries = nn.Parameter(
            torch.randn(n_output_queries, latent_size) * 0.02)
        self.cross_out = _CrossAttnBlock(latent_size, heads)

    def forward(self, features: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = features.size(0)
        kv = self.in_proj(features)
        lat = self.latents.unsqueeze(0).expand(B, -1, -1).contiguous()
        lat = self.cross_in(lat, kv, mask)
        for blk in self.self_blocks:
            lat = blk(lat)
        q = self.out_queries.unsqueeze(0).expand(B, -1, -1).contiguous()
        out = self.cross_out(q, lat, None)
        return out  # [B, n_output_queries, latent_size]


# ---------------------------------------------------------------------------
# 3) LoRA 数据结构
# ---------------------------------------------------------------------------
@dataclass
class LoRASet:
    """
    一次场景内化生成的全部 LoRA 矩阵。
    AB[module] = (A, B):
        A: [n_layers, r, d_in]
        B: [n_layers, r, d_out]
    """

    AB: Dict[str, Tuple[torch.Tensor, torch.Tensor]]
    layer_indices: List[int]
    scaling: float = 1.0
    scene_id: Optional[str] = None

    def num_params(self) -> int:
        return sum(a.numel() + b.numel() for a, b in self.AB.values())

    def num_bytes_fp32(self) -> int:
        return self.num_params() * 4

    def module_names(self) -> List[str]:
        return list(self.AB.keys())


# ---------------------------------------------------------------------------
# 4) 超网络主体 (对齐 HyperLoRA: pre-head -> 归一化 -> head 出 A/B)
# ---------------------------------------------------------------------------
@dataclass
class GPMConfig:
    feature_dim: int = 128
    latent_size: int = 128
    n_latents: int = 16                 # Perceiver 瓶颈 latent 数
    lora_rank: int = 8
    target_modules: Tuple[str, ...] = ("down_proj", "gate_proj", "up_proj")
    layer_indices: Tuple[int, ...] = (0, 1, 2, 3)
    num_pre_head_layers: int = 2
    heads: int = 4
    # 对齐 D2L: scaler_B 训练初始为 0 以保证零扰动; 未训练的演示/推理场景
    # 置 False (scaler_B=1) 以便观察到生成 LoRA 的实际注入效果
    init_scaler_b_zero: bool = True
    # 每个目标模块的输入/输出宽度 (由 base model 自动探测, 也可手动指定)
    dims: Optional[Dict[str, Tuple[int, int]]] = None


class PhysicsHypernetwork(nn.Module):
    """物理场景 -> LoRA A/B, 单次前向、零反向传播。"""

    def __init__(self, config: GPMConfig):
        super().__init__()
        self.cfg = config
        self.n_layers = len(config.layer_indices)
        self.r = config.lora_rank
        self.modules_ = list(config.target_modules)
        # 模块维度: 未显式给定则假定方形、宽度=latent_size*4 (典型 FFN 宽度)
        self.dims = config.dims or {m: (config.latent_size * 4,
                                        config.latent_size * 4)
                                    for m in self.modules_}
        # 对齐 D2L: head 输出槽位宽度需容纳 A(d_in)+B(d_out) 拼接,
        # 取所有目标模块 (d_in+d_out) 的最大值
        self.max_io = max(di + do for di, do in self.dims.values())

        n_out_q = self.n_layers * len(self.modules_) * self.r
        self.aggregator = PerceiverBottleneck(
            config.feature_dim, config.latent_size, config.n_latents,
            n_out_q, heads=config.heads)

        # pre-head 处理
        pre = []
        d = config.latent_size
        for _ in range(config.num_pre_head_layers):
            pre += [nn.Linear(d, d), nn.GELU()]
        self.pre_head = nn.Sequential(*pre)
        # 一次性输出 A(d_in) + B(d_out), 再按模块切分 (对齐 _to_lora_dict)
        self.head = nn.Linear(d, self.max_io)

        # 对齐 D2L: scaler_A=1, scaler_B=0 (训练初始 LoRA 增量为 0)
        init_b = torch.zeros if config.init_scaler_b_zero else torch.ones
        self.scaler_A = nn.ParameterDict({
            m: nn.Parameter(torch.ones(self.n_layers, self.r, 1))
            for m in self.modules_})
        self.scaler_B = nn.ParameterDict({
            m: nn.Parameter(init_b(self.n_layers, self.r, 1))
            for m in self.modules_})
        nn.init.normal_(self.head.weight,
                        std=0.5 / math.sqrt(config.latent_size + self.max_io * self.r))
        # 持久化场景编码器 (首次 forward 时按属性键构建并注册为子模块)
        self.ctx_encoder: Optional[PhysicsContextEncoder] = None

    def _get_encoder(self, scenes: List[PhysicsScene]) -> PhysicsContextEncoder:
        """
        持久化场景编码器: 首次按出现的属性键创建并注册为子模块 (随
        state_dict 保存/训练); 之后复用, 保证同分布场景重复内化的确定性。
        仅当出现编码器未覆盖的新属性键时才重建 (维度随之扩展)。
        """
        keys = sorted({k for s in scenes for t in s.tokens for k in t.attributes})
        existing = self.ctx_encoder
        if existing is not None and set(keys).issubset(
                set(existing.token_encoder.attr_keys)):
            return existing
        enc = PhysicsContextEncoder(self.cfg.feature_dim, self.cfg.feature_dim,
                                    attr_keys=keys)
        self.ctx_encoder = enc  # 重复赋值会自动替换已注册的同名子模块
        return enc

    def generate_weights(self, features: torch.Tensor,
                         mask: Optional[torch.Tensor] = None,
                         scene_ids: Optional[List[str]] = None) -> LoRASet:
        B = features.size(0)
        z = self.aggregator(features, mask)                 # [B, n_out_q, latent]
        z = z.view(B, self.n_layers, len(self.modules_), self.r, -1)
        z = self.pre_head(z)
        z = z / z.norm(dim=-1, keepdim=True).clamp_min(1e-8)   # 对齐归一化
        flat = self.head(z)                                # [B,L,M,r,max_io]

        lora: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        for mi, m in enumerate(self.modules_):
            d_in, d_out = self.dims[m]
            slot = flat[:, :, mi]                          # [B,L,r,max_io]
            A = slot[..., :d_in].clone()
            Bm = slot[..., d_in:d_in + d_out].clone()
            A = A * self.scaler_A[m]
            Bm = Bm * self.scaler_B[m]
            # 去掉 batch 维 (单场景): [L,r,d_in] / [L,r,d_out]
            lora[m] = (A.squeeze(0), Bm.squeeze(0))
        return LoRASet(lora, list(self.cfg.layer_indices),
                       scene_id=(scene_ids[0] if scene_ids and B == 1 else None))

    def forward(self, scene: PhysicsScene,
                encoder: Optional[PhysicsContextEncoder] = None) -> LoRASet:
        enc = encoder or self._get_encoder([scene])
        feats, mask = enc([scene])
        return self.generate_weights(feats, mask, scene_ids=[scene.scene_id])

    @torch.no_grad()
    def scene_embedding(self, scene: PhysicsScene) -> torch.Tensor:
        """
        v2.0.0: 导出场景级嵌入 [latent_size] (Perceiver 输出 query 的均值),
        作为 CTM 场景条件化输入, 让 GPM 的场景表示真正进入 CTM 推演。
        """
        enc = self._get_encoder([scene])
        feats, mask = enc([scene])
        z = self.aggregator(feats, mask)          # [1, n_out_q, latent]
        return z.mean(dim=1).squeeze(0)           # [latent]

    # ---- 分块 + 平均聚合 (对齐 D2L chunking + combine_lora mean-pooling) ----
    @torch.no_grad()
    def forward_chunked(self, chunks: List[PhysicsScene],
                        weights: Optional[Sequence[float]] = None,
                        encoder: Optional[PhysicsContextEncoder] = None) -> LoRASet:
        assert len(chunks) > 0
        enc = encoder or self._get_encoder(chunks)
        sets = []
        for ch in chunks:
            feats, mask = enc([ch])
            sets.append(self.generate_weights(feats, mask,
                                              scene_ids=[ch.scene_id]))
        return aggregate_loras(sets, weights=weights,
                               scene_id=f"chunked:{len(chunks)}")


@torch.no_grad()
def aggregate_loras(sets: Sequence[LoRASet],
                    weights: Optional[Sequence[float]] = None,
                    scene_id: Optional[str] = None) -> LoRASet:
    """
    多场景/多分块 LoRA 聚合。D2L 的依据: 高维空间中不同上下文近似正交,
    平均不会互相破坏。支持加权平均 (weights)。
    """
    base = sets[0]
    if weights is None:
        w = [1.0 / len(sets)] * len(sets)
    else:
        s = float(sum(weights))
        w = [float(x) / s for x in weights]
    merged: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    for m in base.module_names():
        A = sum(wi * s.AB[m][0] for wi, s in zip(w, sets))
        Bm = sum(wi * s.AB[m][1] for wi, s in zip(w, sets))
        merged[m] = (A, Bm)
    return LoRASet(merged, base.layer_indices, scene_id=scene_id)


# ---------------------------------------------------------------------------
# 5) 前向补丁式 LoRA 注入器 (对齐 lora_forward / apply_lora_to_layers)
# ---------------------------------------------------------------------------
class LoRAInjector:
    """
    不修改原始权重: 给目标 nn.Linear 打前向补丁
        y = Linear(x) + scaling * B(Ax)
    reset() 恢复原始 forward, 零误差, 对齐 D2L internalize/reset。
    """

    def __init__(self, base_model: nn.Module, scaling: float = 1.0):
        self.model = base_model
        self.scaling = scaling
        self._patched: List[Tuple[nn.Module, object]] = []
        self.active: Optional[LoRASet] = None

    @staticmethod
    def _make_patched_forward(linear: nn.Linear, A: torch.Tensor,
                              B: torch.Tensor, scaling: float):
        orig_forward = linear.forward

        def patched(x, *args, **kwargs):
            y = orig_forward(x, *args, **kwargs)
            h = torch.einsum("rd,...sd->...sr", A, x.to(A.dtype))
            delta = torch.einsum("rd,...sr->...sd", B, h)
            return y + scaling * delta.to(y.dtype)

        return patched

    def _targets(self) -> List[Tuple[int, str, nn.Linear]]:
        """返回 (layer_idx, module_name, linear_module)。"""
        found = []
        layers = getattr(self.model, "layers", None)
        if layers is None:
            return found
        for li in range(len(layers)):
            layer = layers[li]
            for name, mod in layer.named_modules():
                if isinstance(mod, nn.Linear) and name.split(".")[-1] in (
                        "down_proj", "gate_proj", "up_proj", "q_proj",
                        "k_proj", "v_proj", "o_proj"):
                    found.append((li, name.split(".")[-1], mod))
        return found

    def inject(self, lora: LoRASet) -> nn.Module:
        self.reset()
        for li, mname, mod in self._targets():
            if mname not in lora.AB or li not in lora.layer_indices:
                continue
            slot = lora.layer_indices.index(li)
            A, B = lora.AB[mname]
            mod.forward_orig = mod.forward          # type: ignore[attr-defined]
            mod.forward = self._make_patched_forward(
                mod, A[slot], B[slot], self.scaling)
            mod.patched_forward = True              # type: ignore[attr-defined]
            self._patched.append((mod, mod.forward_orig))
        self.active = lora
        return self.model

    def reset(self) -> nn.Module:
        for mod, orig in self._patched:
            mod.forward = orig
            if hasattr(mod, "forward_orig"):
                del mod.forward_orig
            if hasattr(mod, "patched_forward"):
                del mod.patched_forward
        self._patched.clear()
        self.active = None
        return self.model


# ---------------------------------------------------------------------------
# 6) 轻量基座模型 (演示/测试用; 真实场景替换为 Llama/Qwen 等 HF 模型)
# ---------------------------------------------------------------------------
class TinyMLPLayer(nn.Module):
    """一层 Transformer 风格 MLP: gate/up (扩张) -> SiLU -> down (缩回)。"""

    def __init__(self, hidden: int):
        super().__init__()
        inner = hidden * 4
        self.gate_proj = nn.Linear(hidden, inner, bias=False)
        self.up_proj = nn.Linear(hidden, inner, bias=False)
        self.down_proj = nn.Linear(inner, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TinyBlock(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.mlp = TinyMLPLayer(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(x)


class TinyBaseModel(nn.Module):
    """最小可注入基座: layers[*].mlp.{gate,up,down}_proj。"""

    def __init__(self, hidden: int = 128, n_layers: int = 4):
        super().__init__()
        self.hidden = hidden
        self.layers = nn.ModuleList(
            [TinyBlock(hidden) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def infer_dims_from_model(base_model: nn.Module,
                          target_modules: Sequence[str]
                          ) -> Dict[str, Tuple[int, int]]:
    """
    遍历基座 layers[*], 自动探测每个目标线性层的 (in_features, out_features),
    供 GPMConfig.dims 使用, 避免人工写错 A/B 形状。
    """
    dims: Dict[str, Tuple[int, int]] = {}
    layers = getattr(base_model, "layers", None)
    if layers is None or len(layers) == 0:
        return dims
    for modname, mod in layers[0].named_modules():
        leaf = modname.split(".")[-1]
        if leaf in target_modules and isinstance(mod, nn.Linear):
            dims[leaf] = (mod.in_features, mod.out_features)
    return dims
