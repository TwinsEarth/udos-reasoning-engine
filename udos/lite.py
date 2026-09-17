"""
模型轻量化: 幅值剪枝 / 动态 INT8 量化 / 蒸馏 (v2.7.0.dev3, 全部 opt-in)
============================================================================
三个独立工具, **默认全量模型不变**, 需显式调用:
    * MagnitudePruner: 全局权重幅值剪枝 + 可恢复 mask;
    * DynamicQuantizer: torch.ao.quantization 动态量化 (Linear -> INT8, 仅 CPU);
    * DistillationTrainer: teacher-student 知识蒸馏 (软目标 + 硬目标 MSE 混合损失)。

纪律:
    * 剪枝/量化不引入新可学参数; 量化后模型仅推理、不可再训练;
    * 蒸馏学生为更小 CTM (d_model 减半), 单独训练, 不动 teacher;
    * 三者均不改默认预测路径, 不调用时旧模型逐位一致。

第二引擎一律称 GPM。
"""



from __future__ import annotations

import logging

logger = logging.getLogger("udos.lite")


import dataclasses
from typing import Dict, Optional

import torch
import torch.nn as nn

from .ctm_engine import CTMConfig
from .training import PhysicsPredictor, set_seed


# --------------------------------------------------------------------------- #
# 全局幅值剪枝
# --------------------------------------------------------------------------- #
class MagnitudePruner:
    """按全局权重幅值阈值把 |w| <= threshold 的 Linear 权重置零 (存原权重可恢复)。"""

    def __init__(self) -> None:
        self._originals: Dict[str, torch.Tensor] = {}
        self._masks: Dict[str, torch.Tensor] = {}

    @staticmethod
    def _linear_weights(model) -> Dict[str, nn.Parameter]:
        out = {}
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Linear):
                out[name + ".weight"] = mod.weight
        return out

    def prune(self, model, sparsity: float) -> float:
        """全局幅值剪枝。sparsity∈[0,1): 置零最小幅值的那部分权重。返回实际稀疏度。"""
        if not 0.0 <= sparsity < 1.0:
            raise ValueError("sparsity 需在 [0,1)")
        self._originals, self._masks = {}, {}
        weights = self._linear_weights(model)
        all_w = torch.cat([w.detach().reshape(-1) for w in weights.values()])
        thresh = torch.quantile(all_w.abs().float(), float(sparsity))
        with torch.no_grad():
            for name, w in weights.items():
                self._originals[name] = w.detach().clone()
                mask = (w.detach().abs() > thresh).to(w.dtype)
                self._masks[name] = mask
                w.mul_(mask)
        return self.sparsity_ratio(model)

    @torch.no_grad()
    def unprune(self, model) -> None:
        """恢复剪枝前权重。"""
        weights = self._linear_weights(model)
        for name, orig in self._originals.items():
            weights[name].copy_(orig)
        self._originals, self._masks = {}, {}

    @staticmethod
    def sparsity_ratio(model) -> float:
        """实际零参数占比 (Linear 权重)。"""
        weights = [w.detach().float() for w in
                   MagnitudePruner._linear_weights(model).values()]
        nz = sum(int((w == 0).sum()) for w in weights)
        total = sum(w.numel() for w in weights)
        return nz / max(total, 1)


# --------------------------------------------------------------------------- #
# 动态 INT8 量化 (仅 CPU)
# --------------------------------------------------------------------------- #
class DynamicQuantizer:
    """torch.ao.quantization 动态量化 Linear -> INT8。量化后仅推理。"""

    @staticmethod
    def quantize(model) -> nn.Module:
        if next(model.parameters()).is_cuda:
            raise RuntimeError("动态量化仅支持 CPU")
        return torch.ao.quantization.quantize_dynamic(
            model, {nn.Linear}, dtype=torch.qint8)


# --------------------------------------------------------------------------- #
# 知识蒸馏 (teacher -> 小学生 CTM)
# --------------------------------------------------------------------------- #
class DistillationTrainer:
    """teacher-student 蒸馏: loss = α·MSE(student, teacher.detach) + (1-α)·MSE(student, y)。
    回归任务下用 teacher 单步预测作软目标 (KL 的回归类比), 真值作硬目标。"""

    def __init__(self, alpha: float = 0.7, lr: float = 3e-3) -> None:
        self.alpha = float(alpha)
        self.lr = float(lr)

    def distill(self, teacher: PhysicsPredictor,
                student_config: CTMConfig, train_data,
                epochs: int = 20, batch_size: int = 64,
                seed: int = 0) -> PhysicsPredictor:
        set_seed(seed)
        student = PhysicsPredictor(student_config, scene_param_dim=4)
        opt = torch.optim.AdamW(student.parameters(), lr=self.lr,
                                weight_decay=1e-4)
        teacher.eval()
        losses = []
        for ep in range(epochs):
            ep_loss, nb = 0.0, 0
            for batch in train_data.batches(batch_size, shuffle=True, seed=seed + ep):
                xb, pb, yb = batch
                with torch.no_grad():
                    t_pred = teacher(xb, scene_params=pb)[2]
                s_pred = student(xb, scene_params=pb)[2]
                soft = ((s_pred - t_pred) ** 2).mean()
                hard = ((s_pred - yb[:, 0, :]) ** 2).mean()
                loss = self.alpha * soft + (1.0 - self.alpha) * hard
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(student.parameters(), 5.0)
                opt.step()
                ep_loss += float(loss.detach())
                nb += 1
            losses.append(ep_loss / max(nb, 1))
        student.eval()
        student.distill_history = losses
        return student


# --------------------------------------------------------------------------- #
# 知识蒸馏 v2 (温度软标签 + 中间特征匹配 + 学生自动架构)  (v3.3.0.dev2)
# --------------------------------------------------------------------------- #
def auto_student_config(cfg: CTMConfig, scale: float) -> CTMConfig:
    """按 scale 自动收缩学生 CTM 架构 (d_model 减半=0.5 / 四分之一=0.25)。
    仅收缩宽度类维度并保证下限; heads/memory_length/iterations 等结构超参不变。"""
    if scale not in (0.5, 0.25):
        raise ValueError(f"student_scale 仅支持 0.5 / 0.25, 收到 {scale}")
    low = 4
    s = lambda v: max(low, int(round(v * scale)))
    out = dataclasses.replace(cfg)
    out.d_model = s(cfg.d_model)
    out.d_input = s(cfg.d_input)
    out.n_synch_out = s(cfg.n_synch_out)
    out.n_synch_action = s(cfg.n_synch_action)
    out.nlm_hidden = s(cfg.nlm_hidden)
    out.out_dims = s(cfg.out_dims)
    return out


class DistillationTrainerV2:
    """
    v2 知识蒸馏 (analogy, not reproduction):
        loss = α·(T²·MSE(student_last, teacher_last/T))          # 温度软标签
             + (1-α)·MSE(student_last, y)                       # 硬标签
             + feature_weight·MSE(student_ticks, teacher_ticks/T)  # 中间特征匹配
    回归任务下无 logits/softmax, 温度 T 作用于 teacher 表示的"软化"强度,
    T=1 时软标签项退化为 v1 的 MSE(student, teacher) (T²=1 因子)。
    学生架构由 auto_student_config 自动 d_model 减半/四分之一。
    """

    def __init__(self, alpha: float = 0.7, temperature: float = 4.0,
                 feature_weight: float = 0.5, student_scale: float = 0.5,
                 lr: float = 3e-3) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha 需在 [0,1]")
        if temperature <= 0:
            raise ValueError("temperature 需 > 0 (T=0 见 edge 节点守卫)")
        self.alpha = float(alpha)
        self.temperature = float(temperature)
        self.feature_weight = float(feature_weight)
        self.student_scale = float(student_scale)
        self.lr = float(lr)

    def distill(self, teacher: PhysicsPredictor,
                base_config: CTMConfig, train_data,
                epochs: int = 20, batch_size: int = 64,
                seed: int = 0) -> PhysicsPredictor:
        set_seed(seed)
        student_cfg = auto_student_config(base_config, self.student_scale)
        student = PhysicsPredictor(student_cfg, scene_param_dim=4)
        opt = torch.optim.AdamW(student.parameters(), lr=self.lr,
                                weight_decay=1e-4)
        teacher.eval()
        T = self.temperature
        T2 = T * T
        losses = []
        for ep in range(epochs):
            ep_loss, nb = 0.0, 0
            for batch in train_data.batches(batch_size, shuffle=True,
                                           seed=seed + ep):
                xb, pb, yb = batch
                with torch.no_grad():
                    t_ticks, _, t_last, _ = teacher(xb, scene_params=pb)
                s_ticks, _, s_last, _ = student(xb, scene_params=pb)
                soft = T2 * ((s_last - t_last.detach() / T) ** 2).mean()
                hard = ((s_last - yb[:, 0, :]) ** 2).mean()
                # 中间特征匹配: 逐 tick 解码轨迹 (raw_dim 公共空间, 维度兼容)
                feat = ((s_ticks - t_ticks.detach() / T) ** 2).mean()
                loss = (self.alpha * soft + (1.0 - self.alpha) * hard
                        + self.feature_weight * feat)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(student.parameters(), 5.0)
                opt.step()
                ep_loss += float(loss.detach())
                nb += 1
            losses.append(ep_loss / max(nb, 1))
        student.eval()
        student.distill_history = losses
        student.distill_v2 = {
            "temperature": T, "feature_weight": self.feature_weight,
            "student_scale": self.student_scale,
        }
        return student


# --------------------------------------------------------------------------- #
# 结构化剪枝 v2 (通道 / 注意力头级, 而非逐元素幅值)  (v3.3.0.dev3)
# --------------------------------------------------------------------------- #
class StructuredPrunerV2:
    """
    结构化剪枝 v2 (analogy, not reproduction):
        与 v1 MagnitudePruner 的逐元素幅值剪枝不同, 这里按**输出通道 (Linear 权重的一行)**
        的 L2 范数排序, 整行 (含对应 bias) 一起置零 —— 即一个通道要么全保留、要么全删。
        head_dim 给定时按注意力头分组: 整组 (头) 的所有通道一起删。
    剪枝后可调用 fine_tune 做少量恢复 (mask 冻结, 仅保留通道可学)。
    说明: 本实现保留形状 (整行置零) 以跨层安全; 真正删除通道/重建形状留作被否决候选。
    """

    def __init__(self, prune_ratio: float = 0.3, head_dim: Optional[int] = None):
        if not 0.0 <= prune_ratio < 1.0:
            raise ValueError("prune_ratio 需在 [0,1)")
        self.prune_ratio = float(prune_ratio)
        self.head_dim = head_dim
        self._originals: Dict[str, torch.Tensor] = {}
        self._masks: Dict[str, torch.Tensor] = {}
        self._pruned_channels: Dict[str, int] = {}

    @staticmethod
    def _linear_weights(model) -> Dict[str, nn.Parameter]:
        return {name + ".weight": mod.weight
                for name, mod in model.named_modules()
                if isinstance(mod, nn.Linear)}

    def _linear_modules(self, model) -> Dict[str, nn.Linear]:
        return {name: mod for name, mod in model.named_modules()
                if isinstance(mod, nn.Linear)}

    @torch.no_grad()
    def prune(self, model) -> Dict[str, float]:
        modules = self._linear_modules(model)
        self._originals, self._masks, self._pruned_channels = {}, {}, {}
        for name, lin in modules.items():
            w = lin.weight.detach()                     # [out, in]
            out_f = w.size(0)
            row_norm = w.float().norm(dim=1)           # 每通道 L2
            if self.head_dim is not None and out_f % self.head_dim == 0:
                n_heads = out_f // self.head_dim
                head_norm = row_norm.view(n_heads, self.head_dim).mean(dim=1)
                n_prune_heads = int(round(self.prune_ratio * n_heads))
                prune_heads = torch.topk(head_norm, n_prune_heads,
                                          largest=False).indices
                keep = torch.ones(out_f, dtype=torch.bool)
                for h in prune_heads:
                    keep[h * self.head_dim:(h + 1) * self.head_dim] = False
                mask = keep.to(w.dtype).unsqueeze(1)        # [out, 1]
                n_zero = int((~keep).sum())
            else:
                n_prune = int(round(self.prune_ratio * out_f))
                prune_rows = torch.topk(row_norm, n_prune,
                                        largest=False).indices
                mask = torch.ones(out_f, 1, dtype=w.dtype)
                mask[prune_rows] = 0.0
                n_zero = n_prune
            self._originals[name + ".weight"] = w.clone()
            if lin.bias is not None:
                self._originals[name + ".bias"] = lin.bias.detach().clone()
            self._masks[name] = mask
            self._pruned_channels[name] = n_zero
            lin.weight.mul_(mask)
            if lin.bias is not None:
                lin.bias.mul_(mask.squeeze(-1))
        return self.report(model)

    def report(self, model) -> Dict[str, float]:
        modules = self._linear_modules(model)
        total_ch, zero_ch = 0, 0
        total_w, zero_w = 0, 0
        for name, lin in modules.items():
            w = lin.weight.detach().float()
            total_ch += w.size(0)
            zero_ch += int((w.abs().sum(dim=1) == 0).sum())   # 整行为 0 => 通道剪枝
            total_w += w.numel()
            zero_w += int((w == 0).sum())
        return {
            "channel_sparsity": round(zero_ch / max(total_ch, 1), 4),
            "weight_sparsity": round(zero_w / max(total_w, 1), 4),
            "pruned_channels": zero_ch, "total_channels": total_ch,
        }

    @torch.no_grad()
    def unprune(self, model) -> None:
        modules = self._linear_modules(model)
        for name, lin in modules.items():
            if name + ".weight" in self._originals:
                lin.weight.copy_(self._originals[name + ".weight"])
                if lin.bias is not None and name + ".bias" in self._originals:
                    lin.bias.copy_(self._originals[name + ".bias"])
        self._originals, self._masks, self._pruned_channels = {}, {}, {}

    @torch.no_grad()
    def _reapply_masks(self, model) -> None:
        modules = self._linear_modules(model)
        for name, lin in modules.items():
            if name in self._masks:
                lin.weight.mul_(self._masks[name])
                if lin.bias is not None:
                    lin.bias.mul_(self._masks[name].squeeze(-1))

    def fine_tune(self, model, train_data, epochs: int = 3,
                  batch_size: int = 64, lr: float = 1e-3) -> list:
        """mask 冻结下少量微调恢复 (仅保留通道可学, 剪枝通道保持为 0)。"""
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        losses = []
        model.train()
        for ep in range(epochs):
            ep_loss, nb = 0.0, 0
            for batch in train_data.batches(batch_size, shuffle=True,
                                           seed=ep):
                xb, pb, yb = batch
                opt.zero_grad()
                _, _, last, _ = model(xb, scene_params=pb)
                loss = ((last - yb[:, 0, :]) ** 2).mean()
                loss.backward()
                opt.step()
                self._reapply_masks(model)   # 剪枝通道保持为 0
                ep_loss += float(loss.detach())
                nb += 1
            losses.append(ep_loss / max(nb, 1))
        model.eval()
        return losses
