"""
PCE-Format: 物理世界 Token 化协议 (UDOS 数据层)
================================================
一个 PhysicalToken 描述某个物体在某个时刻的完整物理状态;
一个 PhysicsScene 是一段时间内若干 PhysicalToken 的集合 (可含约束)。

本模块同时负责把变长、异构的物理状态编码为定长向量, 作为
CTM 的时序输入与 GPM 超网络的场景输入。

注意 (相对用户初版 Demo 的修复):
    初版 Demo 直接把 3 维位置 + 若干属性拼起来送进期望 2048 维输入的
    nn.Linear, 存在维度不匹配缺陷。这里由 PhysicsSceneEncoder 统一做
    "物理量 -> 定长 d_model 向量" 的投影, 任意属性维度都可适配。
"""



from __future__ import annotations

import logging

logger = logging.getLogger("udos.pce_format")


import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn as nn

# 基础物理量维度: 位置(3) + 速度(3) + 受力(3) = 9
BASE_PHYS_DIM = 9


@dataclass
class PhysicalToken:
    """单个物理 Token: 物体 object_id 在 timestamp 时刻的状态。"""

    object_id: str
    timestamp: int
    position: Sequence[float]  # [x, y, z]
    velocity: Sequence[float] = (0.0, 0.0, 0.0)  # [vx, vy, vz]
    force: Sequence[float] = (0.0, 0.0, 0.0)  # [fx, fy, fz]
    # 额外标量物理属性 (质量、摩擦系数、温度……), 长度可任意
    attributes: Dict[str, float] = field(default_factory=dict)
    # 因果父节点: 本状态由哪些 (object_id, timestamp) 导致
    causal_parents: List[Dict[str, Any]] = field(default_factory=list)

    def base_vector(self, attr_keys: Sequence[str]) -> torch.Tensor:
        """按统一 attr_keys 顺序拼出定长原始物理向量 (未投影)。"""
        pos = list(self.position) + [0.0, 0.0, 0.0]
        vel = list(self.velocity) + [0.0, 0.0, 0.0]
        frc = list(self.force) + [0.0, 0.0, 0.0]
        vec = pos[:3] + vel[:3] + frc[:3]
        vec += [float(self.attributes.get(k, 0.0)) for k in attr_keys]
        return torch.tensor(vec, dtype=torch.float32)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PhysicsScene:
    """物理场景: 一段时间窗内的物理 Token + 静态约束。"""

    scene_id: str
    tokens: List[PhysicalToken] = field(default_factory=list)
    constraints: List[Dict[str, Any]] = field(default_factory=list)
    duration: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ---- 便捷构造 ----
    def add(self, token: PhysicalToken) -> "PhysicsScene":
        self.tokens.append(token)
        return self

    def object_ids(self) -> List[str]:
        seen: List[str] = []
        for t in self.tokens:
            if t.object_id not in seen:
                seen.append(t.object_id)
        return seen

    def timeline(self) -> List[PhysicalToken]:
        return sorted(self.tokens, key=lambda t: (t.timestamp, t.object_id))

    def attribute_keys(self) -> List[str]:
        keys: List[str] = []
        for t in self.tokens:
            for k in t.attributes:
                if k not in keys:
                    keys.append(k)
        return keys

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "duration": self.duration,
            "constraints": self.constraints,
            "metadata": self.metadata,
            "tokens": [t.to_dict() for t in self.tokens],
        }


class PCEParser:
    """PCE-Format 序列化 / 反序列化 (JSON 文本格式 .pce / .json)。"""

    @staticmethod
    def dumps(scene: PhysicsScene, indent: int = 2) -> str:
        return json.dumps(scene.to_dict(), ensure_ascii=False, indent=indent)

    @staticmethod
    def dump(scene: PhysicsScene, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(PCEParser.dumps(scene), encoding="utf-8")
        return path

    @staticmethod
    def loads(text: str) -> PhysicsScene:
        raw = json.loads(text)
        return PCEParser._from_dict(raw)

    @staticmethod
    def load(path: str | Path) -> PhysicsScene:
        return PCEParser.loads(Path(path).read_text(encoding="utf-8"))

    @staticmethod
    def _from_dict(raw: Dict[str, Any]) -> PhysicsScene:
        tokens = [PhysicalToken(**t) for t in raw.get("tokens", [])]
        return PhysicsScene(
            scene_id=raw["scene_id"],
            tokens=tokens,
            constraints=raw.get("constraints", []),
            duration=raw.get("duration", 0),
            metadata=raw.get("metadata", {}),
        )


class PhysicsSceneEncoder(nn.Module):
    """
    把异构物理 Token 编码为定长 d_model 向量序列。

    对任意属性维度自适应: 输入宽度 = 9 (位/速/力) + len(attr_keys),
    经两层 MLP 投影到 d_model。对应技术方案中
    "物理场景编码器: 将物理Token编码为超网络可处理的向量"。
    """

    def __init__(self, d_model: int, attr_keys: Optional[Sequence[str]] = None,
                 hidden: Optional[int] = None):
        super().__init__()
        self.attr_keys: List[str] = list(attr_keys or [])
        self.in_dim = BASE_PHYS_DIM + len(self.attr_keys)
        hidden = hidden or d_model
        self.projector = nn.Sequential(
            nn.Linear(self.in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
            nn.LayerNorm(d_model),
        )

    def bind_scene(self, scene: PhysicsScene) -> None:
        """场景出现了新属性键时, 重建投影层以适配 (仅初始化期使用)。"""
        keys = scene.attribute_keys()
        if keys == self.attr_keys:
            return
        if not self.attr_keys:
            self.attr_keys = keys
            in_dim = BASE_PHYS_DIM + len(keys)
            d_model = self.projector[-1].normalized_shape[0]
            hidden = self.projector[0].out_features
            self.in_dim = in_dim
            self.projector = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Linear(hidden, d_model),
                nn.LayerNorm(d_model),
            )

    def encode_token(self, token: PhysicalToken) -> torch.Tensor:
        return token.base_vector(self.attr_keys)

    def forward(self, tokens: Sequence[PhysicalToken]) -> torch.Tensor:
        """
        返回 [seq_len, d_model] 的物理 Token 嵌入序列 (按时间排序)。
        """
        ordered = sorted(tokens, key=lambda t: (t.timestamp, t.object_id))
        raw = torch.stack([t.base_vector(self.attr_keys) for t in ordered], dim=0)
        return self.projector(raw)

    def encode_scene_summary(self, scene: PhysicsScene) -> torch.Tensor:
        """场景级摘要向量 [d_model]: 时间序列均值, 供 GPM 使用。"""
        seq = self.forward(scene.tokens)
        return seq.mean(dim=0)


def collate_tokens(batch: Iterable[torch.Tensor]) -> torch.Tensor:
    """把若干 [seq_len, d] 序列堆叠为 [B, max_seq, d] (末尾零填充)。"""
    seqs = list(batch)
    max_len = max(s.size(0) for s in seqs)
    d = seqs[0].size(1)
    out = torch.zeros(len(seqs), max_len, d, dtype=seqs[0].dtype)
    for i, s in enumerate(seqs):
        out[i, : s.size(0)] = s
    return out


# ===========================================================================
# v3.4.0.dev3: PCE 物理提示词数据包 (Demonstration Prompt Packet)
# ===========================================================================
# 一个 DemonstrationPrompt 是 "可 HTTP 传输的 JSON 包", 装着:
#   * 因果块 (causal block): 多条 (输入窗口 -> 动作 -> 结果), 对应一次演示;
#   * 适配块 (adaptation block): 跨本体归一化元信息 (源/目标 DOF、频率);
# PCEPromptParser 负责在 JSON dict 与 ICM 可用的 DemonstrationEpisode 列表间
# 互转 (往返一致性), 不调主模型、不改权重。
# ---------------------------------------------------------------------------

class DemonstrationPrompt:
    """PCE 物理提示词数据包 (因果块 + 适配块), HTTP-ready JSON 序列化。

    Attributes
    ----------
    prompt_id : 包唯一 id。
    kind : 演示运动类型标签。
    causal_blocks : list[dict], 每项 {input_window[W,6], action[6], result[6]}。
    adaptation : 跨本体适配元信息 dict (source_dof/target_dof/src_freq/dst_freq);
                 无跨本体时为空 dict。
    """

    def __init__(self, prompt_id: str, kind: str = "unspecified",
                 causal_blocks: Optional[List[Dict[str, Any]]] = None,
                 adaptation: Optional[Dict[str, Any]] = None) -> None:
        if not isinstance(prompt_id, str) or not prompt_id:
            raise ValueError("prompt_id 须为非空字符串")
        self.prompt_id = str(prompt_id)
        self.kind = str(kind)
        self.causal_blocks = list(causal_blocks or [])
        self.adaptation = dict(adaptation or {})

    def add_block(self, input_window, action, result) -> None:
        """追加一个因果块; 校验形状后存为 list。"""
        iw = torch.as_tensor(input_window, dtype=torch.float32)
        ac = torch.as_tensor(action, dtype=torch.float32).reshape(-1)
        rs = torch.as_tensor(result, dtype=torch.float32).reshape(-1)
        if iw.dim() != 2 or iw.size(-1) != RAW_DIM_DYN:
            raise ValueError("input_window 需为 [W,6]")
        self.causal_blocks.append({
            "input_window": iw.tolist(),
            "action": ac.tolist(),
            "result": rs.tolist(),
        })

    def __len__(self) -> int:
        return len(self.causal_blocks)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "packet": "PCE-DemonstrationPrompt/v1",
            "prompt_id": self.prompt_id,
            "kind": self.kind,
            "adaptation": self.adaptation,
            "causal_blocks": self.causal_blocks,
        }

    def dumps(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


# 本模块原本只知道 BASE_PHYS_DIM=9; ICM 因果块用动力学 RAW_DIM=6。
# 这里延迟导入避免与顶部 PCE 类的循环依赖, 并在 add_block 里复用。
from .dynamics import RAW_DIM as RAW_DIM_DYN  # noqa: E402


class PCEPromptParser:
    """PCE 物理提示词包 <-> ICM DemonstrationEpisode 列表 互转。"""

    @staticmethod
    def from_dict(raw: Dict[str, Any]) -> "DemonstrationPrompt":
        if not isinstance(raw, dict):
            raise ValueError("提示词包须为 JSON 对象")
        if raw.get("packet") != "PCE-DemonstrationPrompt/v1":
            raise ValueError("未知/不兼容的提示词包版本")
        return DemonstrationPrompt(
            prompt_id=raw["prompt_id"],
            kind=raw.get("kind", "unspecified"),
            causal_blocks=raw.get("causal_blocks", []),
            adaptation=raw.get("adaptation", {}),
        )

    @staticmethod
    def load_episodes(prompt: "DemonstrationPrompt") -> list:
        """把提示词包解析为 DemonstrationEpisode 列表 (供 ICM 注册)。

        延迟导入 udos.icm 避免顶层循环依赖。空包返回空列表 (合法)。
        """
        from .icm import DemonstrationEpisode
        eps = []
        for blk in prompt.causal_blocks:
            iw = torch.as_tensor(blk["input_window"], dtype=torch.float32)
            ac = torch.as_tensor(blk.get("action"), dtype=torch.float32) \
                if blk.get("action") is not None else None
            rs = torch.as_tensor(blk["result"], dtype=torch.float32)
            eps.append(DemonstrationEpisode(iw, rs, action=ac,
                                             kind=prompt.kind))
        return eps

    @staticmethod
    def dumps(prompt: "DemonstrationPrompt", indent: int = 2) -> str:
        return prompt.dumps(indent=indent)


__all__ = ["PhysicalToken", "PhysicsScene", "PCEParser", "PhysicsSceneEncoder",
           "collate_tokens", "DemonstrationPrompt", "PCEPromptParser"]
