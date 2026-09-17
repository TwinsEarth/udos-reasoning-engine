"""
场景参数桥 (v5.4.5)
===================
把 GPM"场景记录员"记录的、仅凭观测窗口无法唯一确定的隐藏物理参数, 从 PCE
PhysicsScene 规范地提取为训练过场景门的主预测员可消费的 4 维张量。

槽位定义与 udos.dynamics 完全一致 (SCENE_PARAM_NAMES):
    (v0, accel_a, spring_omega, other_v2)

两条显式承载通道 (协议向后兼容, 不改动 PCEParser):
  1. scene.metadata["scene_params"]: dict(按槽名) 或 list/tuple(固定顺序, 长度 4);
  2. PhysicalToken.attributes[槽名]: 场景级常量, 取时间序最早出现值。
优先级: metadata > attributes > 默认 0.0 (与合成数据空槽位布局一致)。

完全没有任何场景信息时返回 None, 调用方据此走场景盲旧路径 (逐位兼容)。
所有数值强制有限: NaN/inf 显式 ValueError, 不静默传播 (与 PhysicsPredictor
.predict_next 的 scene_params 校验同纪律)。

本模块只做承载/提取/校验; 从观测窗口"估计"隐藏参数在 v5.4.8 的
scene_estimator, reason 主链接线在 v5.4.6。
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from .dynamics import SCENE_PARAM_DIM, SCENE_PARAM_NAMES

logger = logging.getLogger("udos.scene_bridge")

METADATA_KEY = "scene_params"
SLOT_METADATA = "metadata"
SLOT_ATTRIBUTES = "attributes"
SLOT_DEFAULT = "default"


@dataclass
class SceneParams:
    """提取结果。

    values:          [1, SCENE_PARAM_DIM] float32, 全部有限
    source:          本场景最高优先级来源 ("metadata" | "attributes")
    per_slot_source: 每个槽位的实际来源 ("metadata"/"attributes"/"default")
    """

    values: torch.Tensor
    source: str
    per_slot_source: Tuple[str, ...]

    @property
    def present(self) -> bool:
        """是否至少有一个槽位携带真实场景信息 (非全默认)。"""
        return any(s != SLOT_DEFAULT for s in self.per_slot_source)

    def as_tensor(self) -> torch.Tensor:
        return self.values


def _finite_float(name: str, value: Any) -> float:
    """把标量转 float 并强制有限; 不可转/NaN/inf 一律 ValueError。"""
    try:
        f = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"场景参数 {name}={value!r} 无法解析为有限数值") from exc
    if not math.isfinite(f):
        raise ValueError(f"场景参数 {name}={value!r} 非有限值 (NaN/inf 被拒绝)")
    return f


def _from_metadata(meta_container: Any) -> Optional[Dict[str, float]]:
    """解析 metadata['scene_params'], 返回 {槽名: 有限float}; 缺省返回 None。"""
    if not isinstance(meta_container, dict) or METADATA_KEY not in meta_container:
        return None
    raw = meta_container[METADATA_KEY]
    out: Dict[str, float] = {}
    if isinstance(raw, dict):
        for slot in SCENE_PARAM_NAMES:        # 只认标准槽名, 未知键忽略
            if slot in raw and raw[slot] is not None:
                out[slot] = _finite_float(slot, raw[slot])
        return out
    if isinstance(raw, (list, tuple)):
        if len(raw) != SCENE_PARAM_DIM:
            raise ValueError(
                f"metadata.{METADATA_KEY} 列表长度须为 {SCENE_PARAM_DIM}"
                f"(槽位顺序 {SCENE_PARAM_NAMES}), 实际 {len(raw)}")
        for slot, val in zip(SCENE_PARAM_NAMES, raw):
            out[slot] = _finite_float(slot, val)
        return out
    raise ValueError(
        f"metadata.{METADATA_KEY} 须为 dict 或长度 {SCENE_PARAM_DIM} 的列表, "
        f"实际类型 {type(raw).__name__}")


def _from_attributes(scene) -> Dict[str, float]:
    """从时间序最早出现的 token 属性中收集标准槽位 (场景级常量)。"""
    out: Dict[str, float] = {}
    for tok in scene.timeline():
        attrs = getattr(tok, "attributes", None) or {}
        for slot in SCENE_PARAM_NAMES:
            if slot in out:
                continue
            if slot in attrs and attrs[slot] is not None:
                out[slot] = _finite_float(slot, attrs[slot])
    return out


def extract_scene_params(scene) -> Optional[SceneParams]:
    """从 PhysicsScene 提取 4 维隐藏场景参数; 无任何场景信息返回 None。"""
    meta_vals = _from_metadata(getattr(scene, "metadata", None)) or {}
    attr_vals = _from_attributes(scene)

    values: List[float] = []
    per_slot: List[str] = []
    top_source: Optional[str] = None
    for slot in SCENE_PARAM_NAMES:
        if slot in meta_vals:
            values.append(meta_vals[slot]); per_slot.append(SLOT_METADATA)
            top_source = SLOT_METADATA
        elif slot in attr_vals:
            values.append(attr_vals[slot]); per_slot.append(SLOT_ATTRIBUTES)
            if top_source is None:
                top_source = SLOT_ATTRIBUTES
        else:
            values.append(0.0); per_slot.append(SLOT_DEFAULT)

    if top_source is None:
        return None

    tensor = torch.tensor([values], dtype=torch.float32)
    if not bool(torch.isfinite(tensor).all()):
        # 理论上 _finite_float 已拦截, 这里做收口防御 (不静默传播)
        raise ValueError("场景参数张量存在非有限值")
    return SceneParams(values=tensor, source=top_source,
                       per_slot_source=tuple(per_slot))


def scene_params_tensor(scene) -> Optional[torch.Tensor]:
    """便捷封装: 只返回 [1, D] 张量或 None (供 predictor.rollout 直接使用)。"""
    sp = extract_scene_params(scene)
    return None if sp is None else sp.as_tensor()
