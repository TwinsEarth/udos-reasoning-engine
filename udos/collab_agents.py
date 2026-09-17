"""能力注册表 —— Agent-as-Tool (dev1, v4.4.0) collab_agents.py
======================================================================
对应蓝图图8: Main Agent 只看能力接口; 每个被注册能力是一个带 **Tool Schema** 的
Agent-as-Tool:
    name / input_schema / output_schema / confidence∈[0,1] / error_type 枚举。
Expert(Tool) 内部封装规划/检索/测试; 主调度只看 I/O, 降低主 agent 上下文负担。

把 UDOS 既有能力注册为工具: CTM 推演、GPM、SFM 空间、PWM 想象、分层神经控制、
curriculum、self_train、self_evolution、wla 动作。

设计纪律 (与全工程一致):
    * 纯注册/调度, 无可训练参数, 零梯度, 确定性, opt-in。
    * 工具 fn 为**纯函数** (CPU 合成机制类比), 不改主 predictor 52191 参数。
    * input 不符合 schema / fn 抛错 => 统一封装为 {output, confidence, error_type},
      不把异常直接炸穿主调度 (error_type 枚举化)。
    * 空 / 非法显式 ValueError。
analogy, not reproduction —— Tool Schema 思想对标 MCP function-calling, 不实现 MCP 线上协议。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("udos.collab_agents")

# error_type 枚举 (图8): 工具失败的结构化分类, 供上游路由/重试决策
ERROR_TYPES = ("none", "invalid_input", "unsupported", "internal", "timeout",
               "low_confidence")


def _confidence(v: float) -> float:
    c = float(v)
    if not (0.0 <= c <= 1.0):
        raise ValueError(f"confidence 须在 [0,1], 实际 {c}")
    return c


@dataclass
class ToolSpec:
    """一个 Agent-as-Tool 的稳定接口契约 (图8)。"""

    name: str
    description: str
    input_schema: Dict[str, str]          # {字段: 类型说明}
    output_schema: Dict[str, str]         # {字段: 类型说明}
    confidence: float = 1.0               # 该工具默认置信度
    error_type: str = "none"              # 正常态; 运行时失败会改写
    fn: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None
    # 能力标签 (供 swarm 能力发现): 如 ["reasoning","planning"]
    tags: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("ToolSpec.name 须为非空字符串")
        if self.error_type not in ERROR_TYPES:
            raise ValueError(f"error_type 须在 {ERROR_TYPES}")
        self.confidence = _confidence(self.confidence)
        if not isinstance(self.input_schema, dict) or not self.input_schema:
            raise ValueError("input_schema 须为非空 dict")
        if not isinstance(self.output_schema, dict) or not self.output_schema:
            raise ValueError("output_schema 须为非空 dict")

    def schema(self) -> Dict[str, Any]:
        """对外暴露的纯接口 (不含 fn) —— 主调度/peer 只看这段。"""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": dict(self.input_schema),
            "output_schema": dict(self.output_schema),
            "confidence": self.confidence,
            "tags": list(self.tags),
        }


class ToolResult(Dict[str, Any]):
    """工具调用结构化结果: output + confidence + error_type (图8)。"""

    @staticmethod
    def ok(output: Dict[str, Any], confidence: float = 1.0) -> "ToolResult":
        r = ToolResult(output=output, confidence=_confidence(confidence),
                       error_type="none")
        return r

    @staticmethod
    def fail(error_type: str, message: str,
             confidence: float = 0.0) -> "ToolResult":
        if error_type not in ERROR_TYPES:
            raise ValueError(f"error_type 须在 {ERROR_TYPES}")
        return ToolResult(output={"error": message},
                          confidence=_confidence(confidence),
                          error_type=error_type)


class CapabilityRegistry:
    """能力注册表: name -> ToolSpec。主调度/peer 只看 schema。"""

    def __init__(self) -> None:
        self._tools: Dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if not isinstance(spec, ToolSpec):
            raise ValueError("register 需 ToolSpec")
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec:
        if name not in self._tools:
            raise ValueError(f"未注册的能力: {name}")
        return self._tools[name]

    def has(self, name: str) -> bool:
        return name in self._tools

    def names(self) -> List[str]:
        return list(self._tools.keys())

    def describe(self) -> List[Dict[str, Any]]:
        return [t.schema() for t in self._tools.values()]

    # -- swarm 能力发现: 按标签找能处理某类任务的 peer ---------------- #
    def discover(self, tags: List[str]) -> List[ToolSpec]:
        """返回命中任一给定标签的工具 (能力发现, 供 mesh 委派)。"""
        need = set(tags)
        out = [t for t in self._tools.values() if need & set(t.tags)]
        # 确定性排序: 名字字典序
        return sorted(out, key=lambda t: t.name)

    # -- 调用: 校验输入 -> 调 fn -> 结构化封装 (含 error_type) ---------- #
    def call(self, name: str, payload: Dict[str, Any]) -> ToolResult:
        spec = self.get(name)
        if not isinstance(payload, dict):
            return ToolResult.fail("invalid_input", "payload 须为 dict")
        missing = [k for k in spec.input_schema if k not in payload]
        if missing:
            return ToolResult.fail("invalid_input",
                                   f"缺输入字段: {missing}")
        if spec.fn is None:
            return ToolResult.fail("unsupported",
                                   f"能力 {name} 未挂载可执行 fn (仅 schema)")
        try:
            out = spec.fn(payload)
        except (ValueError, TypeError) as e:
            return ToolResult.fail("invalid_input", str(e))
        except Exception as e:  # 工具内部错 => 5xx 类, 不炸穿主调度
            logger.exception("能力 %s 执行失败", name)
            return ToolResult.fail("internal", str(e))
        if not isinstance(out, dict):
            return ToolResult.fail("internal", "工具输出须为 dict")
        conf = out.get("confidence", spec.confidence)
        return ToolResult.ok(out, confidence=float(conf))


# -- 默认能力集: 把 UDOS 既有能力注册为 Agent-as-Tool --------------------- #
# 这里的 fn 是 CPU 合成机制类比: 纯函数把合成输入映射到合成输出, 用于在不加载重模型
# 的前提下跑通 star/chain/mesh 协作骨架; 真实推理仍走各自 udos 模块 (opt-in)。
def _noop_reasoning(payload: Dict[str, Any]) -> Dict[str, Any]:
    q = str(payload.get("query", ""))
    return {"answer": f"ctm:{q}", "horizon": int(payload.get("horizon", 1)),
            "confidence": 0.9}


def _noop_scene(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {"scene_id": str(payload.get("scene_id", "s0")),
            "gpm_compressed": True, "confidence": 0.85}


def _noop_space(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {"region": str(payload.get("query_region", "occupancy")),
            "confidence": 0.8}


def _noop_imagine(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {"predicted_n": int(payload.get("n_steps", 2)),
            "consistent": True, "confidence": 0.78}


def _noop_control(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {"command_dim": 6, "reflex_fired": bool(payload.get("hazard", False)),
            "confidence": 0.82}


def _noop_curriculum(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {"levels": int(payload.get("n_levels", 3)),
            "solvable": True, "confidence": 0.75}


def _noop_self_train(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {"triplets": int(payload.get("n_triplets", 4)),
            "validated": True, "confidence": 0.7}


def _noop_self_evo(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {"n_candidates": int(payload.get("n_candidates", 3)),
            "improved": True, "confidence": 0.68}


def _noop_action(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {"action_token": str(payload.get("body_part", "limb")),
            "dim": int(payload.get("dim", 7)), "confidence": 0.8}


def build_default_registry() -> CapabilityRegistry:
    """注册 UDOS 既有能力为 Agent-as-Tool (CPU 合成 fn 类比)。"""
    reg = CapabilityRegistry()
    reg.register(ToolSpec(
        name="ctm_reasoning", description="CTM 连续时间推演",
        input_schema={"query": "str", "horizon": "int"},
        output_schema={"answer": "str", "horizon": "int", "confidence": "float"},
        confidence=0.9, fn=_noop_reasoning, tags=["reasoning", "planning"]))
    reg.register(ToolSpec(
        name="gpm_scene", description="GPM 物理场景压缩",
        input_schema={"scene_id": "str"},
        output_schema={"scene_id": "str", "gpm_compressed": "bool",
                       "confidence": "float"},
        confidence=0.85, fn=_noop_scene, tags=["physics", "scene"]))
    reg.register(ToolSpec(
        name="sfm_space", description="SFM 空间占用查询",
        input_schema={"query_region": "str"},
        output_schema={"region": "str", "confidence": "float"},
        confidence=0.8, fn=_noop_space, tags=["spatial", "perception"]))
    reg.register(ToolSpec(
        name="pwm_imagine", description="PWM 世界模型多步想象",
        input_schema={"n_steps": "int"},
        output_schema={"predicted_n": "int", "consistent": "bool",
                       "confidence": "float"},
        confidence=0.78, fn=_noop_imagine, tags=["reasoning", "imagination"]))
    reg.register(ToolSpec(
        name="hierarchical_control", description="分层神经控制(大脑/小脑/脊髓)",
        input_schema={"hazard": "bool"},
        output_schema={"command_dim": "int", "reflex_fired": "bool",
                       "confidence": "float"},
        confidence=0.82, fn=_noop_control, tags=["control", "action"]))
    reg.register(ToolSpec(
        name="curriculum", description="课程生成与可解性自验证",
        input_schema={"n_levels": "int"},
        output_schema={"levels": "int", "solvable": "bool",
                       "confidence": "float"},
        confidence=0.75, fn=_noop_curriculum, tags=["learning", "planning"]))
    reg.register(ToolSpec(
        name="self_train", description="自训练三元组生成",
        input_schema={"n_triplets": "int"},
        output_schema={"triplets": "int", "validated": "bool",
                       "confidence": "float"},
        confidence=0.7, fn=_noop_self_train, tags=["learning", "data"]))
    reg.register(ToolSpec(
        name="self_evolution", description="自进化配置搜索",
        input_schema={"n_candidates": "int"},
        output_schema={"n_candidates": "int", "improved": "bool",
                       "confidence": "float"},
        confidence=0.68, fn=_noop_self_evo, tags=["learning", "optimization"]))
    reg.register(ToolSpec(
        name="wla_action", description="统一动作空间 token 化",
        input_schema={"body_part": "str", "dim": "int"},
        output_schema={"action_token": "str", "dim": "int",
                       "confidence": "float"},
        confidence=0.8, fn=_noop_action, tags=["action", "control"]))
    return reg
