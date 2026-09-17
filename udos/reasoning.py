"""
UDOS 双引擎协同: GPM(场景内化) + CTM(时序推演)
=============================================
流程 (对齐技术方案 2.4):
    物理场景 (PCE)
      -> [GPM] 超网络单次前向生成 LoRA (零反向传播)
      -> [注入] 前向补丁注入基座, 模型"记住"物理场景
      -> [CTM] 在场景记忆上沿内部时间轴展开因果推演
      -> 输出: 预测 + certainty 轨迹 + 因果链 + 最优策略量
      -> reset_scene() 无损移除场景记忆 ("阅后即焚")
"""



from __future__ import annotations

import logging

logger = logging.getLogger("udos.reasoning")


from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from .ctm_engine import CTMConfig, CTMPhysicsEngine
from .gpm_engine import (
    GPMConfig,
    LoRAInjector,
    LoRASet,
    PhysicsHypernetwork,
    infer_dims_from_model,
)
from .pce_format import PhysicalToken, PhysicsScene, PhysicsSceneEncoder
from .scene_bridge import extract_scene_params
from .dynamics import RAW_DIM


@dataclass
class ReasoningResult:
    scene_id: Optional[str]
    query: str
    prediction: torch.Tensor           # [out_dims] 最终预测
    prediction_trajectory: torch.Tensor  # [out_dims, ticks]
    certainty_trajectory: torch.Tensor   # [2, ticks] (熵, 1-熵)
    ticks_used: int
    sync_representation: torch.Tensor
    causal_chain: List[Dict[str, Any]] = field(default_factory=list)
    lora_params: int = 0
    scene_conditioned: bool = False          # v2: 是否注入了 GPM 场景条件
    predicted_state: Optional[Dict[str, List[float]]] = None  # v2: 可解释下一时刻 pos/vel
    future_states: Optional[List[Dict[str, List[float]]]] = None  # v2.1 多步 rollout
    # v5.4.6: 主预测员 PhysicsPredictor 是否真正消费了场景参数 (区别于上面
    # scene_conditioned 仅指内部演示 CTM 是否收到 GPM 嵌入)。
    predictor_conditioned: bool = False
    scene_params: Optional[List[float]] = None       # 4 维隐藏物理参数
    scene_params_source: Optional[str] = None        # metadata / attributes / None

    def convergence(self) -> float:
        return float(self.certainty_trajectory[1, -1])

    def summary(self) -> Dict[str, Any]:
        out = {
            "scene_id": self.scene_id,
            "query": self.query,
            "ticks_used": self.ticks_used,
            "final_certainty": round(self.convergence(), 4),
            "lora_params": self.lora_params,
            "causal_edges": len(self.causal_chain),
            "scene_conditioned": self.scene_conditioned,
            "predictor_conditioned": self.predictor_conditioned,
        }
        if self.scene_params is not None:
            out["scene_params"] = self.scene_params
            out["scene_params_source"] = self.scene_params_source
        if self.predicted_state is not None:
            out["predicted_next_state"] = self.predicted_state
        if self.future_states is not None:
            out["future_states"] = self.future_states
        return out


class UDOSReasoningEngine(nn.Module):
    def __init__(self,
                 ctm_config: Optional[CTMConfig] = None,
                 gpm_config: Optional[GPMConfig] = None,
                 base_model: Optional[nn.Module] = None,
                 lora_scaling: float = 1.0,
                 scene_conditioning: bool = True,
                 predictor: Optional[Any] = None):
        super().__init__()
        self.scene_conditioning = scene_conditioning
        self.predictor = predictor  # 可选: 训练好的 PhysicsPredictor, 提供可解释物理预测
        # 未提供基座时给出轻量演示基座, 并自动探测 LoRA 维度
        if base_model is None:
            from .gpm_engine import TinyBaseModel
            hidden = (gpm_config.feature_dim if gpm_config else 128)
            n_layers = len(gpm_config.layer_indices) if gpm_config else 4
            base_model = TinyBaseModel(hidden=hidden, n_layers=n_layers)
        self.base_model = base_model

        if gpm_config is None:
            gpm_config = GPMConfig()
        if gpm_config.dims is None:
            gpm_config.dims = infer_dims_from_model(
                base_model, gpm_config.target_modules)
        self.gpm = PhysicsHypernetwork(gpm_config)
        self.injector = LoRAInjector(base_model, scaling=lora_scaling)

        if ctm_config is None:
            ctm_config = CTMConfig()
        # v2 真耦合: GPM 场景嵌入维度作为 CTM 场景条件维度 (gate 零初始化, 初始等价一代)
        if self.scene_conditioning and ctm_config.scene_dim is None:
            ctm_config.scene_dim = self.gpm.cfg.latent_size
        self.ctm = CTMPhysicsEngine(ctm_config)
        self.ctm_encoder = PhysicsSceneEncoder(ctm_config.d_input)

        self.scene_memory: Dict[str, LoRASet] = {}

    def attach_predictor(self, predictor: Any) -> None:
        """挂载训练好的 PhysicsPredictor, reason 时输出可解释下一时刻物理量。"""
        self.predictor = predictor

    @staticmethod
    def _tokens_raw(scene: PhysicsScene, window: Optional[int] = None
                    ) -> "torch.Tensor":
        """取场景时间线末尾窗口的 [position(3), velocity(3)] -> [1, W, RAW_DIM]。"""
        ordered = scene.timeline()
        if window is not None:
            ordered = ordered[-window:]
        raw = [[*(t.position[:3]), *(t.velocity[:3])] for t in ordered]
        return torch.tensor(raw, dtype=torch.float32).unsqueeze(0)

    # ---------- 步骤 1: GPM 内化物理场景 ----------
    @torch.no_grad()
    def internalize_scene(self, scene: PhysicsScene,
                          chunks: Optional[List[PhysicsScene]] = None,
                          weights: Optional[Sequence[float]] = None) -> str:
        self.ctm_encoder.bind_scene(scene)
        if chunks:
            for c in chunks:
                self.ctm_encoder.bind_scene(c)
            lora = self.gpm.forward_chunked(chunks, weights=weights)
            lora.scene_id = scene.scene_id
        else:
            lora = self.gpm(scene)
        self.injector.inject(lora)
        self.scene_memory[scene.scene_id] = lora
        return (f"场景 {scene.scene_id} 已内化: "
                f"{lora.num_params():,} 个 LoRA 参数 "
                f"({lora.num_bytes_fp32()/1024:.1f} KB fp32), "
                f"覆盖模块 {lora.module_names()}")

    # ---------- 步骤 2: CTM 时序因果推演 ----------
    @torch.no_grad()
    def reason(self,
               scene: PhysicsScene,
               query: str = "",
               scene_id: Optional[str] = None,
               track: bool = True,
               horizon: int = 1) -> ReasoningResult:
        sid = scene_id or scene.scene_id
        # 注意: 内化不是强制的 (允许无场景记忆的纯 CTM 推演), 但给出提示
        # 物理 Token 序列 -> CTM 输入 [1, seq, d_input]
        seq = self.ctm_encoder(scene.tokens).unsqueeze(0)
        # v2 真耦合: GPM 场景嵌入 -> CTM 场景条件 (gate 为 0 时等价一代)
        scene_ctx = None
        conditioned = False
        if self.scene_conditioning:
            scene_ctx = self.gpm.scene_embedding(scene).unsqueeze(0)  # [1, latent]
            conditioned = True
        preds, certs, sync_out, info = self.ctm(
            seq, track=track, scene_context=scene_ctx)
        preds = preds[0]        # [out, ticks]
        certs = certs[0]        # [2, ticks]

        # v2/v2.1 可解释输出: 训练好的预测器给出下一时刻, 并可多步自由滚动
        # v5.4.6: 经 scene_bridge 提取 PCE 承载的隐藏场景参数, 喂给主预测员
        # 训练过的场景门; 无场景参数时保持旧版场景盲 rollout (逐位兼容)。
        predicted_state = None
        future_states = None
        predictor_conditioned = False
        sp_values: Optional[List[float]] = None
        sp_source: Optional[str] = None
        if self.predictor is not None and len(scene.tokens) >= 1:
            raw = self._tokens_raw(scene)
            H = max(1, int(horizon))
            sp = extract_scene_params(scene)
            if sp is not None:
                predictor_conditioned = True
                sp_source = sp.source
                sp_values = [round(float(v), 6) for v in
                             sp.values.flatten().tolist()]
                roll = self.predictor.rollout(
                    raw, H, scene_params=sp.values)[0]  # [H,6]
            else:
                roll = self.predictor.rollout(raw, H)[0]  # 场景盲旧路径
            def _state(vec):
                return {"position": [round(v, 6) for v in vec[:3]],
                        "velocity": [round(v, 6) for v in vec[3:6]]}
            predicted_state = _state(roll[0].tolist())
            if H > 1:
                future_states = [_state(v.tolist()) for v in roll]

        chain = self._build_causal_chain(scene, info)
        lora = self.scene_memory.get(sid)
        return ReasoningResult(
            scene_id=sid,
            query=query,
            prediction=preds[:, -1],
            prediction_trajectory=preds,
            certainty_trajectory=certs,
            ticks_used=info["ticks_used"],
            sync_representation=sync_out[0],
            causal_chain=chain,
            lora_params=lora.num_params() if lora else 0,
            scene_conditioned=conditioned,
            predicted_state=predicted_state,
            future_states=future_states,
            predictor_conditioned=predictor_conditioned,
            scene_params=sp_values,
            scene_params_source=sp_source,
        )

    def _build_causal_chain(self, scene: PhysicsScene,
                            info: Dict) -> List[Dict[str, Any]]:
        """融合 PCE 显式因果父节点与时间邻接, 产出因果链。"""
        chain: List[Dict[str, Any]] = []
        ordered = scene.timeline()
        # 1) PCE 显式声明的因果边
        for t in ordered:
            for parent in t.causal_parents:
                chain.append({
                    "from": f"{parent.get('object_id')}@t{parent.get('timestamp')}",
                    "to": f"{t.object_id}@t{t.timestamp}",
                    "source": "pce-explicit",
                })
        # 2) 同一物体时间邻接边 (惯性/状态延续)
        by_obj: Dict[str, List[PhysicalToken]] = {}
        for t in ordered:
            by_obj.setdefault(t.object_id, []).append(t)
        for obj, ts in by_obj.items():
            for a, b in zip(ts[:-1], ts[1:]):
                chain.append({
                    "from": f"{obj}@t{a.timestamp}",
                    "to": f"{obj}@t{b.timestamp}",
                    "source": "temporal-adjacent",
                })
        return chain

    # ---------- 步骤 3: 移除场景记忆 ----------
    def reset_scene(self, scene_id: str) -> str:
        self.injector.reset()
        self.scene_memory.pop(scene_id, None)
        return f"场景 {scene_id} 已重置 (LoRA 前向补丁无损移除)"

    def reset_all(self) -> None:
        self.injector.reset()
        self.scene_memory.clear()
