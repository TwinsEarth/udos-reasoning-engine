"""
统一具身推理头 + 动作三分组骨架 (v3.9.0) — 宇树 UnifoLM-WLA 机制类比线
=========================================================================
analogy, not reproduction —— 类比 UnifoLM-ER-1 (具身推理) 与 WLA 统一动作空间三分,
**非复现**: UDOS 是 CPU-only、~52k 参数合成动力学小模型, 不碰真机/VLM/6B 权重。

本模块仅做两件事 (均**外挂、零梯度、opt-in, 不改主 predictor 52191 参数**):

1. ``EmbodiedReasoningHead`` —— 统一具身推理聚合头:
   * **编排而非重造**: 复用 predictor.obs_encoder 提取共享 latent (与 MultiTaskHead
     同一特征路径), 在其上以**确定性投影**聚合三类 ER-1 结构化代理输出:
       - spatial_relation  空间关系代理 (latent 前 2 维 -> 单位圆内相对方位)
       - target_point       目标点代理 (latent 中位 2 维 -> 单位球内 2D 目标点)
       - trajectory_proxy   2D 轨迹代理 (latent 后 2 维 -> 单位步长位移序列)
   * 不新增可训练权重 (纯确定性函数); 默认 enable=False, 开启也不改主预测路径。
   * 与既有 spatial / affordance / unified_head(MultiTaskHead) 协同: describe() 报告
     已编排的子系统, 不重复实现其几何/打分逻辑。

2. ``ActionTriGroup`` —— 统一动作空间三分组骨架:
   * 类比 WLA 官方三分量: ①末端执行器位姿 EEF pose; ②末端关节 EEF joints;
     ③下肢关节 lower-body joints。
   * 把合成统一动作向量按 (eef_pose_dim, eef_joints_dim, lower_body_dim) 三段切分/
     合并; 不引入可训练权重, 仅做结构路由 + 限幅守卫。

设计纪律 (与全工程一致):
    * 纯前向、确定性、@torch.no_grad, 只读 predictor, 不改主权重;
    * 空/非法 (空动作、维度不符、NaN/inf、分组和 != 总维) 显式 ValueError;
    * 日志走 ``logging.getLogger("udos.embodied")`` (默认 stderr, 不污染 stdout/HTTP);
    * 第二引擎一律称 GPM。
"""

from __future__ import annotations

import logging

logger = logging.getLogger("udos.embodied")

from typing import Any, Dict, List, Optional, Tuple

import torch


# --------------------------------------------------------------------------- #
# 统一具身推理聚合头 (编排 spatial / affordance / unified-head, 不重造)
# --------------------------------------------------------------------------- #
class EmbodiedReasoningHead:
    """统一具身推理头 (只读外挂, 持有 predictor 引用, 不注册其参数)。

    Parameters
    ----------
    predictor:
        已训练 PhysicsPredictor (只读)。latent 宽度取 obs_encoder 输出 d_input。
    enable:
        默认 False -> forward 返回空 dict, 主预测路径完全不变 (opt-in)。
    """

    def __init__(self, predictor, enable: bool = False) -> None:
        self.predictor = predictor
        self.latent_dim = int(predictor.ctm.cfg.d_input)
        self.enable = bool(enable)
        # ER-1 三类结构化代理各取 2 维 latent, 共需 >=6 维 (d_input=32 满足)
        if self.latent_dim < 6:
            raise ValueError(
                f"EmbodiedReasoningHead 需 latent_dim>=6, 当前 {self.latent_dim}")

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _latent(self, window: torch.Tensor,
                scene_params: Optional[torch.Tensor]) -> torch.Tensor:
        """window[B,W,RAW] -> latent[B,d_input] (obs_encoder 时间维末帧)。"""
        w = torch.as_tensor(window, dtype=torch.float32)
        if w.dim() == 2:
            w = w.unsqueeze(0)
        if w.dim() != 3:
            raise ValueError("window 需为 [W,RAW] 或 [B,W,RAW]")
        self.predictor.eval()
        return self.predictor.obs_encoder(w)[:, -1, :]     # [B, d_input]

    @staticmethod
    def _safe_tanh(x: torch.Tensor) -> torch.Tensor:
        """tanh 软压到 (-1,1), 避免非有限传播 (NaN/inf -> 0 前先守卫)。"""
        if not bool(torch.isfinite(x).all()):
            raise ValueError("latent 含 NaN/inf, 拒绝生成 ER 结构化代理")
        return torch.tanh(x)

    @torch.no_grad()
    def forward(self, window: torch.Tensor,
                scene_params: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """enable=False -> {}; 否则返回三类 ER-1 结构化代理聚合。

        返回 dict:
            spatial_relation [B,2]  相对方位代理 (单位圆内)
            target_point     [B,2]  目标点代理 (单位球内 2D)
            trajectory_proxy  [B,2] 单步 2D 轨迹位移代理
            latent_dim       int
        """
        if not self.enable:
            return {}
        z = self._latent(window, scene_params)             # [B, d_input]
        rel = self._safe_tanh(z[:, 0:2])
        tgt = self._safe_tanh(z[:, 2:4])
        traj = self._safe_tanh(z[:, 4:6])
        return {
            "spatial_relation": rel,
            "target_point": tgt,
            "trajectory_proxy": traj,
            "latent_dim": self.latent_dim,
        }

    def __call__(self, window, scene_params=None):
        return self.forward(window, scene_params=scene_params)

    def describe(self) -> Dict[str, Any]:
        return {
            "kind": "EmbodiedReasoningHead",
            "orchestrates": ["spatial", "affordance", "unified_head(MultiTaskHead)"],
            "rebuild": False,
            "latent_dim": self.latent_dim,
            "enable": self.enable,
            "no_new_trainable_params": True,
            "main_predictor_params_untouched": True,
            "analogy_not_reproduction": True,
        }


# --------------------------------------------------------------------------- #
# 统一动作空间三分组骨架 (EEF pose / EEF joints / lower-body)
# --------------------------------------------------------------------------- #
class ActionTriGroup:
    """把合成统一动作向量按 WLA 三分量切分 / 合并 (纯结构路由, 无可训参数)。

    Parameters
    ----------
    eef_pose_dim:
        末端执行器位姿维度 (>0)。
    eef_joints_dim:
        末端关节 (夹爪/灵巧手) 维度 (>=0)。
    lower_body_dim:
        下肢关节维度 (>=0)。
    """

    GROUPS: Tuple[str, ...] = ("eef_pose", "eef_joints", "lower_body")

    def __init__(self, eef_pose_dim: int, eef_joints_dim: int,
                 lower_body_dim: int) -> None:
        for name, v in (("eef_pose_dim", eef_pose_dim),
                        ("eef_joints_dim", eef_joints_dim),
                        ("lower_body_dim", lower_body_dim)):
            if not isinstance(v, int) or v < 0:
                raise ValueError(f"{name} 须为非负整数")
        if eef_pose_dim < 1:
            raise ValueError("eef_pose_dim 须 >=1 (末端位姿不可缺)")
        self.eef_pose_dim = eef_pose_dim
        self.eef_joints_dim = eef_joints_dim
        self.lower_body_dim = lower_body_dim
        self.total_dim = eef_pose_dim + eef_joints_dim + lower_body_dim

    # ------------------------------------------------------------------ #
    def _check(self, action: torch.Tensor) -> torch.Tensor:
        a = torch.as_tensor(action, dtype=torch.float32)
        if a.numel() == 0:
            raise ValueError("动作向量为空")
        if a.size(-1) != self.total_dim:
            raise ValueError(
                f"动作最后一维 {a.size(-1)} != 统一动作总维 {self.total_dim}")
        if not bool(torch.isfinite(a).all()):
            raise ValueError("动作含 NaN/inf 非有限值")
        return a

    @torch.no_grad()
    def split(self, action: torch.Tensor) -> Dict[str, torch.Tensor]:
        """统一动作 [..., total_dim] -> 三分量 dict (沿最后一维切分)。"""
        a = self._check(action)
        p, j, lb = self.eef_pose_dim, self.eef_joints_dim, self.lower_body_dim
        out = {"eef_pose": a[..., :p]}
        if j > 0:
            out["eef_joints"] = a[..., p:p + j]
        if lb > 0:
            out["lower_body"] = a[..., p + j:p + j + lb]
        return out

    @torch.no_grad()
    def merge(self, groups: Dict[str, torch.Tensor]) -> torch.Tensor:
        """三分量 dict -> 统一动作 [..., total_dim] (沿最后一维拼接)。"""
        if "eef_pose" not in groups:
            raise ValueError("groups 必须含 'eef_pose'")
        parts: List[torch.Tensor] = []
        for name in self.GROUPS:
            v = groups.get(name)
            if v is None:
                continue
            t = torch.as_tensor(v, dtype=torch.float32)
            expect = {"eef_pose": self.eef_pose_dim,
                      "eef_joints": self.eef_joints_dim,
                      "lower_body": self.lower_body_dim}[name]
            if expect == 0:
                raise ValueError(f"本配置 {name} 维度为 0, 不应提供该分量")
            if t.size(-1) != expect:
                raise ValueError(
                    f"分量 {name} 维 {t.size(-1)} != 配置 {expect}")
            if not bool(torch.isfinite(t).all()):
                raise ValueError(f"分量 {name} 含 NaN/inf")
            parts.append(t)
        return torch.cat(parts, dim=-1)

    def describe(self) -> Dict[str, Any]:
        return {
            "kind": "ActionTriGroup",
            "groups": list(self.GROUPS),
            "dims": {"eef_pose": self.eef_pose_dim,
                     "eef_joints": self.eef_joints_dim,
                     "lower_body": self.lower_body_dim},
            "total_dim": self.total_dim,
            "no_trainable_params": True,
            "main_predictor_params_untouched": True,
            "analogy_not_reproduction": True,
        }
