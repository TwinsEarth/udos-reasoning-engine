"""
形态无关动作重定向 (Morphology-Independent Action Retargeting) — v2.9.0
========================================================================
analogy, not reproduction —— 借鉴 Human-as-Humanoid 形态无关动作重定向思想的
**轻量化类比实现**, 非复现:

UDOS 是 CPU-only、~52k 参数的合成参数化动力学小模型, 不涉及真实机器人本体、
URDF/MJCF 或真机控制。这里用 **不同维度的动作向量代理不同机器人形态** (DOF 数
不同即视为不同形态), 在纯合成数据上验证 "源形态动作 -> 目标形态动作" 的三件事:

    1. DOF 映射      : 源维 S 动作向量 -> 目标维 T 动作向量 (确定性近邻映射,
                      端点对齐, 不引入可训练权重);
    2. 时间重采样    : (dev2) 源控制频率 -> 目标控制频率的插值对齐;
    3. 关节限幅      : 重定向后动作严格裁剪到目标形态关节限位内。

设计纪律 (与全工程一致):
    * **纯函数式、确定性、不修改主模型权重** (推理时外挂);
    * **DOF 映射端点对齐**: target 首/末关节对应 source 首/末关节, 与时间重采样
      的 "保持端点一致" 互为呼应;
    * **关节限幅不可绕过**: 输出一定落在目标限位 [lo, hi] 内 (clamp);
    * **空/非法输入守卫**: 空动作、维度不符、限位非法显式 ValueError,
      不静默 inf/越界传播;
    * 不参与正式件训练 (2.9.0 主模型 52191 参数不变)。

第二引擎一律称 GPM。
"""



from __future__ import annotations

import logging

logger = logging.getLogger("udos.retargeting")


from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch


# --------------------------------------------------------------------------- #
# 形态配置
# --------------------------------------------------------------------------- #
class MorphologyConfig:
    """合成机器人形态配置 (DOF 数 / 控制频率 / 关节限位 / 运动学参数)。

    Parameters
    ----------
    dof:
        自由度数量 (>=0; 0 仅作占位形态, 不可用于 ActionRetargeter)。
    control_freq:
        控制频率 (Hz, >0)。
    joint_limits:
        每关节限位, 形状 [dof, 2] 的 (lo, hi); lo <= hi。dof=0 时须为空。
    kinematics:
        运动学参数字典 (link_lengths / dh 等合成占位量, 可选)。
    name:
        形态名 (默认 "unnamed")。
    """

    def __init__(self, dof: int, control_freq: float,
                 joint_limits: Optional[Sequence[Sequence[float]]] = None,
                 kinematics: Optional[Dict[str, Any]] = None,
                 name: str = "unnamed") -> None:
        dof = int(dof)
        if dof < 0:
            raise ValueError("dof 必须 >= 0")
        if not (control_freq > 0) or not (control_freq == control_freq) \
                or control_freq in (float("inf"), float("-inf")):
            raise ValueError("control_freq 必须为有限正数 (>0)")
        self.dof = dof
        self.control_freq = float(control_freq)
        self.name = str(name)
        if kinematics is not None and not isinstance(kinematics, dict):
            raise ValueError("kinematics 必须为 dict 或 None")
        self.kinematics: Dict[str, Any] = dict(kinematics or {})

        limits = list(joint_limits) if joint_limits is not None else []
        if dof == 0:
            if limits:
                raise ValueError("dof=0 时 joint_limits 必须为空")
            lo = torch.empty(0)
            hi = torch.empty(0)
        else:
            if len(limits) != dof:
                raise ValueError(
                    f"joint_limits 长度 {len(limits)} 与 dof {dof} 不符")
            lo_list, hi_list = [], []
            for k, lim in enumerate(limits):
                lo_k, hi_k = float(lim[0]), float(lim[1])
                if not (lo_k == lo_k and hi_k == hi_k) or \
                        lo_k in (float("inf"), float("-inf")) or \
                        hi_k in (float("inf"), float("-inf")):
                    raise ValueError(f"关节 {k} 限位含 NaN/inf 非有限值")
                if lo_k > hi_k:
                    raise ValueError(
                        f"关节 {k} 限位 lo({lo_k}) > hi({hi_k})")
                lo_list.append(lo_k)
                hi_list.append(hi_k)
            lo = torch.tensor(lo_list, dtype=torch.float32)
            hi = torch.tensor(hi_list, dtype=torch.float32)
        self.low: torch.Tensor = lo
        self.high: torch.Tensor = hi

    # ------------------------------------------------------------------ #
    # 序列化 / 反序列化
    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": "MorphologyConfig",
            "name": self.name,
            "dof": self.dof,
            "control_freq": self.control_freq,
            "joint_limits": (torch.stack([self.low, self.high], dim=-1)
                             .tolist() if self.dof > 0 else []),
            "kinematics": self.kinematics,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MorphologyConfig":
        if d.get("kind", "MorphologyConfig") != "MorphologyConfig":
            raise ValueError(f"非法 MorphologyConfig dict: {d!r}")
        return cls(dof=int(d["dof"]), control_freq=float(d["control_freq"]),
                   joint_limits=d.get("joint_limits"),
                   kinematics=d.get("kinematics"),
                   name=d.get("name", "unnamed"))

    # ------------------------------------------------------------------ #
    def compatible_with(self, other: "MorphologyConfig") -> bool:
        """形态间兼容性粗检: 两者都 dof>0 且控制频率为正 (已在构造时保证)。

        返回 True 表示二者均可作为重定向的源/目标 (频率/DOF 差异由 retargeter
        处理, 不视为不兼容)。未知/退化形态 (dof=0) 之间不兼容。
        """
        return isinstance(other, MorphologyConfig) and self.dof > 0 \
            and other.dof > 0

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return (f"MorphologyConfig(name={self.name!r}, dof={self.dof}, "
                f"freq={self.control_freq})")


# --------------------------------------------------------------------------- #
# DOF 维度映射工具 (确定性, 端点对齐)
# --------------------------------------------------------------------------- #
def map_dof(src: torch.Tensor, src_dim: int, dst_dim: int) -> torch.Tensor:
    """把最后一维 src_dim 的动作向量确定性映射到 dst_dim。

    对每个目标关节 j (0..dst_dim-1), 取源关节
        i = round( j * (src_dim - 1) / max(dst_dim - 1, 1) )
    即最近邻对齐映射, 且 j=0 -> i=0, j=dst_dim-1 -> i=src_dim-1 (端点一致)。
    src_dim==dst_dim 时逐位恒等。其余维度保持不变。
    """
    if src_dim < 1 or dst_dim < 1:
        raise ValueError("src_dim / dst_dim 必须 >= 1")
    src = torch.as_tensor(src, dtype=torch.float32)
    if src.size(-1) != src_dim:
        raise ValueError(
            f"最后一维 {src.size(-1)} 与 src_dim {src_dim} 不符")
    if src_dim == dst_dim:
        return src.clone()
    if dst_dim == 1:
        return src[..., src_dim - 1:src_dim]
    idx = [int(round(j * (src_dim - 1) / (dst_dim - 1)))
           for j in range(dst_dim)]
    return src[..., torch.tensor(idx, dtype=torch.long)]


# --------------------------------------------------------------------------- #
# 动作重定向器
# --------------------------------------------------------------------------- #
class ActionRetargeter:
    """把源形态动作轨迹重定向到目标形态 (DOF 映射 + 关节限幅)。

    Parameters
    ----------
    source, target:
        源 / 目标 MorphologyConfig (均需 dof>=1)。
    """

    def __init__(self, source: MorphologyConfig,
                 target: MorphologyConfig) -> None:
        if not isinstance(source, MorphologyConfig) or \
                not isinstance(target, MorphologyConfig):
            raise ValueError("source/target 必须为 MorphologyConfig")
        if source.dof < 1 or target.dof < 1:
            raise ValueError(
                "ActionRetargeter 要求 source/target dof >= 1 "
                "(dof=0 形态不可重定向)")
        self.source = source
        self.target = target

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def retarget(self, actions: torch.Tensor) -> torch.Tensor:
        """源形态动作 [..., source.dof] -> 目标形态动作 [..., target.dof]。

        步骤: DOF 映射 (端点对齐) -> 关节限幅 (clamp 到 target 限位)。
        """
        a = torch.as_tensor(actions, dtype=torch.float32)
        if a.numel() == 0:
            raise ValueError("动作序列为空")
        if a.size(-1) != self.source.dof:
            raise ValueError(
                f"动作最后一维 {a.size(-1)} != 源 dof {self.source.dof}")
        if not bool(torch.isfinite(a).all()):
            raise ValueError("动作含 NaN/inf 非有限值")
        mapped = map_dof(a, self.source.dof, self.target.dof)
        # 关节限幅 (广播到前置维度)
        lo = self.target.low
        hi = self.target.high
        while lo.dim() < mapped.dim():
            lo = lo.unsqueeze(0)
            hi = hi.unsqueeze(0)
        clamped = torch.max(torch.min(mapped, hi), lo)
        return clamped

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def zero_shot_transfer(self, source_actions: torch.Tensor) -> Dict[str, Any]:
        """零样本形态切换: 源形态训练动作策略直接重定向到未见目标形态。

        合成数据验证: 重定向后动作有限、严格满足目标限位 (clamp)。
        返回 dict:
            actions            [..., target.dof] 重定向后动作;
            violation_rate     clamp 前被限位约束的动作分量占比 (诚实报告);
            within_limits      clamp 后是否全部落在目标限位内 (应为 True)。
        analogy, not reproduction: "未见目标形态" 指构造时未见过的 MorphologyConfig。
        """
        a = torch.as_tensor(source_actions, dtype=torch.float32)
        if a.numel() == 0:
            raise ValueError("动作序列为空")
        if a.size(-1) != self.source.dof:
            raise ValueError(
                f"动作最后一维 {a.size(-1)} != 源 dof {self.source.dof}")
        mapped = map_dof(a, self.source.dof, self.target.dof)
        lo, hi = self.target.low, self.target.high
        while lo.dim() < mapped.dim():
            lo = lo.unsqueeze(0)
            hi = hi.unsqueeze(0)
        viol = ((mapped < lo) | (mapped > hi)).float().mean()
        clamped = torch.max(torch.min(mapped, hi), lo)
        return {
            "actions": clamped,
            "violation_rate": float(viol),
            "within_limits": bool(((clamped >= self.target.low) &
                                   (clamped <= self.target.high)).all()),
        }

    # ------------------------------------------------------------------ #
    @staticmethod
    def action_energy(actions: torch.Tensor, dt: float) -> float:
        """动作代理能量 = ∫||a(t)||² dt (梯形积分, actuation proxy, 标量)。

        用梯形而非矩形: 重采样前后若为同一分段线性连续函数, 梯形积分在
        不同采样率下近似守恒 (矩形 Riemann 会因采样率不同而漂移)。
        """
        a = torch.as_tensor(actions, dtype=torch.float32)
        if a.dim() < 2:
            raise ValueError("actions 需至少 [..., T, D]")
        sq = (a * a).sum(dim=-1)          # [..., T]
        trapz = sq[..., 1:] + sq[..., :-1]   # [..., T-1]
        return float(0.5 * trapz.sum().item() * dt)

    @torch.no_grad()
    def resample(self, actions: torch.Tensor,
                 src_freq: Optional[float] = None,
                 dst_freq: Optional[float] = None,
                 kind: str = "linear") -> torch.Tensor:
        """把源频率动作轨迹 [..., T_src, source.dof] 重采样到目标频率。

        端点一致: 物理时长 = (T_src-1)/src_freq; 目标帧数
            T_dst = round(duration * dst_freq) + 1,
        首帧对应 t=0 (源首帧), 末帧对应 t=duration (源末帧)。
        支持频率比非整数 (dst/src 任意正值)。

        kind:
            "linear"  分段线性插值 (单调, 不超调);
            "cubic"   Catmull-Rom 三次样条 (端点处钳制复制端点切线)。
        """
        a = torch.as_tensor(actions, dtype=torch.float32)
        if a.dim() < 2:
            raise ValueError("actions 需为 [T,D] 或 [...,T,D]")
        T = a.size(-2)
        if T < 2:
            raise ValueError("动作轨迹至少需 2 帧 (T>=2)")
        if a.size(-1) != self.source.dof:
            raise ValueError(
                f"动作最后一维 {a.size(-1)} != 源 dof {self.source.dof}")
        src_f = float(src_freq or self.source.control_freq)
        dst_f = float(dst_freq or self.target.control_freq)
        if not (src_f > 0 and dst_f > 0):
            raise ValueError("频率必须 > 0")
        ratio = dst_f / src_f
        t_dst = int(round((T - 1) * ratio)) + 1
        t_dst = max(2, t_dst)
        # 目标时间点映射到源轴索引空间: pos_j = j / ratio (源索引单位)
        pos = torch.arange(t_dst, dtype=torch.float32) / ratio
        pos = pos.clamp(0.0, T - 1)
        if kind == "linear":
            lo = torch.floor(pos).long()
            hi = torch.clamp(lo + 1, max=T - 1)
            w = (pos - lo.float()).unsqueeze(-1)          # [T_dst,1]
            out = a.index_select(-2, lo) * (1.0 - w) + \
                a.index_select(-2, hi) * w
        elif kind == "cubic":
            i = torch.floor(pos).long()
            t = (pos - i.float())
            i0 = torch.clamp(i - 1, 0, T - 1)
            i2 = torch.clamp(i + 1, 0, T - 1)
            i3 = torch.clamp(i + 2, 0, T - 1)
            p0 = a.index_select(-2, i0)
            p1 = a.index_select(-2, i)
            p2 = a.index_select(-2, i2)
            p3 = a.index_select(-2, i3)
            tt = t.unsqueeze(-1)
            tt2 = tt * tt
            tt3 = tt2 * tt
            out = 0.5 * ((2.0 * p1) + (-p0 + p2) * tt +
                         (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * tt2 +
                         (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * tt3)
        else:
            raise ValueError(f"未知插值 kind: {kind!r} (linear/cubic)")
        return out

    # ------------------------------------------------------------------ #
    def config_dict(self) -> Dict[str, Any]:
        """重定向器元数据 (源/目标形态, 纯元数据可 JSON 化)。"""
        return {
            "kind": "ActionRetargeter",
            "source": self.source.to_dict(),
            "target": self.target.to_dict(),
            "src_dof": self.source.dof,
            "dst_dof": self.target.dof,
        }

    @classmethod
    def from_config_dict(cls, d: Dict[str, Any]) -> "ActionRetargeter":
        if d.get("kind") != "ActionRetargeter":
            raise ValueError(f"非法 ActionRetargeter dict: {d!r}")
        return cls(MorphologyConfig.from_dict(d["source"]),
                   MorphologyConfig.from_dict(d["target"]))


# --------------------------------------------------------------------------- #
# 形态预设库 (v2.9.0.dev1)
# --------------------------------------------------------------------------- #
def _angle_lim(n: int, mag: float) -> List[List[float]]:
    return [[-mag, mag]] * n


class MorphologyLibrary:
    """合成形态预设库 (analogy, not reproduction)。

    预设 3 种合成形态, 用 DOF 数 / 控制频率 / 限位代理不同机器人本体:
        * ``prime_u_60dof`` : 代理 60 自由度全身 (低频大范围关节角)
        * ``arm_7dof``       : 代理 7 自由度机械臂 (高频)
        * ``gripper_4dof``   : 代理 4 自由度夹爪 (极高频小行程)

    库本身是纯元数据注册表, 可整体序列化/反序列化。
    """

    # 预设表: name -> (dof, control_freq, limit_mag, kinematics)
    PRESETS: Dict[str, Dict[str, Any]] = {
        "prime_u_60dof": {
            "dof": 60, "control_freq": 100.0, "limit_mag": 1.5707963,
            "kinematics": {"body": "proxy", "desc": "60-dof whole-body proxy"},
        },
        "arm_7dof": {
            "dof": 7, "control_freq": 500.0, "limit_mag": 3.1415927,
            "kinematics": {"chain": 7, "desc": "7-dof arm proxy"},
        },
        "gripper_4dof": {
            "dof": 4, "control_freq": 1000.0, "limit_mag": 1.0,
            "kinematics": {"fingers": 2, "desc": "4-dof gripper proxy"},
        },
    }

    def __init__(self) -> None:
        self._configs: Dict[str, MorphologyConfig] = {}
        for name, spec in self.PRESETS.items():
            self._configs[name] = MorphologyConfig(
                dof=spec["dof"], control_freq=spec["control_freq"],
                joint_limits=_angle_lim(spec["dof"], spec["limit_mag"]),
                kinematics=spec["kinematics"], name=name)

    def names(self) -> List[str]:
        return list(self._configs)

    def get(self, name: str) -> MorphologyConfig:
        if name not in self._configs:
            raise KeyError(
                f"未知形态 {name!r}; 可用: {self.names()}")
        return self._configs[name]

    def register(self, name: str, cfg: MorphologyConfig) -> None:
        """注册自定义形态 (不覆盖预设名除非显式)。"""
        if not isinstance(name, str) or not name:
            raise ValueError("形态名须为非空字符串")
        if not isinstance(cfg, MorphologyConfig):
            raise ValueError("cfg 必须为 MorphologyConfig")
        self._configs[name] = cfg

    # ------------------------------------------------------------------ #
    def are_compatible(self, name_a: str, name_b: str) -> bool:
        """两命名形态间兼容性检查 (均 dof>0 即可重定向)。"""
        return self.get(name_a).compatible_with(self.get(name_b))

    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        return {"kind": "MorphologyLibrary",
                "configs": {n: c.to_dict() for n, c in self._configs.items()}}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MorphologyLibrary":
        if d.get("kind") != "MorphologyLibrary":
            raise ValueError(f"非法 MorphologyLibrary dict: {d!r}")
        lib = cls()
        lib._configs = {n: MorphologyConfig.from_dict(c)
                        for n, c in d["configs"].items()}
        return lib
