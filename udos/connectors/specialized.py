"""专门格式 L1/L2 connector: 把公开资源的真实数据格式解析/转换为 UDOS 表示。

每个适配器都用最小内联 fixture 代表真实格式; 单测里 parse->convert 往返一致。
不下载任何大数据集/大权重。"""
from __future__ import annotations

import math
from typing import Any, Dict, List

from ..resource_registry import normalize_trajectory
from .base import GenericConnector


# ---------- 动作分块 (ACT / Diffusion Policy / RDT) ----------
class ActionChunkingConnector(GenericConnector):
    """ACT/DP 把一段连续动作分块 (action chunking) 一次性预测 H 步。
    输入 chunk: [chunk_len, action_dim] + 可选 proprio, 归一化为 UDOS 轨迹。"""

    def to_udos(self, payload: Any) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("action-chunk payload 需为 dict")
        chunk = payload.get("chunk") or payload.get("actions")
        if chunk is None:
            raise ValueError("缺少 'chunk' (动作分块)")
        proprio = payload.get("proprio") or payload.get("state")
        obs = [[float(x) for x in proprio]] * len(chunk) if proprio else []
        return normalize_trajectory({
            "episode_id": payload.get("episode_id"),
            "observations": obs or None,
            "actions": [[float(x) for x in a] for a in chunk],
        })


# ---------- VLA 离散动作 token (OpenVLA 256-bin) ----------
class VLADiscreteConnector(GenericConnector):
    """OpenVLA 把每维动作量化到 256 个 bin (基于动作分布 q01/q99)。
    正向: bins + (q01,q99) -> 连续动作; 反向: 连续 -> bins。往返量化误差有界。"""

    NBINS = 256

    def to_udos(self, payload: Any) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("vla-token payload 需为 dict")
        bins = payload.get("bins")
        q01 = payload.get("q01")
        q99 = payload.get("q99")
        if bins is None or q01 is None or q99 is None:
            raise ValueError("缺少 bins/q01/q99")
        cont = self._bins_to_continuous(bins, q01, q99)
        return normalize_trajectory({"actions": [cont],
                                     "episode_id": payload.get("episode_id")})

    def _bins_to_continuous(self, bins, q01, q99) -> List[float]:
        if len(q01) != len(q99) or len(bins) != len(q01):
            raise ValueError("bins/q01/q99 维度不一致")
        out = []
        for b, lo, hi in zip(bins, q01, q99):
            b = max(0, min(self.NBINS - 1, int(b)))
            out.append(float(lo) + (b / (self.NBINS - 1)) * (float(hi) - float(lo)))
        return out

    def continuous_to_bins(self, cont, q01, q99) -> List[int]:
        bins = []
        for v, lo, hi in zip(cont, q01, q99):
            span = float(hi) - float(lo)
            f = 0.0 if span == 0 else (float(v) - float(lo)) / span
            bins.append(int(round(max(0.0, min(1.0, f)) * (self.NBINS - 1))))
        return bins


# ---------- 机器人 episode schema (LeRobot v2/v3, Open-X, RoboMimic) ----------
class EpisodeSchemaConnector(GenericConnector):
    """解析 LeRobot/Open-X 风格的 step 列表 (点分列名 observation.images.top /
    action / state / timestamp) 为 UDOS canonical 轨迹。"""

    def to_udos(self, payload: Any) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("episode payload 需为 dict")
        steps = payload.get("steps")
        if steps is None:
            raise ValueError("缺少 'steps' (episode 步骤列表)")
        obs, acts, ts = [], [], []
        for st in steps:
            if not isinstance(st, dict):
                raise ValueError("每个 step 需为 dict")
            state = st.get("state") or st.get("observation.state") or st.get("proprio")
            action = st.get("action")
            if action is not None:
                acts.append([float(x) for x in action])
            if state is not None:
                obs.append([float(x) for x in state])
            ts.append(float(st.get("timestamp", st.get("frame_index", len(ts)))))
        return normalize_trajectory({"episode_id": payload.get("episode_id"),
                                     "observations": obs, "actions": acts,
                                     "timestamps": ts})


# ---------- SMPL / SMPL-X 姿态 (AMASS / GRAB) ----------
class SMPLPoseConnector(GenericConnector):
    """SMPL 全身姿态 (axis-angle, 72=24*3) -> 旋转 6D 表示 (UDOS  retargeting 友好)。"""

    @staticmethod
    def _axis_angle_to_rot6d(ax, ay, az) -> List[float]:
        angle = math.sqrt(ax * ax + ay * ay + az * az)
        if angle < 1e-9:
            return [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]   # 单位旋转 6D (前两列)
        kx, ky, kz = ax / angle, ay / angle, az / angle
        c, s = math.cos(angle), math.sin(angle)
        # Rodrigues -> 旋转矩阵 R, 取前两列作为 6D 表示
        r00 = c + kx * kx * (1 - c)
        r01 = kx * ky * (1 - c) - kz * s
        r02 = kx * kz * (1 - c) + ky * s
        r10 = ky * kx * (1 - c) + kz * s
        r11 = c + ky * ky * (1 - c)
        r12 = ky * kz * (1 - c) - kx * s
        return [r00, r10, r01, r11, r02, r12]

    def to_udos(self, payload: Any) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("smpl payload 需为 dict")
        pose = payload.get("body_pose") or payload.get("pose")
        if pose is None:
            raise ValueError("缺少 'body_pose' (axis-angle 列表, 长度 3 的倍数)")
        if len(pose) % 3 != 0:
            raise ValueError("body_pose 长度需为 3 的倍数")
        rot6d = []
        for i in range(0, len(pose), 3):
            rot6d += self._axis_angle_to_rot6d(pose[i], pose[i + 1], pose[i + 2])
        return {"schema": "udos/smpl-rot6d/v1",
                "episode_id": payload.get("episode_id"),
                "n_joints": len(pose) // 3,
                "rot6d": rot6d,
                "global_orient": payload.get("global_orient")}


# ---------- LAFAN1 / BVH 片段 ----------
class BVHClipConnector(GenericConnector):
    """解析最小 BVH MOTION 行 (每帧各关节旋转) -> UDOS 关节轨迹。
    这里只解析代表真实 BVH 数值行格式的内联文本, 不读大文件。"""

    def to_udos(self, payload: Any) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("bvh payload 需为 dict")
        frames = payload.get("frames")          # List[List[float]] 每帧通道
        frame_time = payload.get("frame_time", 1.0 / 30.0)
        joints = payload.get("joints", [])
        if frames is None:
            raise ValueError("缺少 'frames' (BVH MOTION 数值行列表)")
        return {"schema": "udos/bvh-clip/v1",
                "episode_id": payload.get("clip_name"),
                "joints": joints,
                "n_frames": len(frames),
                "frame_time": frame_time,
                "motion": [[float(x) for x in row] for row in frames]}


# ---------- URDF / 正逆运动学 (yourdfpy L2, 真实 import) ----------
class URDFFKConnector(GenericConnector):
    """yourdfpy / pytorch_kinematics: URDF 解析 + 正运动学。
    yourdfpy 真实安装后 _smoke 跑通; 缺包时 probe 标 absent, 不崩核心。"""

    # 内联极小 URDF (单关节旋转, 代表真实 URDF 格式; 不读外部文件)
    _INLINE_URDF = (
        '<robot name="udos_fk_demo">'
        '<link name="base"/><link name="tip"/>'
        '<joint name="j1" type="revolute"><parent link="base"/>'
        '<child link="tip"/><axis xyz="0 0 1"/>'
        '<origin xyz="0 0 0"/>'
        '<limit effort="0" velocity="0" lower="-3.14" upper="3.14"/></joint>'
        '</robot>')

    def _smoke(self, module: Any, mod: Any) -> Any:
        import io
        from ..resource_registry import CapabilityReport
        try:
            robot = mod.URDF.load(io.BytesIO(self._INLINE_URDF.encode("utf-8")))
            robot.update_cfg(robot.zero_cfg)
            T = robot.get_transform("tip")
            return CapabilityReport(
                status="available", level=self.spec.level,
                detail=f"yourdfpy 真实 import 并对内联 URDF 跑通 FK "
                       f"(update_cfg+get_transform), dofs={robot.num_dofs}",
                evidence=f"yourdfpy FK OK, tip pose={list(T.shape)}, "
                         f"links={len(robot.link_map)}")
        except Exception as e:             # noqa: BLE001
            return CapabilityReport(
                status="degraded", level=self.spec.level,
                detail=f"yourdfpy 已 import 但 FK smoke 失败: {e}",
                evidence=f"import OK, FK FAIL: {e}")

    def to_udos(self, payload: Any) -> Dict[str, Any]:
        # L1: 即便 yourdfpy 未装, 也能把关节角归一化为 canonical joint states。
        if not isinstance(payload, dict):
            raise ValueError("urdf payload 需为 dict")
        joints = payload.get("joint_angles")
        if joints is None:
            raise ValueError("缺少 'joint_angles' (dict: name->rad)")
        return {"schema": "udos/urdf-joints/v1",
                "robot": payload.get("robot", self.spec.name),
                "joint_angles": {k: float(v) for k, v in joints.items()}}

    def invoke(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if action == "fk":
            self._ensure_loadable()
            import io
            import importlib
            mod = importlib.import_module(self.spec.requires_pkg)
            urdf_text = params.get("urdf") or self._INLINE_URDF
            robot = mod.URDF.load(io.BytesIO(urdf_text.encode("utf-8")))
            robot.update_cfg(params.get("joint_cfg") or robot.zero_cfg)
            return {"status": "ok", "id": self.spec.id,
                    "dofs": int(robot.num_dofs),
                    "links": sorted(robot.link_map.keys())}
        return super().invoke(action, params)


# ---------- 世界模型 RSSM 状态结构 (DreamerV3, L1 格式/状态表示) ----------
class WorldModelRSSMConnector(GenericConnector):
    """DreamerV3 Recurrent State-Space Model 的状态结构: 离散随机项 stoch
    (categorical 概率) + 连续确定性隐状态 deter (GRU 隐向量)。
    本连接器只实现该状态结构在 UDOS 侧的编/解码 schema (无权重、不跑世界模型前向)。"""

    def to_udos(self, payload: Any) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("rssm payload 需为 dict")
        stoch = payload.get("stoch")
        deter = payload.get("deter")
        if stoch is None or deter is None:
            raise ValueError("缺少 'stoch' (离散类别概率) 与 'deter' (确定性隐向量)")
        stoch = [float(x) for x in stoch]
        deter = [float(x) for x in deter]
        s = sum(stoch) or 1.0
        probs = [x / s for x in stoch]
        return {"schema": "udos/rssm-state/v1",
                "episode_id": payload.get("episode_id"),
                "stoch_categories": len(stoch),
                "stoch_probs": probs,
                "deter_dim": len(deter),
                "deter": deter}
