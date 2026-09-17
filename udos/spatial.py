"""
空间基础模型核心 (Spatial Foundation Model, SFM) — v3.5.0
================================================================
analogy, not reproduction —— 这是 **合成低维 3D 几何代理**, 不宣称复现
SpatialVLM / 真实 RGBD / 点云 / 真实相机。所有空间推理为**纯推理外挂**:

    * **纯前向、确定性、零梯度**: 不参与主 predictor (52191 参数) 训练,
      不修改任何主权重; 空间关系全部由解析几何计算 (非学习)。
    * **opt-in**: 默认路径不引入本模块, 不改变既有 846 测试的逐位输出。
    * **空/非法输入显式 ValueError**: 空场景、形状不符、非有限值、
      退化旋转矩阵等一律显式报错, 不伪造证据。
    * **坐标变换可逆性可验**: SpatialTransform 提供解析逆变换,
      apply(inverse(p)) ≈ p (数值容差内逐分量可验)。

与既有模块的接口对齐:
    * `PhysicalToken` (pce_format) 含 position[3]/velocity[3];
      SpatialObject 提供 from_physical_token / to_physical_token 互转。
    * `AffordanceScorer` 用 [pos3, vel3] 代理物体部位; 本模块的
      position/velocity 布局与其一致, 便于上层组合。

第二引擎一律称 GPM。
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger("udos.spatial")

# 旋转矩阵正交性 / 可逆性数值容差
_ROT_TOL = 1e-8
_RT_TOL = 1e-9


# --------------------------------------------------------------------------- #
# 内部工具
# --------------------------------------------------------------------------- #
def _as_vec3(v: Any, name: str) -> np.ndarray:
    """把任意可迭代值规整为长度 3 的 float64 向量; 非法/非有限显式 ValueError。"""
    try:
        arr = np.asarray(v, dtype=np.float64).reshape(-1)
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"{name} 无法转为数值向量: {e}")
    if arr.size != 3:
        raise ValueError(f"{name} 须为长度 3 的向量, 实际长度 {arr.size}: {v!r}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} 含 NaN/inf 非有限值: {v!r}")
    return arr.copy()


def _check_rotation(R: np.ndarray) -> np.ndarray:
    """校验 3x3 旋转矩阵: 正交且 det=1 (纯旋转, 不含反射)。"""
    try:
        M = np.asarray(R, dtype=np.float64).reshape(3, 3)
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"rotation 须为 3x3 矩阵: {e}")
    if not np.isfinite(M).all():
        raise ValueError("rotation 含 NaN/inf")
    err = float(np.max(np.abs(M @ M.T - np.eye(3))))
    if err > _ROT_TOL:
        raise ValueError(f"rotation 非正交 (M M^T-I 最大误差 {err:.2e} > {_ROT_TOL})")
    det = float(np.linalg.det(M))
    if abs(det - 1.0) > _ROT_TOL:
        raise ValueError(f"rotation 行列式须为 +1 (纯旋转), 实际 {det:.6f}; "
                         f"反射/翻转不是刚体旋转")
    return M


# --------------------------------------------------------------------------- #
# 空间物体
# --------------------------------------------------------------------------- #
class SpatialObject:
    """单个空间物体 (合成代理): 位置 pos3 / 速度 vel3 / 球体半径。

    布局 [pos(3), vel(3)] 与 affordance / RAW_DIM 约定对齐。
    """

    def __init__(self, object_id: str, position: Sequence[float],
                 velocity: Sequence[float] = (0.0, 0.0, 0.0),
                 radius: float = 0.5) -> None:
        if not isinstance(object_id, str) or not object_id.strip():
            raise ValueError("object_id 须为非空字符串")
        if not (np.isfinite(radius) and radius > 0.0):
            raise ValueError(f"radius 须为正有限值, 收到 {radius!r}")
        self.object_id = object_id
        self.position = _as_vec3(position, "position")
        self.velocity = _as_vec3(velocity, "velocity")
        self.radius = float(radius)

    # -- 几何量 --------------------------------------------------------- #
    def distance_to(self, other: "SpatialObject") -> float:
        """到另一物体中心的欧氏距离 (标量, 代理)。"""
        if not isinstance(other, SpatialObject):
            raise ValueError("other 须为 SpatialObject")
        return float(np.linalg.norm(self.position - other.position))

    def relative_position(self, other: "SpatialObject") -> np.ndarray:
        """other - self 的位移向量 (3,)。"""
        if not isinstance(other, SpatialObject):
            raise ValueError("other 须为 SpatialObject")
        return other.position - self.position

    # -- 与 pce_format.PhysicalToken 互转 ------------------------------- #
    @classmethod
    def from_physical_token(cls, tok: Any, radius: float = 0.5) -> "SpatialObject":
        """从 pce_format.PhysicalToken 构造 (position/velocity 取前 3 维)。"""
        pos = list(tok.position)[:3]
        vel = list(getattr(tok, "velocity", (0.0, 0.0, 0.0)))[:3]
        return cls(object_id=str(tok.object_id), position=pos, velocity=vel,
                   radius=radius)

    def to_physical_token(self, timestamp: int = 0) -> Any:
        """转回 pce_format.PhysicalToken (惰性导入, 避免循环依赖)。"""
        from .pce_format import PhysicalToken
        return PhysicalToken(object_id=self.object_id, timestamp=int(timestamp),
                             position=self.position.tolist(),
                             velocity=self.velocity.tolist())

    # -- 序列化 --------------------------------------------------------- #
    def to_dict(self) -> Dict[str, Any]:
        return {"object_id": self.object_id,
                "position": self.position.tolist(),
                "velocity": self.velocity.tolist(),
                "radius": self.radius}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SpatialObject":
        return cls(object_id=d["object_id"], position=d["position"],
                   velocity=d.get("velocity", (0.0, 0.0, 0.0)),
                   radius=d.get("radius", 0.5))

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        p = self.position.tolist()
        return (f"SpatialObject({self.object_id!r}, pos={[round(x, 3) for x in p]}, "
                f"r={self.radius})")


# --------------------------------------------------------------------------- #
# 坐标变换 (平移 / 旋转 / 缩放, 解析可逆)
# --------------------------------------------------------------------------- #
class SpatialTransform:
    """刚体/相似坐标变换: p' = t + R @ diag(s) @ p; v' = R @ diag(s) @ v。

    * R 为纯旋转 (正交, det=1); s 为三轴缩放 (>0); t 为平移。
    * inverse() 返回解析逆变换, apply(inverse(p)) ≈ p。
    * compose(other) 返回 self ∘ other (先 other 后 self)。
    * 纯解析, 零参数, 确定性。
    """

    def __init__(self, translate: Sequence[float] = (0.0, 0.0, 0.0),
                 rotation: Optional[Sequence[Sequence[float]]] = None,
                 scale: Sequence[float] = (1.0, 1.0, 1.0)) -> None:
        self.t = _as_vec3(translate, "translate")
        self.R = np.eye(3) if rotation is None else _check_rotation(rotation)
        s = _as_vec3(scale, "scale")
        if (s <= 0).any() or not np.isfinite(s).all():
            raise ValueError("scale 各分量须为正有限值")
        self.s = s
        # 内部统一存 3x3 线性矩阵 L = R @ diag(s); apply/inverse/compose
        # 全部用完整 L 做精确矩阵运算 (复合/逆不依赖可对角化假设)。
        self._L = (self.R * self.s).copy()

    @property
    def linear(self) -> np.ndarray:
        """3x3 线性部分 L (只读拷贝): p' = L @ p + t。"""
        return self._L.copy()

    @property
    def effective_scale(self) -> float:
        """等效各向同性缩放因子: L 奇异值的几何平均 (用于球体半径代理)。"""
        sv = np.linalg.svd(self._L, compute_uv=False)
        return float(np.cbrt(float(np.prod(sv))))

    @classmethod
    def _from_linear(cls, L: np.ndarray, t: np.ndarray) -> "SpatialTransform":
        """内部工厂: 由完整线性矩阵 L + 平移 t 构造 (逆/复合用)。"""
        obj = cls.__new__(cls)
        obj._L = np.asarray(L, dtype=np.float64).reshape(3, 3).copy()
        obj.t = np.asarray(t, dtype=np.float64).reshape(3).copy()
        if not (np.isfinite(obj._L).all() and np.isfinite(obj.t).all()):
            raise ValueError("线性变换含 NaN/inf")
        # 内省用: 极分解旋转 + 奇异值缩放 (仅供查看, 不参与 apply)
        obj.R = cls._polar_rotation(obj._L)
        sv = np.linalg.svd(obj._L, compute_uv=False)
        obj.s = sv
        return obj

    # -- 工厂 ----------------------------------------------------------- #
    @classmethod
    def identity(cls) -> "SpatialTransform":
        return cls()

    @classmethod
    def translation(cls, offset: Sequence[float]) -> "SpatialTransform":
        return cls(translate=offset)

    @classmethod
    def scaling(cls, factor: Sequence[float] | float) -> "SpatialTransform":
        if isinstance(factor, (int, float)):
            factor = (float(factor),) * 3
        return cls(scale=factor)

    @classmethod
    def rotation_from_axis_angle(cls, axis: Sequence[float],
                                 angle_rad: float) -> "SpatialTransform":
        """绕单位轴右手旋转 angle_rad 弧度 (Rodrigues)。"""
        k = _as_vec3(axis, "axis")
        n = float(np.linalg.norm(k))
        if n == 0.0:
            raise ValueError("旋转轴不能为零向量")
        k = k / n
        if not math.isfinite(angle_rad):
            raise ValueError("angle_rad 须为有限值")
        K = np.array([[0.0, -k[2], k[1]],
                      [k[2], 0.0, -k[0]],
                      [-k[1], k[0], 0.0]])
        R = (np.eye(3) * math.cos(angle_rad)
             + (1.0 - math.cos(angle_rad)) * np.outer(k, k)
             + math.sin(angle_rad) * K)
        return cls(rotation=R)

    # -- 前向 ----------------------------------------------------------- #
    def _as_points(self, points: Any) -> np.ndarray:
        arr = np.asarray(points, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, 3)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f"points 须为 [..., 3], 实际 shape {arr.shape}")
        if not np.isfinite(arr).all():
            raise ValueError("points 含 NaN/inf")
        return arr

    def apply(self, points: Sequence[float] | np.ndarray) -> np.ndarray:
        """变换位置点集: p' = L @ p + t。返回与输入同结构的 (N,3)。"""
        pts = self._as_points(points)
        return (self._L @ pts.T).T + self.t

    def apply_vectors(self, vectors: Sequence[float] | np.ndarray) -> np.ndarray:
        """变换自由向量 (速度/方向): 仅线性部分 v' = L @ v, 不加平移。"""
        vec = self._as_points(vectors)
        return (self._L @ vec.T).T

    # -- 逆变换 / 复合 -------------------------------------------------- #
    def inverse(self) -> "SpatialTransform":
        """解析逆: p = L^{-1} (p' - t)。用完整矩阵求逆, 精确可逆。"""
        Linv = np.linalg.inv(self._L)
        tinv = -Linv @ self.t
        return SpatialTransform._from_linear(Linv, tinv)

    def compose(self, other: "SpatialTransform") -> "SpatialTransform":
        """self ∘ other: 先 other 后 self, 用完整线性矩阵精确复合。

        other: p1 = other.L p + other.t
        self:  p2 = self.L p1 + self.t = (self.L other.L) p
                   + (self.L other.t + self.t)
        """
        if not isinstance(other, SpatialTransform):
            raise ValueError("other 须为 SpatialTransform")
        L2 = self._L @ other._L
        t2 = self.t + self._L @ other.t
        return SpatialTransform._from_linear(L2, t2)

    @staticmethod
    def _polar_rotation(A: np.ndarray) -> np.ndarray:
        """极分解提取旋转部分 R = A (A^T A)^{-1/2}; 正交化后强制 det=1。"""
        AtA = A.T @ A
        w, V = np.linalg.eigh(AtA)
        w = np.clip(w, 1e-12, None)
        inv_sqrt = V @ np.diag(1.0 / np.sqrt(w)) @ V.T
        R = A @ inv_sqrt
        if float(np.linalg.det(R)) < 0:
            # 数值退化为反射时, 翻转最后一列使其成为纯旋转
            R[:, -1] *= -1.0
        return R

    # -- 自检 ----------------------------------------------------------- #
    def roundtrip_error(self, points: Any) -> float:
        """apply(inverse(apply(p))) - p 的最大绝对值误差 (可逆性可验)。"""
        pts = self._as_points(points)
        inv = self.inverse()
        back = inv.apply(self.apply(pts))
        return float(np.max(np.abs(back - pts)))

    def is_identity(self) -> bool:
        return (np.allclose(self.t, 0.0, atol=_RT_TOL)
                and np.allclose(self._L, np.eye(3), atol=_RT_TOL))

    def to_dict(self) -> Dict[str, Any]:
        return {"translate": self.t.tolist(), "rotation": self.R.tolist(),
                "scale": self.s.tolist()}


# --------------------------------------------------------------------------- #
# 空间场景 (多物体容器)
# --------------------------------------------------------------------------- #
class SpatialScene:
    """多物体空间场景: object_id -> SpatialObject 的有序容器。

    设计纪律:
        * add 拒绝重复 id; get/remove 缺失 id 显式 ValueError;
        * 依赖非空场景的操作 (如最近邻/碰撞) 对空场景抛 ValueError;
        * apply_transform 返回**新**场景 (位置按 R diag(s), 速度仅线性部分,
          球体半径随各向同性缩放; 非各向同性缩放时半径取几何平均等效代理)。
    """

    def __init__(self, objects: Optional[Iterable[SpatialObject]] = None) -> None:
        self._objs: Dict[str, SpatialObject] = {}
        if objects is not None:
            for o in objects:
                self.add(o)

    # -- 增删查 --------------------------------------------------------- #
    def add(self, obj: SpatialObject) -> None:
        if not isinstance(obj, SpatialObject):
            raise ValueError("add 需 SpatialObject 实例")
        if obj.object_id in self._objs:
            raise ValueError(f"物体 id 重复: {obj.object_id}")
        self._objs[obj.object_id] = obj

    def remove(self, object_id: str) -> None:
        if object_id not in self._objs:
            raise ValueError(f"场景中不存在物体: {object_id}")
        del self._objs[object_id]

    def get(self, object_id: str) -> SpatialObject:
        if object_id not in self._objs:
            raise ValueError(f"场景中不存在物体: {object_id}")
        return self._objs[object_id]

    def objects(self) -> List[SpatialObject]:
        return list(self._objs.values())

    def ids(self) -> List[str]:
        return list(self._objs.keys())

    def _require_nonempty(self, what: str = "该操作") -> None:
        if len(self._objs) == 0:
            raise ValueError(f"空场景不支持 {what}")

    # -- 容器协议 ------------------------------------------------------- #
    def __len__(self) -> int:
        return len(self._objs)

    def __contains__(self, object_id: str) -> bool:
        return object_id in self._objs

    def __iter__(self):
        return iter(self._objs.values())

    # -- 批量几何 ------------------------------------------------------- #
    def position_matrix(self) -> np.ndarray:
        """所有物体位置堆叠为 (N,3); 空场景返回 shape (0,3)。"""
        if not self._objs:
            return np.zeros((0, 3), dtype=np.float64)
        return np.stack([o.position for o in self._objs.values()], axis=0)

    def velocity_matrix(self) -> np.ndarray:
        if not self._objs:
            return np.zeros((0, 3), dtype=np.float64)
        return np.stack([o.velocity for o in self._objs.values()], axis=0)

    def radius_vector(self) -> np.ndarray:
        if not self._objs:
            return np.zeros((0,), dtype=np.float64)
        return np.array([o.radius for o in self._objs.values()], dtype=np.float64)

    def pairwise_distances(self) -> np.ndarray:
        """对称距离矩阵 (N,N); N<2 时返回 0x0。"""
        P = self.position_matrix()
        n = P.shape[0]
        if n < 2:
            return np.zeros((0, 0), dtype=np.float64)
        diff = P[:, None, :] - P[None, :, :]
        return np.sqrt((diff ** 2).sum(axis=-1))

    # -- 变换 ----------------------------------------------------------- #
    def apply_transform(self, t: SpatialTransform) -> "SpatialScene":
        """对场景整体施加坐标变换, 返回新场景 (不改原场景)。

        * position = t.apply(p); velocity = t.apply_vectors(v);
        * radius: 非各向同性缩放时取三轴缩放几何平均作为等效球体半径代理。
        """
        if not isinstance(t, SpatialTransform):
            raise ValueError("t 须为 SpatialTransform")
        self._require_nonempty("apply_transform")
        pos = t.apply(self.position_matrix())
        vel = t.apply_vectors(self.velocity_matrix())
        rscale = t.effective_scale
        out = SpatialScene()
        for i, o in enumerate(self._objs.values()):
            out.add(SpatialObject(object_id=o.object_id,
                                  position=pos[i].tolist(),
                                  velocity=vel[i].tolist(),
                                  radius=o.radius * rscale))
        return out

    # -- 序列化 --------------------------------------------------------- #
    def to_dict(self) -> Dict[str, Any]:
        return {"objects": [o.to_dict() for o in self._objs.values()]}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SpatialScene":
        sc = cls()
        for od in d.get("objects", []):
            sc.add(SpatialObject.from_dict(od))
        return sc

    def __repr__(self) -> str:  # pragma: no cover
        return f"SpatialScene(n={len(self._objs)}, ids={self.ids()})"


# --------------------------------------------------------------------------- #
# 多视角正交投影代理 (v3.5.0.dev4)
# --------------------------------------------------------------------------- #
class OrthographicView:
    """正交视角投影代理 (非真实相机, 无畸变/无透视)。

    给定视点 eye 与注视点 look_at, 构造右/上/前三个正交基向量。世界点
    p 投影到像面 (u, v) 并附带沿视线深度 d; unproject(u, v, d) 可精确
    还原世界点 (正交投影无透视歧义)。

    analogy, not reproduction —— 非真实相机标定/内参/畸变。
    """

    def __init__(self, eye: Sequence[float], look_at: Sequence[float],
                 up_hint: Sequence[float] = (0.0, 0.0, 1.0)) -> None:
        self.eye = _as_vec3(eye, "eye")
        target = _as_vec3(look_at, "look_at")
        up_hint = _as_vec3(up_hint, "up_hint")
        forward = target - self.eye
        n = float(np.linalg.norm(forward))
        if n == 0.0:
            raise ValueError("eye 与 look_at 不能重合")
        self.forward = forward / n                       # 视线方向
        right = np.cross(self.forward, up_hint)
        if float(np.linalg.norm(right)) < 1e-9:
            raise ValueError("up_hint 与视线平行, 无法定义像面")
        self.right = right / np.linalg.norm(right)
        self.up = np.cross(self.forward, self.right)    # 正交化的上方向

    # -- 投影 / 反投影 -------------------------------------------------- #
    def project(self, points: Sequence[float] | np.ndarray
                ) -> Dict[str, np.ndarray]:
        """世界点 -> {u, v, depth}。正交投影: u/v 为像面坐标, depth 沿视线。"""
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim == 1:
            pts = pts.reshape(1, 3)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"points 须为 [...,3], 实际 {pts.shape}")
        rel = pts - self.eye                             # (N,3)
        u = rel @ self.right
        v = rel @ self.up
        d = rel @ self.forward
        return {"u": u, "v": v, "depth": d,
                "uv": np.stack([u, v], axis=-1)}

    def unproject(self, uv: Sequence[float] | np.ndarray,
                  depth: float | np.ndarray) -> np.ndarray:
        """像面 (u, v) + 深度 -> 世界点 (N,3)。正交投影精确可逆。

        depth 可为标量 (统一深度) 或与 N 等长的逐点深度。
        """
        uv = np.asarray(uv, dtype=np.float64)
        if uv.ndim == 1:
            uv = uv.reshape(1, 2)
        if uv.ndim != 2 or uv.shape[1] != 2:
            raise ValueError("uv 须为 [...,2]")
        depth = np.asarray(depth, dtype=np.float64)
        if depth.ndim == 0:
            depth = np.full(uv.shape[0], float(depth))
        if depth.shape[0] != uv.shape[0]:
            raise ValueError("depth 长度须与 uv 行数一致")
        return (self.eye
                + np.outer(uv[:, 0], self.right)
                + np.outer(uv[:, 1], self.up)
                + (depth[:, None] * self.forward[None, :]))

    def roundtrip_error(self, points: np.ndarray) -> float:
        """project 后逐点用各自 depth unproject 回世界坐标的最大误差。"""
        proj = self.project(points)
        back = self.unproject(proj["uv"], proj["depth"])
        return float(np.max(np.abs(back - np.asarray(points, dtype=np.float64))))


def multiview_consistency_error(view_a: "OrthographicView",
                                view_b: "OrthographicView",
                                points: np.ndarray) -> float:
    """同一世界点经两视角投影再反投影回世界坐标的最大不一致误差。

    正交投影无透视歧义, 理想误差 ~机器精度; 用于回归视角变换一致性。
    """
    if not isinstance(view_a, OrthographicView) \
            or not isinstance(view_b, OrthographicView):
        raise ValueError("view_a/view_b 须为 OrthographicView")
    pa = view_a.project(points)
    pb = view_b.project(points)
    # 逐点用各自深度反投影 (正交投影无透视歧义)
    back_a = view_a.unproject(pa["uv"], pa["depth"])
    back_b = view_b.unproject(pb["uv"], pb["depth"])
    pts = np.asarray(points, dtype=np.float64)
    return float(max(np.max(np.abs(back_a - pts)),
                     np.max(np.abs(back_b - pts))))
