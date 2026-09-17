"""
合成物理动力学数据集 (v2.0.0 学习闭环的数据层)
=============================================
用四类可解析的一维/三维运动生成 ground-truth 轨迹, 自监督构造
"历史窗口 -> 下一时刻运动学状态" 样本, 供 CTM 物理预测训练:

    uniform   匀速直线:        x = x0 + v t
    accel     匀加速:          x = x0 + v0 t + 1/2 a t^2
    spring    简谐振动:        x = A cos(wt+phi), v = -A w sin(wt+phi)
    collision 一维弹性碰撞:    两质点碰撞瞬间交换速度 (动量/动能守恒)

原始运动学向量 raw = [position(3), velocity(3)] (RAW_DIM=6), 只在 x 轴
产生非平凡运动, y/z 保持初值, 与 PCE 物理 Token 位/速字段同口径。
"""



from __future__ import annotations

import logging

logger = logging.getLogger("udos.dynamics")


import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

RAW_DIM = 6  # position(3) + velocity(3)


def _vec3(x: float, y: float = 0.0, z: float = 0.0) -> List[float]:
    return [x, y, z]


# ---------------------------------------------------------------------------
# 轨迹生成器: 返回 list[(pos[3], vel[3])], 长度 = n_steps
# ---------------------------------------------------------------------------
def traj_uniform(n_steps: int, dt: float, v0: float, x0: float = 0.0
                 ) -> List[Tuple[List[float], List[float]]]:
    out = []
    for t in range(n_steps):
        tt = t * dt
        out.append((_vec3(x0 + v0 * tt), _vec3(v0)))
    return out


def traj_accel(n_steps: int, dt: float, v0: float, a: float, x0: float = 0.0
               ) -> List[Tuple[List[float], List[float]]]:
    out = []
    for t in range(n_steps):
        tt = t * dt
        out.append((_vec3(x0 + v0 * tt + 0.5 * a * tt * tt),
                    _vec3(v0 + a * tt)))
    return out


def traj_spring(n_steps: int, dt: float, amp: float, omega: float,
                phi: float = 0.0) -> List[Tuple[List[float], List[float]]]:
    out = []
    for t in range(n_steps):
        tt = t * dt
        out.append((_vec3(amp * math.cos(omega * tt + phi)),
                    _vec3(-amp * omega * math.sin(omega * tt + phi))))
    return out


def traj_collision(n_steps: int, dt: float, x1: float, v1: float,
                   x2: float, v2: float) -> List[Tuple[List[float], List[float]]]:
    """两个等质量质点一维弹性碰撞: 相遇时刻交换速度。"""
    p1, p2, u1, u2 = x1, x2, v1, v2
    swapped = False
    out = []
    for t in range(n_steps):
        # 相遇 (p1 从左追上/迎上 p2) 且尚未碰撞 -> 交换速度
        if not swapped and p1 >= p2:
            u1, u2 = u2, u1
            swapped = True
        out.append((_vec3(p1), _vec3(u1)))  # 主体轨迹记录质点1
        # 同步推进两质点 (主体轨迹只记录质点1, 但推进质点2以判碰撞)
        p1 += u1 * dt
        p2 += u2 * dt
    return out


@dataclass
class DynamicsDataset:
    """
    滑窗自监督数据集。
        X: [N, window, RAW_DIM]  历史窗口运动学序列
        y: [N, RAW_DIM]          窗口下一时刻的 ground-truth 状态
        kinds: [N]               每条样本的运动类型
    """

    X: torch.Tensor
    y: torch.Tensor
    kinds: List[str]
    class_names: Tuple[str, ...] = ("uniform", "accel", "spring", "collision")

    def __len__(self) -> int:
        return self.X.size(0)

    def split(self, ratio: float = 0.8) -> Tuple["DynamicsDataset", "DynamicsDataset"]:
        n = len(self)
        cut = int(n * ratio)
        idx = torch.randperm(n, generator=torch.Generator().manual_seed(0))
        tr, te = idx[:cut], idx[cut:]
        return (DynamicsDataset(self.X[tr], self.y[tr], [self.kinds[i] for i in tr.tolist()],
                                self.class_names),
                DynamicsDataset(self.X[te], self.y[te], [self.kinds[i] for i in te.tolist()],
                                self.class_names))

    def batches(self, batch_size: int, shuffle: bool = True, seed: int = 0):
        n = len(self)
        idx = torch.randperm(n, generator=torch.Generator().manual_seed(seed)) if shuffle \
            else torch.arange(n)
        for s in range(0, n, batch_size):
            b = idx[s:s + batch_size]
            yield self.X[b], self.y[b]


def _raw(pos: Sequence[float], vel: Sequence[float]) -> List[float]:
    return list(pos[:3]) + list(vel[:3])


def build_dynamics_dataset(n_per_kind: int = 64, n_steps: int = 12,
                           window: int = 8, dt: float = 0.5
                           ) -> DynamicsDataset:
    """
    为每类运动随机采样 n_per_kind 条轨迹, 每条滑窗切样本。
    随机参数在确定性网格 + 抖动内取, 保证可复现。
    """
    g = torch.Generator().manual_seed(42)
    Xs, ys, kinds = [], [], []

    def push(traj, kind):
        raw = torch.tensor([_raw(p, v) for p, v in traj], dtype=torch.float32)
        for s in range(raw.size(0) - window):
            Xs.append(raw[s:s + window])
            ys.append(raw[s + window])
            kinds.append(kind)

    for i in range(n_per_kind):
        jitter = lambda lo, hi: lo + (hi - lo) * torch.rand(1, generator=g).item()
        push(traj_uniform(n_steps, dt, v0=jitter(-2, 2), x0=jitter(-1, 1)), "uniform")
        push(traj_accel(n_steps, dt, v0=jitter(-1, 1), a=jitter(-1.5, 1.5)), "accel")
        push(traj_spring(n_steps, dt, amp=jitter(0.5, 2.0),
                         omega=jitter(0.5, 1.5), phi=jitter(-1, 1)), "spring")
        push(traj_collision(n_steps, dt, x1=-2.0, v1=jitter(1.0, 2.5),
                            x2=jitter(1.0, 2.5), v2=jitter(-0.5, 0.5)), "collision")

    X = torch.stack(Xs, 0)
    y = torch.stack(ys, 0)
    return DynamicsDataset(X, y, kinds)


def naive_baseline_mse(dataset: DynamicsDataset) -> float:
    """朴素基线: 预测'状态不变'(最后一帧=下一帧), 作为训练增益对照。"""
    last = dataset.X[:, -1, :]
    return torch.mean((last - dataset.y) ** 2).item()


# ===========================================================================
# v2.1.0 参数化场景 + 多步目标 (多步滚动推演 / 场景条件联合训练的数据层)
# ===========================================================================
# 场景物理参数向量, 固定槽位顺序; 这些是"仅凭观测窗口无法唯一确定"的隐藏量,
# 作为 GPM 场景条件信号: 模型需要它才能准确外推 (尤其 spring.omega / collision.v2)。
SCENE_PARAM_NAMES = ("v0", "accel_a", "spring_omega", "other_v2")
SCENE_PARAM_DIM = len(SCENE_PARAM_NAMES)


@dataclass
class ParametricDynamicsDataset:
    """
        X:     [N, W, RAW_DIM]      历史窗口
        Y:     [N, H, RAW_DIM]      未来 H 步 ground-truth (多步目标)
        P:     [N, SCENE_PARAM_DIM] 场景隐藏物理参数
        kinds: [N]
    horizon=1 时 Y[:,0] 即 v2.0 的单步目标 y。
    """

    X: torch.Tensor
    Y: torch.Tensor
    P: torch.Tensor
    kinds: List[str]
    dt: float = 0.5
    class_names: Tuple[str, ...] = ("uniform", "accel", "spring", "collision")

    def __len__(self) -> int:
        return self.X.size(0)

    @property
    def horizon(self) -> int:
        return self.Y.size(1)

    def split(self, ratio: float = 0.8
              ) -> Tuple["ParametricDynamicsDataset", "ParametricDynamicsDataset"]:
        n = len(self)
        idx = torch.randperm(n, generator=torch.Generator().manual_seed(0))
        tr, te = idx[:int(n * ratio)], idx[int(n * ratio):]
        mk = lambda ii: ParametricDynamicsDataset(
            self.X[ii], self.Y[ii], self.P[ii],
            [self.kinds[i] for i in ii.tolist()], self.dt, self.class_names)
        return mk(tr), mk(te)

    def batches(self, batch_size: int, shuffle: bool = True, seed: int = 0):
        n = len(self)
        idx = torch.randperm(n, generator=torch.Generator().manual_seed(seed)) \
            if shuffle else torch.arange(n)
        for s in range(0, n, batch_size):
            b = idx[s:s + batch_size]
            yield self.X[b], self.P[b], self.Y[b]

    def kind_mask(self, kind: str) -> torch.Tensor:
        return torch.tensor([k == kind for k in self.kinds], dtype=torch.bool)


def _sample_parametric(kind: str, g: "torch.Generator",
                       n_steps: int, dt: float
                       ) -> Tuple[List[Tuple[List[float], List[float]]], List[float]]:
    """采样一条参数化轨迹, 返回 (轨迹, 场景参数向量[4])。"""
    jit = lambda lo, hi: lo + (hi - lo) * torch.rand(1, generator=g).item()
    if kind == "uniform":
        v0 = jit(-2, 2)
        return traj_uniform(n_steps, dt, v0=v0, x0=jit(-1, 1)), [v0, 0.0, 0.0, 0.0]
    if kind == "accel":
        v0, a = jit(-1, 1), jit(-1.5, 1.5)
        return traj_accel(n_steps, dt, v0=v0, a=a), [v0, a, 0.0, 0.0]
    if kind == "spring":
        omega = jit(0.6, 1.6)               # 隐藏角频率: 短窗口不足一周期, 难从窗口反推
        return traj_spring(n_steps, dt, amp=jit(0.5, 2.0), omega=omega,
                           phi=jit(-1, 1)), [0.0, 0.0, omega, 0.0]
    if kind == "collision":
        v1, v2 = jit(1.0, 2.5), jit(-0.5, 0.5)  # v2 是被撞质点速度, 主体窗口完全看不到
        return traj_collision(n_steps, dt, x1=-2.0, v1=v1,
                              x2=jit(1.0, 2.5), v2=v2), [v1, 0.0, 0.0, v2]
    raise ValueError(f"unknown kind {kind}")


def build_parametric_dataset(n_per_kind: int = 48, n_steps: int = 14,
                             window: int = 6, horizon: int = 4,
                             dt: float = 0.5, seed: int = 42
                             ) -> ParametricDynamicsDataset:
    """构造 (历史窗口 -> 未来 H 步) + 场景隐藏参数的参数化数据集。"""
    assert n_steps >= window + horizon, "轨迹长度需容纳窗口与多步未来"
    g = torch.Generator().manual_seed(seed)
    Xs, Ys, Ps, kinds = [], [], [], []
    for _ in range(n_per_kind):
        for kind in ("uniform", "accel", "spring", "collision"):
            traj, params = _sample_parametric(kind, g, n_steps, dt)
            raw = torch.tensor([_raw(p, v) for p, v in traj], dtype=torch.float32)
            pv = torch.tensor(params, dtype=torch.float32)
            for s in range(raw.size(0) - window - horizon + 1):
                Xs.append(raw[s:s + window])
                Ys.append(raw[s + window:s + window + horizon])
                Ps.append(pv)
                kinds.append(kind)
    return ParametricDynamicsDataset(torch.stack(Xs, 0), torch.stack(Ys, 0),
                                     torch.stack(Ps, 0), kinds, dt=dt)


def kinematic_residual(pred: torch.Tensor, prev_state: torch.Tensor,
                       dt: float) -> torch.Tensor:
    """
    运动学一致性残差 (v2.1 物理一致性诊断): 一阶欧拉应有 x(t+1)=x(t)+v(t)*dt,
    用**前一帧速度** v(t) 积分 (而非预测时刻速度, 后者对加速运动不成立)。
    pred/prev_state 最后一维布局 [pos(3), vel(3)]; 返回每样本位置维 RMS 残差。
    注意: 这是一阶近似——匀速段严格为 0, 加速/振动段真值本身有 O(a*dt^2) 偏离,
    因此应分运动类型解读, 不宜作为混合运动的全局硬损失。
    """
    violation = pred[..., :3] - prev_state[..., :3] - prev_state[..., 3:6] * dt
    return violation.pow(2).mean(dim=-1).sqrt()


def noise_augment(x: torch.Tensor, sigma: float,
                  generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """
    v2.4.3 训练数据增强: 向输入窗口 x 注入高斯噪声 (零均值, 标准差 sigma)。
    sigma=0 时原样返回 x (等价旧版, 不增加任何计算); sigma>0 时
    x' = x + sigma * N(0,1)。确定性 (受全局/传入 generator 种子控制)。
    """
    if sigma is None or sigma <= 0.0:
        return x
    noise = torch.randn(x.shape, generator=generator, dtype=x.dtype,
                        device=x.device)
    return x + sigma * noise
