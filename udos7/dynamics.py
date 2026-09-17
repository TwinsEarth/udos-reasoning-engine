"""v7 合成动力学：轨迹级严格三分 + 真三维方向运动。

与旧 udos.dynamics 的根本区别：
1. **轨迹级划分**：先按独立 seed 生成整条轨迹，再在轨迹内切窗；train/val/test
   绝不共享同一条轨迹的相邻窗口（旧版按窗口随机切分存在跨集泄漏）。
2. **真三维**：匀速/加速/弹簧沿随机三维单位方向运动，px/py/pz 三轴都非平凡；
   碰撞仍是 x 轴两体（记录质点 1），其 y/z 平凡在指标里诚实标注。
3. 每个窗口带 traj_id，可审计分组、证明无泄漏。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

from .contracts import (DT, HORIZON, KINDS, SCENE_DIM, STATE_DIM, WINDOW,
                        TRAIN_SEED, VAL_SEED, TEST_SEED, CALIB_SEED)


def _unit_dir(g: torch.Generator) -> torch.Tensor:
    """随机三维单位方向（各轴非平凡）。"""
    v = torch.randn(3, generator=g)
    while v.norm() < 1e-6:
        v = torch.randn(3, generator=g)
    return v / v.norm()


def traj3d_uniform(T: int, dt: float, v0: float, d: torch.Tensor,
                   x0: torch.Tensor) -> torch.Tensor:
    states = torch.zeros(T, STATE_DIM)
    for t in range(T):
        tt = t * dt
        states[t, :3] = x0 + v0 * d * tt
        states[t, 3:6] = v0 * d
    return states


def traj3d_accel(T: int, dt: float, v0: float, a: float, d: torch.Tensor,
                 x0: torch.Tensor) -> torch.Tensor:
    states = torch.zeros(T, STATE_DIM)
    for t in range(T):
        tt = t * dt
        states[t, :3] = x0 + v0 * d * tt + 0.5 * a * d * tt * tt
        states[t, 3:6] = (v0 + a * tt) * d
    return states


def traj3d_spring(T: int, dt: float, amp: float, omega: float, phi: float,
                  d: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
    states = torch.zeros(T, STATE_DIM)
    for t in range(T):
        tt = t * dt
        states[t, :3] = x0 + amp * math.cos(omega * tt + phi) * d
        states[t, 3:6] = -amp * omega * math.sin(omega * tt + phi) * d
    return states


def traj3d_collision(T: int, dt: float, x1: float, v1: float, x2: float,
                     v2: float) -> torch.Tensor:
    """等质量两体一维弹性碰撞（x 轴），记录质点 1；y/z 恒 0。"""
    states = torch.zeros(T, STATE_DIM)
    p1, p2, u1, u2 = x1, x2, v1, v2
    swapped = False
    for t in range(T):
        if not swapped and p1 >= p2:
            u1, u2 = u2, u1
            swapped = True
        states[t, 0] = p1
        states[t, 3] = u1
        p1 += u1 * dt
        p2 += u2 * dt
    return states


@dataclass
class TrajectoryDataset:
    X: torch.Tensor            # [N,W,6]
    Y: torch.Tensor            # [N,H,6]
    P: torch.Tensor            # [N,4]
    kinds: List[str]
    traj_ids: List[int]
    dt: float = DT

    def __len__(self) -> int:
        return self.X.size(0)

    @property
    def horizon(self) -> int:
        return self.Y.size(1)

    def kind_mask(self, kind: str) -> torch.Tensor:
        return torch.tensor([k == kind for k in self.kinds], dtype=torch.bool)

    def kind_indices(self) -> Dict[str, torch.Tensor]:
        return {k: self.kind_mask(k) for k in KINDS}

    def trajectories(self) -> int:
        return len(set(self.traj_ids))


# 各运动类型下“有意义”的场景参数槽位（槽序 v0, accel_a, spring_omega, other_v2）。
# collision 的 v0 槽承载 v1（窗口可观），other_v2 为被撞质点速度（窗口不可观）。
ACTIVE_SLOTS: Dict[str, tuple] = {
    "uniform": (1, 0, 0, 0),
    "accel": (1, 1, 0, 0),
    "spring": (0, 0, 1, 0),
    "collision": (1, 0, 0, 1),
}


def active_param_mask(kinds) -> torch.Tensor:
    """kinds: list[str] -> [N,4] 有效槽位掩码（参数辨识损失只在有效槽上计）。"""
    return torch.tensor([ACTIVE_SLOTS[k] for k in kinds], dtype=torch.float32)


def _jit(g: torch.Generator, lo: float, hi: float) -> float:
    return lo + (hi - lo) * float(torch.rand(1, generator=g))


def _sample_trajectory(kind: str, g: torch.Generator, T: int, dt: float
                       ) -> Tuple[torch.Tensor, List[float]]:
    if kind == "collision":
        v1, v2 = _jit(g, 1.0, 2.5), _jit(g, -0.5, 0.5)
        st = traj3d_collision(T, dt, x1=-2.0, v1=v1,
                              x2=_jit(g, 1.0, 2.5), v2=v2)
        return st, [v1, 0.0, 0.0, v2]
    d = _unit_dir(g)
    x0 = torch.randn(3, generator=g) * 0.3
    if kind == "uniform":
        v0 = _jit(g, -2.0, 2.0)
        return traj3d_uniform(T, dt, v0, d, x0), [v0, 0.0, 0.0, 0.0]
    if kind == "accel":
        v0, a = _jit(g, -1.0, 1.0), _jit(g, -1.5, 1.5)
        return traj3d_accel(T, dt, v0, a, d, x0), [v0, a, 0.0, 0.0]
    if kind == "spring":
        omega = _jit(g, 0.6, 1.6)
        return (traj3d_spring(T, dt, _jit(g, 0.5, 2.0), omega,
                             _jit(g, -1.0, 1.0), d, x0),
                [0.0, 0.0, omega, 0.0])
    raise ValueError(kind)


def build_split(seed: int, n_traj_per_kind: int = 64,
                window: int = WINDOW, horizon: int = HORIZON, dt: float = DT,
                margin: int = 4, traj_id_offset: int = 0) -> TrajectoryDataset:
    """按独立 seed 生成一整个 split（轨迹级，无跨集窗口泄漏）。

    traj_id_offset 让不同 split 的轨迹 ID 全局不重叠，可审计无泄漏。
    """
    g = torch.Generator().manual_seed(seed)
    T = window + horizon + margin
    Xs, Ys, Ps, kinds, tids = [], [], [], [], []
    traj_id = traj_id_offset
    for kind in KINDS:
        for _ in range(n_traj_per_kind):
            states, params = _sample_trajectory(kind, g, T, dt)
            pv = torch.tensor(params, dtype=torch.float32)
            for s in range(states.size(0) - window - horizon + 1):
                Xs.append(states[s:s + window])
                Ys.append(states[s + window:s + window + horizon])
                Ps.append(pv)
                kinds.append(kind)
                tids.append(traj_id)
            traj_id += 1
    return TrajectoryDataset(torch.stack(Xs), torch.stack(Ys),
                             torch.stack(Ps), kinds, tids, dt)


def three_way_splits(n_traj_per_kind: int = 64, window: int = WINDOW,
                     horizon: int = HORIZON, dt: float = DT
                     ) -> Dict[str, TrajectoryDataset]:
    """train/val/test 三分 + conformal 校准集（均独立 seed、独立轨迹，ID 不重叠）。"""
    n_val = n_traj_per_kind // 2 or 1
    n_test = n_traj_per_kind // 2 or 1
    n_cal = n_traj_per_kind // 2 or 1
    n_kind = len(KINDS)
    train = build_split(TRAIN_SEED, n_traj_per_kind, window, horizon, dt,
                        traj_id_offset=0)
    val = build_split(VAL_SEED, n_val, window, horizon, dt,
                      traj_id_offset=n_traj_per_kind * n_kind)
    test = build_split(TEST_SEED, n_test, window, horizon, dt,
                       traj_id_offset=(n_traj_per_kind + n_val) * n_kind)
    calib = build_split(CALIB_SEED, n_cal, window, horizon, dt,
                        traj_id_offset=(n_traj_per_kind + n_val + n_test)
                                        * n_kind)
    return {"train": train, "val": val, "test": test, "calib": calib}
