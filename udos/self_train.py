"""
UDOS 完全自训练线 (Fully Self-Training / RSI 一代闭环) (v4.2.0 起)
=====================================================================
analogy, not reproduction —— 受 RSI (Recursive Self-Improvement) / 自蒸馏 /
世界模型自数据思想启发的**合成低维类比实现**, 不宣称复现任何大规模自训练系统;
全部 CPU 合成数据, 外挂优先零梯度, 默认 opt-in。

设计纪律 (与全工程一致):
    * 纯前向、确定性; 推理路径整体 @torch.no_grad, 只读冻结主预测器, 不改主权重。
    * 所有可学外挂小模型 (student 等) **不入主 predictor.state_dict**, 主参数恒 52191。
    * 空/非法输入显式 ValueError, 绝不静默污染; NaN/inf 显式拒绝。
    * 日志走 logging_config (默认 stderr), 绝不写 stdout / HTTP 体 / metrics。
    * **重点防退化**: teacher→student 必须量化真单调改进 vs 漂移/崩塌; 收益不稳或被
      反证的机制照实 opt-in 并留候选账本, 不为讲故事而宣称 RSI 必然提升。
      (历史教训: 历代自蒸馏学生 mse=2.63 已被 REJECT, 引以为戒。)

本文件按 v4.2 线 10 节点逐步追加:
    4.2.0   TransitionTripletGenerator      世界模型自生成 (s,a,s') 三元组骨架
    dev1    SelfPlayExplorer + rule check  自博弈探索 + 守恒/约束规则校验
    dev2    ExecutionValidator + ModelReviewer  执行验证 + ensemble/calibration 评审
    dev3    HumanAuditHook + DataQualityLedger  人工抽检 hook + 数据质量账本
    dev4    TeacherStudentLoop              teacher→student 一代可验证闭环
    dev5    DataRefluxMixer                 数据回流三档配比代理
    dev6    DegradationDetector + RollbackManager  退化检测与回滚
    4.2.1   DiminishingReturnsCriterion + SelfTrainAB  收益递减 + 自训练 A/B
    4.2.2   (HTTP 端点在 server.py 集成)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .dynamics import RAW_DIM
from .world_model import LatentWorldModel
from .wm_conservation import ConservationChecker

logger = logging.getLogger("udos.self_train")

# 自训练线内部阶段标记 (CHANGELOG 用; 全局 __version__ 仅在正式训练点 bump)
LINE_STAGES = (
    "4.2.0", "4.2.0.dev1", "4.2.0.dev2", "4.2.0.dev3", "4.2.0.dev4",
    "4.2.0.dev5", "4.2.0.dev6", "4.2.1", "4.2.2", "4.2.9",
)


class SelfGeneratedTriplets:
    """一次自生成 (state, action, next_state) 三元组批次 (只读容器)。

    属性
    ----
    states      : [N, RAW_DIM]  窗口末帧物理状态 s_t (教师侧真实锚定)
    actions     : [N, action_dim]  施加的动作 a_t (自博弈/确定性采样)
    next_states : [N, RAW_DIM]  世界模型想象的下一物理状态 s_{t+1}
    scene_params: [N, P_DIM] or None  透传的场景隐藏参数
    kinds       : [N]  来源轨迹类别标签
    origin      : str  三元组来源标记 ("self_generated")
    """

    __slots__ = ("states", "actions", "next_states", "scene_params",
                 "kinds", "origin")

    def __init__(self, states: torch.Tensor, actions: torch.Tensor,
                 next_states: torch.Tensor,
                 scene_params: Optional[torch.Tensor],
                 kinds: List[str]):
        if states.ndim != 2 or states.size(-1) != RAW_DIM:
            raise ValueError(f"states 形状应为 [N,{RAW_DIM}], 得 {tuple(states.shape)}")
        if next_states.shape != states.shape:
            raise ValueError(
                f"next_states 形状应与 states 一致 {tuple(states.shape)}, "
                f"得 {tuple(next_states.shape)}")
        if actions.ndim != 2:
            raise ValueError(f"actions 应为 [N, action_dim], 得 {tuple(actions.shape)}")
        if states.size(0) != actions.size(0) or states.size(0) != next_states.size(0):
            raise ValueError("states/actions/next_states 第一维须一致")
        if not (torch.isfinite(states).all() and torch.isfinite(actions).all()
                and torch.isfinite(next_states).all()):
            raise ValueError("三元组含 NaN/inf, 拒绝自生成污染")
        if scene_params is not None and scene_params.size(0) != states.size(0):
            raise ValueError("scene_params 第一维须与 states 一致")
        self.states = states
        self.actions = actions
        self.next_states = next_states
        self.scene_params = scene_params
        self.kinds = list(kinds)
        self.origin = "self_generated"

    def __len__(self) -> int:
        return int(self.states.size(0))

    @property
    def action_dim(self) -> int:
        return int(self.actions.size(-1))

    def describe(self) -> Dict[str, Any]:
        return {
            "n": len(self),
            "action_dim": self.action_dim,
            "origin": self.origin,
            "states_shape": list(self.states.shape),
            "next_states_shape": list(self.next_states.shape),
            "finite": True,
        }


class TransitionTripletGenerator:
    """世界模型自生成 (state, action, next_state) 三元组骨架 (v4.2.0)。

    只读外挂: 持有冻结主预测器 ``predictor`` 与**已拟合**的 LatentWorldModel
    (action_dim>0)。纯前向零梯度, 不入主 state_dict。

    三元组构造 (一步转移):
        s_t      = 物理窗口末帧 (教师侧真实锚定, 来自真实 dataset.X)
        a_t      = 动作 (v4.2.0 骨架: 确定性均匀网格 [-1,0,+1] 冲量类比;
                   dev1 升级为自博弈探索)
        s_{t+1}  = decode( transit( encode(window), a_t ) )  —— 世界模型想象的
                   下一物理状态 (外挂转移+读出, 零梯度)。

    类比 RSI: 世界模型自己 rollout 出 (s,a,s') 训练样本, 不再逐条依赖人工标注。
    v4.2.0 仅交付**骨架与可复算质量度量**, 不宣称任何 student 改进 (防退化纪律)。

    Parameters
    ----------
    predictor:
        已训练 PhysicsPredictor (只读, eval)。
    world_model:
        已 ``fit`` 的 LatentWorldModel, 须 action_dim>0。
    action_dim:
        动作维度 (须与 world_model.action_dim 一致)。
    action_grid:
        v4.2.0 确定性动作网格 (1D 冲量类比); 默认 (-1.0, 0.0, +1.0)。
    """

    def __init__(self, predictor, world_model: LatentWorldModel,
                 action_dim: int = 1,
                 action_grid: Sequence[float] = (-1.0, 0.0, 1.0)):
        if not isinstance(world_model, LatentWorldModel):
            raise ValueError("world_model 须为 LatentWorldModel 实例")
        if not world_model.fitted:
            raise ValueError("world_model 须先 fit (自生成依赖已拟合转移+读出)")
        self.predictor = predictor
        self.wm = world_model
        # v4.2.0 骨架: 世界模型走无动作自治转移 (兼容 wm.fit); action 是自博弈
        # 独立通道的候选动作标签。若 wm 已配置 action_dim>0 (后续 dev 扩展),
        # generate 会把动作真正喂进转移。
        self.wm_action_dim = int(world_model.action_dim)
        self.action_dim = int(action_dim)
        if self.wm_action_dim > 0 and self.wm_action_dim != self.action_dim:
            raise ValueError(
                f"world_model.action_dim={self.wm_action_dim} 与 action_dim="
                f"{self.action_dim} 不一致")
        self.action_grid = tuple(float(a) for a in action_grid)
        self.predictor.eval()

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _sample_actions(self, n: int, seed: int = 0) -> torch.Tensor:
        """v4.2.0 骨架: 确定性动作网格循环铺开 (1D 冲量类比)。

        不引入随机探索 (dev1 升级为自博弈); 网格循环保证可复算。
        返回 [N, action_dim]。
        """
        g = torch.Generator().manual_seed(seed)
        grid = torch.tensor(self.action_grid, dtype=torch.float32)
        if self.action_dim == 1:
            idx = torch.arange(n) % grid.numel()
            return grid[idx].unsqueeze(-1)
        # action_dim>1: 首维用网格, 其余维零 (骨架足够, dev 节点再扩展)
        cols = [grid[torch.arange(n) % grid.numel()].unsqueeze(-1)]
        for _ in range(1, self.action_dim):
            cols.append(torch.zeros(n, 1))
        return torch.cat(cols, dim=-1)

    @torch.no_grad()
    def generate(self, dataset, n: Optional[int] = None,
                 seed: int = 0) -> SelfGeneratedTriplets:
        """从真实 dataset 的窗口出发, 自生成一批 (s,a,s') 三元组。

        Parameters
        ----------
        dataset:
            ParametricDynamicsDataset (提供 X 窗口 / P 场景参数 / kinds)。
        n:
            生成样本数; None -> 全量 dataset。须为正整数。
        seed:
            动作采样确定性种子。

        Returns
        -------
        SelfGeneratedTriplets
        """
        X = getattr(dataset, "X", None)
        P = getattr(dataset, "P", None)
        kinds = getattr(dataset, "kinds", None)
        if X is None:
            raise ValueError("dataset 须提供 .X (历史窗口)")
        if n is None:
            n = int(X.size(0))
        if not isinstance(n, int) or n <= 0:
            raise ValueError("n 须为正整数")
        n = min(n, int(X.size(0)))

        windows = X[:n]                                   # [N, W, RAW]
        sp = P[:n] if P is not None else None             # [N, P_DIM]
        kinds = list(kinds)[:n] if kinds is not None else ["unknown"] * n

        states = windows[:, -1, :].clone()                # [N, RAW] 窗口末帧 s_t
        actions = self._sample_actions(n, seed=seed)       # [N, action_dim]

        # 世界模型一步想象: encode -> transit(动作, 若已条件化) -> decode
        next_states_list = []
        for i in range(n):
            w = windows[i:i + 1]
            s = sp[i:i + 1] if sp is not None else None
            z = self.wm.encode_latent(w, scene_params=s)
            act = actions[i:i + 1] if self.wm_action_dim > 0 else None
            z1 = self.wm.transit_step(z, act)
            ns = self.wm.decode_latent(z1)                # [1, RAW]
            next_states_list.append(ns)
        next_states = torch.cat(next_states_list, 0)      # [N, RAW]

        return SelfGeneratedTriplets(
            states=states, actions=actions, next_states=next_states,
            scene_params=sp, kinds=kinds)

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def triplet_quality(self, triplets: SelfGeneratedTriplets
                        ) -> Dict[str, float]:
        """自生成三元组的质量度量 (v4.2.0 骨架; 仅描述, 不背书改进)。

        度量全部确定性、可复算:
            n                  样本数
            action_spread      动作离散度 (std), 过低说明探索坍缩
            next_state_finite  是否全有限 (1.0/0.0)
            next_state_std     下一状态分布离散度 (过低 = 想象模式坍缩)
            transition_norm    ||s_{t+1} - s_t|| 均值 (转移活跃度)
            momentum_residual  冲量类比残差: (v_{t+1}-v_t) - a_t 均值绝对值
                               (仅 action_dim>=1 且冲量映射到 x 速度分量时近似)
        """
        s = triplets.states
        ns = triplets.next_states
        a = triplets.actions
        out: Dict[str, float] = {
            "n": float(len(triplets)),
            "action_spread": round(float(a.std()), 6),
            "next_state_finite": 1.0 if bool(torch.isfinite(ns).all()) else 0.0,
            "next_state_std": round(float(ns.std()), 6),
            "transition_norm": round(
                float((ns - s).pow(2).sum(-1).sqrt().mean()), 6),
        }
        # 冲量类比残差: RAW=[pos(3), vel(3)], 速度在索引 3:6; 仅 x 速度(索引3)
        # 受 1D 动作冲量影响。这是骨架级近似, dev1 用 ConservationChecker 严格化。
        if s.size(-1) == RAW_DIM and a.size(-1) >= 1:
            v_t = s[:, 3]            # x 速度
            v_next = ns[:, 3]        # 想象下一 x 速度
            impulse = a[:, 0]
            out["momentum_residual"] = round(
                float((v_next - v_t - impulse).abs().mean()), 6)
        return out


# ====================================================================== #
# dev1: 自博弈探索 + 规则校验 (守恒/约束)
# ====================================================================== #
class SelfPlayExplorer:
    """自博弈探索 (dev1): 策略与世界模型互搏, 选物理自洽的动作。

    类比自博弈 (self-play): 策略 (动作选择) 与世界模型互搏。对每个候选窗口,
    枚举动作网格, 用世界模型想象 next_state (叠加 action 冲量), 再用
    `ConservationChecker` 校验动量/能量守恒违反, 逐样本选**守恒违反最小**的
    动作 (物理自洽性 argmax, 确定性 tie-break)。纯前向、零梯度、主权重只读。

    与 v4.2.0 骨架的区别: action 不再是独立标签, 而是通过**冲量定理**
    Δv_x = a·dt/m 显式注入世界模型想象的下一速度 (牛顿冲量约束), 并用
    ConservationChecker 严格量化守恒违反 (而非骨架级近似残差)。

    Parameters
    ----------
    generator:
        已构造的 TransitionTripletGenerator (复用其 wm/predictor)。
    mass, dt:
        冲量定理 F=m·a 的参数; Δv_x = a·dt/mass。
    action_grid:
        候选动作网格 (1D 冲量); 默认 (-1, 0, +1)。
    """

    def __init__(self, generator: TransitionTripletGenerator,
                 mass: float = 1.0, dt: float = 0.5,
                 action_grid: Sequence[float] = (-1.0, 0.0, 1.0)):
        if not isinstance(generator, TransitionTripletGenerator):
            raise ValueError("generator 须为 TransitionTripletGenerator")
        if mass <= 0:
            raise ValueError("mass 须为正")
        if dt <= 0:
            raise ValueError("dt 须为正")
        self.gen = generator
        self.wm = generator.wm
        self.predictor = generator.predictor
        self.mass = float(mass)
        self.dt = float(dt)
        self.action_grid = tuple(float(a) for a in action_grid)
        self.checker = ConservationChecker(mass=mass)

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _imagine_with_impulse(self, windows: torch.Tensor,
                              scene_params: Optional[torch.Tensor],
                              action: float) -> torch.Tensor:
        """对全批 windows, 用世界模型想象 + 冲量修正, 返回 [N, RAW]。

        世界模型自治转移给出基础 next_state; 叠加冲量定理 Δv_x = a·dt/mass。
        """
        z = self.wm.encode_latent(windows, scene_params=scene_params)
        z1 = self.wm.transit_step(z, None)
        ns = self.wm.decode_latent(z1).clone()       # [N, RAW]
        ns[:, 3] += action * self.dt / self.mass     # 冲量修正 x 速度
        return ns

    @torch.no_grad()
    def explore(self, dataset, n: Optional[int] = None,
                momentum_tol: float = 1e-3,
                energy_tol: float = 1e-2) -> Tuple[SelfGeneratedTriplets, Dict[str, Any]]:
        """自博弈探索: 逐样本选守恒违反最小的 action。

        Returns
        -------
        triplets : SelfGeneratedTriplets  选中动作后的三元组
        report   : dict  自博弈统计 (动作分布/守恒违反/规则通过率)
        """
        X = getattr(dataset, "X", None)
        P = getattr(dataset, "P", None)
        kinds = getattr(dataset, "kinds", None)
        if X is None:
            raise ValueError("dataset 须提供 .X")
        if n is None:
            n = int(X.size(0))
        if not isinstance(n, int) or n <= 0:
            raise ValueError("n 须为正整数")
        n = min(n, int(X.size(0)))

        windows = X[:n]
        sp = P[:n] if P is not None else None
        kinds = list(kinds)[:n] if kinds is not None else ["unknown"] * n
        s_t = windows[:, -1, :].clone()               # [N, RAW]

        grid = self.action_grid
        n_grid = len(grid)
        # 对每个候选 action, 批量算守恒违反
        viol = torch.zeros(n, n_grid)
        ns_by_action = []
        for k, a in enumerate(grid):
            ns = self._imagine_with_impulse(windows, sp, a)
            ns_by_action.append(ns)
            # 拼 [N,2,6] 两步轨迹喂 ConservationChecker
            traj = torch.stack([s_t, ns], dim=1)      # [N,2,6]
            rep = self.checker.check(traj, momentum_tol=momentum_tol,
                                     energy_tol=energy_tol)
            viol[:, k] = torch.tensor(
                [r["momentum_violation"] for r in rep["batch"]])

        # 逐样本选 viol 最小的 action (argmin 天然 tie-break 选首个)
        best_idx = viol.argmin(dim=1)                 # [N] long
        best_list = best_idx.tolist()
        best_ns = torch.stack([ns_by_action[int(k)][i]
                               for i, k in enumerate(best_list)], 0)
        best_a = torch.tensor([[grid[int(k)]] for k in best_list],
                              dtype=torch.float32)    # [N,1]

        trip = SelfGeneratedTriplets(
            states=s_t, actions=best_a, next_states=best_ns,
            scene_params=sp, kinds=kinds)

        # 汇总报告
        chosen_viol = viol.gather(1, best_idx.unsqueeze(1)).squeeze(1)
        action_counts = {f"a={a:+.1f}": int((best_idx == k).sum())
                         for k, a in enumerate(grid)}
        report = {
            "n": int(n),
            "n_actions": n_grid,
            "chosen_action_counts": action_counts,
            "mean_momentum_violation": round(float(chosen_viol.mean()), 6),
            "max_momentum_violation": round(float(chosen_viol.max()), 6),
            "rule_pass_rate": round(
                float((chosen_viol <= momentum_tol).float().mean()), 6),
        }
        return trip, report


# ====================================================================== #
# dev2: 执行验证 + 模型评审 (ensemble/calibration 筛选)
# ====================================================================== #
class ExecutionValidator:
    """执行验证 (dev2): 自生成 next_state 是否"可执行" (不发散/界内)。

    纯前向零梯度检查 (不依赖 window, 只看三元组本身):
        1. next_state 全有限 (NaN/inf -> 拒)
        2. 位置分量在 [pos_bound, pos_bound] 内
        3. 速度分量在 [vel_bound, vel_bound] 内
        4. 转移范数 ||s_{t+1}-s_t|| 不超 max_transition_norm (不发散)
    """

    def __init__(self, pos_bound: float = 10.0, vel_bound: float = 10.0,
                 max_transition_norm: float = 20.0):
        if pos_bound <= 0 or vel_bound <= 0 or max_transition_norm <= 0:
            raise ValueError("边界参数须为正")
        self.pos_bound = float(pos_bound)
        self.vel_bound = float(vel_bound)
        self.max_transition_norm = float(max_transition_norm)

    @torch.no_grad()
    def validate(self, triplets: SelfGeneratedTriplets) -> Dict[str, Any]:
        s = triplets.states
        ns = triplets.next_states
        pos = ns[:, 0:3]
        vel = ns[:, 3:6]
        finite_mask = torch.isfinite(ns).all(dim=1)
        pos_mask = (pos.abs() <= self.pos_bound).all(dim=1)
        vel_mask = (vel.abs() <= self.vel_bound).all(dim=1)
        trans_norm = (ns - s).pow(2).sum(-1).sqrt()
        trans_mask = trans_norm <= self.max_transition_norm
        valid = finite_mask & pos_mask & vel_mask & trans_mask
        return {
            "n": int(len(triplets)),
            "pass": int(valid.sum()),
            "fail": int((~valid).sum()),
            "pass_rate": round(float(valid.float().mean()), 6),
            "finite_fail": int((~finite_mask).sum()),
            "pos_oob": int((~pos_mask).sum()),
            "vel_oob": int((~vel_mask).sum()),
            "transition_diverge": int((~trans_mask).sum()),
        }


class ModelReviewer:
    """模型评审 (dev2): ensemble/calibration 类比筛选高质量自生成样本。

    ensemble 类比 (deep ensemble 思想): 对每个 window 做 n_perturb 次小扰动
    想象, 取 n_perturb 个 next_state 想象的方差作不确定性代理;
    calibration 类比 (保序回归思想): 方差越低 = 置信越高。
    **筛除高方差 (低置信) 样本**, 只留评审通过的高质量三元组。

    纯前向、零梯度、主权重只读。不宣称筛选后必然提升下游 (防退化纪律)。
    """

    def __init__(self, generator: TransitionTripletGenerator,
                 n_perturb: int = 3, noise: float = 0.05):
        if not isinstance(generator, TransitionTripletGenerator):
            raise ValueError("generator 须为 TransitionTripletGenerator")
        if n_perturb < 2:
            raise ValueError("n_perturb 须 >=2 (ensemble 多视角)")
        if noise < 0:
            raise ValueError("noise 须非负")
        self.gen = generator
        self.wm = generator.wm
        self.n_perturb = int(n_perturb)
        self.noise = float(noise)

    @torch.no_grad()
    def _imagine_variance(self, windows: torch.Tensor,
                          scene_params: Optional[torch.Tensor],
                          seed: int) -> torch.Tensor:
        """对每批 window 做 n_perturb 次扰动想象, 返回 [N] 总方差。"""
        g = torch.Generator().manual_seed(seed)
        n = windows.size(0)
        perts = []
        for _ in range(self.n_perturb):
            w_noisy = windows + self.noise * torch.randn(
                windows.shape, generator=g)
            z = self.wm.encode_latent(w_noisy, scene_params=scene_params)
            z1 = self.wm.transit_step(z, None)
            ns = self.wm.decode_latent(z1)
            perts.append(ns)
        stack = torch.stack(perts, dim=0)            # [P, N, RAW]
        var = stack.var(dim=0).sum(dim=-1)           # [N]
        return var

    @torch.no_grad()
    def review(self, dataset, n: Optional[int] = None,
               seed: int = 0) -> Tuple[SelfGeneratedTriplets, Dict[str, Any]]:
        """生成三元组 + ensemble 方差筛选。

        Returns
        -------
        filtered : SelfGeneratedTriplets  评审通过的子集
        report   : dict  评审统计 (保留率/方差分位/筛除数)
        """
        X = getattr(dataset, "X", None)
        P = getattr(dataset, "P", None)
        kinds = getattr(dataset, "kinds", None)
        if X is None:
            raise ValueError("dataset 须提供 .X")
        if n is None:
            n = int(X.size(0))
        if not isinstance(n, int) or n <= 0:
            raise ValueError("n 须为正整数")
        n = min(n, int(X.size(0)))

        windows = X[:n]
        sp = P[:n] if P is not None else None
        kinds = list(kinds)[:n] if kinds is not None else ["unknown"] * n

        trip = self.gen.generate(dataset, n=n, seed=seed)
        var = self._imagine_variance(windows, sp, seed=seed)   # [N]

        # calibration 类比: 以方差中位数为阈值, 保序筛除高方差样本
        thr = float(var.median())
        keep = var <= thr

        filtered = SelfGeneratedTriplets(
            states=trip.states[keep], actions=trip.actions[keep],
            next_states=trip.next_states[keep],
            scene_params=(sp[keep] if sp is not None else None),
            kinds=[k for k, kp in zip(kinds, keep.tolist()) if kp])

        report = {
            "n_total": int(n),
            "n_kept": int(keep.sum()),
            "n_rejected": int((~keep).sum()),
            "keep_rate": round(float(keep.float().mean()), 6),
            "variance_median": round(thr, 6),
            "variance_mean_kept": round(
                float(var[keep].mean()) if keep.any() else 0.0, 6),
            "variance_mean_rejected": round(
                float(var[~keep].mean()) if (~keep).any() else 0.0, 6),
        }
        return filtered, report


# ====================================================================== #
# dev3: 人工抽检 hook + 数据质量账本
# ====================================================================== #
class HumanAuditHook:
    """人工抽检 hook (dev3): 把自生成样本确定性抽样式交回抽检, 记录决策。

    纯治理工具, 不改数据、不训练。`sample` 按 sample_rate 用种子确定性抽出
    待审索引 (可复算); `record_decision` 记录每条抽检的 approve/reject/pending
    决策 (人工或自动标注)。**不自动通过任何样本**——抽检是治理门, 不是捷径。
    """

    VALID_DECISIONS = ("approve", "reject", "pending")

    def __init__(self, sample_rate: float = 0.1, seed: int = 0):
        if not (0.0 < sample_rate <= 1.0):
            raise ValueError("sample_rate 须在 (0, 1]")
        self.sample_rate = float(sample_rate)
        self.seed = int(seed)

    def sample(self, triplets: SelfGeneratedTriplets
               ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """确定性抽样式, 返回待审索引 [M] 与抽样报告。"""
        n = len(triplets)
        if n == 0:
            raise ValueError("triplets 为空, 无可抽检样本")
        g = torch.Generator().manual_seed(self.seed)
        m = max(1, int(round(n * self.sample_rate)))
        m = min(m, n)
        idx = torch.randperm(n, generator=g)[:m].sort().values
        return idx, {
            "n_total": int(n), "n_audited": int(m),
            "sample_rate": self.sample_rate,
            "audit_indices": idx.tolist()}

    def record_decision(self, indices: Sequence[int],
                        decisions: Sequence[str]) -> Dict[str, int]:
        """记录抽检决策; 决策须 ∈ {approve/reject/pending}。"""
        if len(indices) != len(decisions):
            raise ValueError("indices 与 decisions 长度须一致")
        counts = {"approve": 0, "reject": 0, "pending": 0}
        for d in decisions:
            if d not in self.VALID_DECISIONS:
                raise ValueError(
                    f"非法决策 {d!r}, 须 ∈ {self.VALID_DECISIONS}")
            counts[d] += 1
        return counts


class DataQualityLedger:
    """数据质量账本 (dev3): 逐批次记录自生成数据的完整质量画像。

    纯记录工具 (不训练、不改数据)。每个批次落一条 entry:
        batch_id / n / 三元组质量 / 执行验证 / 模型评审 / 抽检决策 / 时间戳
    `to_json` 落盘到 benchmarks/results/ (可复算); `summary` 给聚合视图。
    日志走 logging_config (stderr), 不写 stdout。
    """

    def __init__(self) -> None:
        self.entries: List[Dict[str, Any]] = []

    def log(self, batch_id: str, n: int,
            triplet_quality: Optional[Dict[str, float]] = None,
            execution: Optional[Dict[str, Any]] = None,
            review: Optional[Dict[str, Any]] = None,
            audit: Optional[Dict[str, int]] = None,
            note: str = "") -> Dict[str, Any]:
        if not isinstance(batch_id, str) or not batch_id:
            raise ValueError("batch_id 须为非空字符串")
        entry: Dict[str, Any] = {
            "batch_id": batch_id, "n": int(n), "note": note,
        }
        if triplet_quality is not None:
            entry["triplet_quality"] = dict(triplet_quality)
        if execution is not None:
            entry["execution"] = dict(execution)
        if review is not None:
            entry["review"] = dict(review)
        if audit is not None:
            entry["audit"] = dict(audit)
        self.entries.append(entry)
        logger.info("数据质量账本记录批次 %s (n=%d)", batch_id, n)
        return entry

    def summary(self) -> Dict[str, Any]:
        return {
            "n_batches": len(self.entries),
            "total_samples": sum(int(e["n"]) for e in self.entries),
            "batch_ids": [e["batch_id"] for e in self.entries],
        }

    def to_json(self, path: str) -> str:
        import json
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"entries": self.entries,
                       "summary": self.summary()},
                      f, ensure_ascii=False, indent=2)
        logger.info("数据质量账本落盘 %s (%d 批次)", path, len(self.entries))
        return path


# ====================================================================== #
# dev4: teacher -> student 自训练递归 (仅一代可验证闭环)
# ====================================================================== #
class _StudentHead(nn.Module):
    """外挂 student 单帧预测头 (不入主 state_dict, 主参恒 52191)。

    输入 s_t [RAW_DIM=6], 输出 s_{t+1} [RAW_DIM=6]。末层零初始化 =>
    未训练时输出恒等 (不引入随机偏置)。
    """

    def __init__(self, in_dim: int = RAW_DIM, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, in_dim))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)     # 残差形式


class TeacherStudentLoop:
    """teacher→student 一代自训练闭环 (dev4)。

    **这是本线最高优先级防退化节点。** 历史教训: 历代自蒸馏学生 mse=2.63
    已被 REJECT (学生远差于教师)。本闭环必须**诚实量化** student 在**真实
    holdout** 上的 mse 是否真的低于 teacher, 而非在自生成数据上自欺。

    机制:
        teacher  = 冻结主预测器 (只读, eval_mse 锚点)
        student  = 外挂 `_StudentHead` (单帧->单帧小 MLP), 在自生成三元组
                   (s_t -> s_{t+1}) 上拟合; **不入主 state_dict**。
    判定 (在同一真实 holdout dataset 上, 单步 mse):
        improved : student_mse <= teacher_mse * (1 - margin)   # 真单调改进
        degraded : student_mse >  teacher_mse                # 漂移/崩塌
        collapsed: student_mse >>  teacher_mse (如 >10x)      # 严重崩塌
    若 student 未证明改进, 照实报 degraded/collapsed 并留候选账本, **不宣称 RSI 提升**。
    """

    def __init__(self, teacher, triplets: SelfGeneratedTriplets,
                 hidden: int = 32, lr: float = 1e-2, epochs: int = 60,
                 seed: int = 0, margin: float = 0.01):
        self.teacher = teacher
        self.teacher.eval()
        if not isinstance(triplets, SelfGeneratedTriplets) or len(triplets) == 0:
            raise ValueError("triplets 须为非空 SelfGeneratedTriplets")
        if epochs <= 0:
            raise ValueError("epochs 须为正")
        self.trip = triplets
        self.student = _StudentHead(hidden=hidden)
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.seed = int(seed)
        self.margin = float(margin)
        self._fitted = False
        self.train_loss_history: List[float] = []

    @property
    def n_params_student(self) -> int:
        return self.student.n_params

    def fit(self) -> Dict[str, float]:
        """在自生成 (s_t -> s_{t+1}) 上拟合 student。只动 student 参数。"""
        g = torch.Generator().manual_seed(self.seed)
        opt = torch.optim.AdamW(self.student.parameters(), lr=self.lr,
                                weight_decay=1e-4)
        s = self.trip.states
        ns = self.trip.next_states
        n = s.size(0)
        first = last = float("nan")
        for ep in range(self.epochs):
            idx = torch.randperm(n, generator=g)
            ep_loss = 0.0
            nb = 0
            for i in range(0, n, 32):
                b = idx[i:i + 32]
                opt.zero_grad()
                pred = self.student(s[b])
                loss = ((pred - ns[b]) ** 2).mean()
                loss.backward()
                opt.step()
                ep_loss += float(loss.detach())
                nb += 1
            ep_loss /= max(nb, 1)
            if ep == 0:
                first = ep_loss
            last = ep_loss
            self.train_loss_history.append(ep_loss)
        self.student.eval()
        self._fitted = True
        return {"first_loss": round(first, 6), "last_loss": round(last, 6),
                "student_n_params": self.student.n_params}

    @torch.no_grad()
    def student_mse_on(self, dataset) -> float:
        """student 在真实 dataset 上的单步 mse (输入窗口末帧, 输出下一状态)。"""
        if not self._fitted:
            raise ValueError("student 须先 fit")
        X = dataset[:, -1, :] if torch.is_tensor(dataset) else None
        # 接受 ParametricDynamicsDataset
        windows = getattr(dataset, "X", None)
        Y = getattr(dataset, "Y", None)
        if windows is None or Y is None:
            raise ValueError("dataset 须提供 .X/.Y")
        s_last = windows[:, -1, :]              # [N, RAW]
        target = Y[:, 0, :]                     # [N, RAW] 真实下一状态
        pred = self.student(s_last)
        return round(float(((pred - target) ** 2).mean()), 6)

    @torch.no_grad()
    def compare(self, dataset, teacher_mse: float) -> Dict[str, Any]:
        """在同一真实 holdout 上量化 teacher vs student。

        Parameters
        ----------
        dataset: 真实 holdout dataset (.X/.Y)
        teacher_mse: teacher 已测得的单步 mse (锚点, 外部传入避免重算)

        Returns
        -------
        dict: teacher_mse / student_mse / delta / verdict / ratio
        """
        if not self._fitted:
            raise ValueError("student 须先 fit")
        sm = self.student_mse_on(dataset)
        tm = float(teacher_mse)
        ratio = sm / tm if tm > 0 else float("inf")
        if sm <= tm * (1.0 - self.margin):
            verdict = "improved"
        elif ratio > 10.0:
            verdict = "collapsed"
        else:
            verdict = "degraded"
        return {
            "teacher_mse": tm,
            "student_mse": sm,
            "delta": round(sm - tm, 6),
            "ratio_student_over_teacher": round(ratio, 4),
            "verdict": verdict,
            "honest_note": ("student 在真实 holdout 上未优于 teacher; "
                            "不宣称 RSI 改进, 留候选账本" if verdict != "improved"
                            else "student 在真实 holdout 上略优于 teacher (一代)"),
        }


# ====================================================================== #
# dev5: 数据回流三档配比代理 (预/中/后训练)
# ====================================================================== #
class DataRefluxMixer:
    """数据回流三档配比代理 (dev5): 预/中/后训练的原始:自生成配比。

    **纯配比代理, 不真训练主预测器**。给出三档采样方案, 并按 dev4 的
    teacher→student verdict 诚实标注后训练档的崩塌风险。

    三档 (原始:自生成):
        pre  (预训练): 100:0    纯原始, 奠基
        mid  (中训练): 70:30    原始为主, 少量自生成探索
        post (后训练): 30:70    自生成为主, 但若 student 已 degraded,
                                高比例回流有崩塌风险 -> caution
    """

    STAGES = ("pre", "mid", "post")

    def __init__(self, mid_orig: float = 0.7, post_orig: float = 0.3):
        if not (0.0 < mid_orig < 1.0) or not (0.0 < post_orig < 1.0):
            raise ValueError("配比须在 (0,1)")
        self.mid_orig = float(mid_orig)
        self.post_orig = float(post_orig)

    def mix(self, n_orig: int, n_self: int, stage: str,
            student_verdict: str = "unknown") -> Dict[str, Any]:
        """返回某档的采样配比方案。

        Parameters
        ----------
        n_orig:  原始数据可用样本数
        n_self:   自生成数据可用样本数
        stage:    pre/mid/post
        student_verdict:  dev4 判定 (improved/degraded/collapsed/unknown)
        """
        if stage not in self.STAGES:
            raise ValueError(f"stage 须 ∈ {self.STAGES}, 得 {stage!r}")
        if n_orig < 0 or n_self < 0:
            raise ValueError("样本数须非负")
        if stage == "pre":
            orig_w, self_w = 1.0, 0.0
        elif stage == "mid":
            orig_w, self_w = self.mid_orig, 1.0 - self.mid_orig
        else:
            orig_w, self_w = self.post_orig, 1.0 - self.post_orig

        # 总预算 = 原始可用量 (自生成是补充, 不扩总量)
        total = max(n_orig, 1)
        n_orig_use = int(round(total * orig_w))
        n_self_use = min(int(round(total * self_w)), n_self)

        # 诚实标注: 后训练档 + student 已退化 -> 崩塌风险
        caution = (stage == "post" and student_verdict in
                   ("degraded", "collapsed"))
        return {
            "stage": stage,
            "orig_weight": orig_w,
            "self_weight": self_w,
            "n_orig_use": n_orig_use,
            "n_self_use": n_self_use,
            "student_verdict": student_verdict,
            "caution": caution,
            "caution_note": ("后训练档自生成占比高且 student 已 degraded, "
                             "高比例回流有漂移/崩塌风险, 建议降档或回滚"
                             if caution else ""),
        }


# ====================================================================== #
# dev6: 退化检测与回滚 (teacher->student 真单调改进 vs 漂移/崩塌量化)
# ====================================================================== #
class DegradationDetector:
    """退化检测 (dev6): 量化 teacher→student 代际真改进 vs 漂移/崩塌。

    输入代际 student 在**真实 holdout** 上的 mse 序列, 判定:
        improving : 整体单调下降 (或斜率显著为负)
        drifting  : 波动上升 (斜率非负, 无严格单调)
        collapsed : 某一代 mse > 前一代 * collapse_ratio (突增崩塌)
    纯解析、确定性、零梯度。**不假设 RSI 必然改善**。
    """

    def __init__(self, collapse_ratio: float = 2.0, min_delta: float = 1e-4):
        if collapse_ratio <= 1.0:
            raise ValueError("collapse_ratio 须 >1")
        self.collapse_ratio = float(collapse_ratio)
        self.min_delta = float(min_delta)

    def analyze(self, mse_series: Sequence[float]) -> Dict[str, Any]:
        if len(mse_series) < 2:
            raise ValueError("退化检测至少需 2 代 mse")
        s = [float(x) for x in mse_series]
        # 严格单调下降?
        deltas = [s[i] - s[i - 1] for i in range(1, len(s))]
        monotonic = all(d <= self.min_delta for d in deltas)
        # 线性斜率 (最小二乘)
        n = len(s)
        xs = list(range(n))
        x_mean = sum(xs) / n
        y_mean = sum(s) / n
        num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, s))
        den = sum((x - x_mean) ** 2 for x in xs)
        slope = num / den if den > 0 else 0.0
        # 崩塌点: 某代 mse > 前代 * collapse_ratio
        collapse_gen = None
        for i in range(1, n):
            if s[i] > s[i - 1] * self.collapse_ratio:
                collapse_gen = i
                break

        if collapse_gen is not None:
            verdict = "collapsed"
        elif monotonic and slope < 0:
            verdict = "improving"
        else:
            verdict = "drifting"
        return {
            "n_generations": n,
            "mse_series": s,
            "deltas": [round(d, 6) for d in deltas],
            "linear_slope": round(slope, 6),
            "strictly_monotonic_decreasing": bool(monotonic),
            "collapse_generation": collapse_gen,
            "verdict": verdict,
        }


class RollbackManager:
    """回滚管理 (dev6): 退化时回滚到历史最佳代快照。

    维护代际 student 状态快照 (gen -> (state_dict, mse))。`should_rollback`
    在当代 mse 超过历史最佳 * patience_ratio 时建议回滚; `best` 返回历史最佳代。
    纯治理工具, 不改主预测器权重。
    """

    def __init__(self, patience_ratio: float = 1.1):
        if patience_ratio <= 1.0:
            raise ValueError("patience_ratio 须 >1")
        self.patience_ratio = float(patience_ratio)
        self.snapshots: Dict[int, Dict[str, Any]] = {}

    def register(self, gen: int, state_dict: Dict[str, torch.Tensor],
                 mse: float) -> Dict[str, Any]:
        if gen in self.snapshots:
            raise ValueError(f"代 {gen} 已注册")
        self.snapshots[gen] = {"state_dict": state_dict, "mse": float(mse)}
        return {"gen": gen, "mse": float(mse),
                "is_best": self.best()["gen"] == gen}

    def should_rollback(self, current_mse: float) -> bool:
        if not self.snapshots:
            raise ValueError("无快照可回滚")
        best_mse = self.best()["mse"]
        return float(current_mse) > best_mse * self.patience_ratio

    def best(self) -> Dict[str, Any]:
        if not self.snapshots:
            raise ValueError("无快照")
        gen = min(self.snapshots, key=lambda g: self.snapshots[g]["mse"])
        return {"gen": gen, "mse": self.snapshots[gen]["mse"]}

    def rollback_target(self) -> Dict[str, Any]:
        return self.best()


# ====================================================================== #
# 4.2.1: 收益递减判据 + 自训练 A/B (自产数据 vs 原始数据)
# ====================================================================== #
class DiminishingReturnsCriterion:
    """收益递减判据 (4.2.1): 连续 k 代真实 holdout mse 改善不足 -> 停止自训练。

    纯解析、确定性。避免 RSI 无界递归 (防退化: 收益不稳就停)。
    """

    def __init__(self, patience: int = 3, min_delta: float = 1e-3):
        if patience < 1:
            raise ValueError("patience 须 >=1")
        if min_delta <= 0:
            raise ValueError("min_delta 须 >0")
        self.patience = int(patience)
        self.min_delta = float(min_delta)

    def check(self, mse_series: Sequence[float]) -> Dict[str, Any]:
        if len(mse_series) < 2:
            raise ValueError("至少需 2 代 mse")
        s = [float(x) for x in mse_series]
        recent = s[-(self.patience + 1):]
        improvements = [recent[i - 1] - recent[i]
                        for i in range(1, len(recent))]
        diminishing = all(imp < self.min_delta for imp in improvements)
        return {
            "n_generations": len(s),
            "patience": self.patience,
            "recent_improvements": [round(x, 6) for x in improvements],
            "diminishing_returns": bool(diminishing),
            "recommendation": "stop_self_training" if diminishing
                              else "continue",
        }


class SelfTrainAB:
    """自训练 A/B (4.2.1): 自产数据 vs 原始数据训练外挂 student。

    **诚实 A/B, 不预设哪方赢**。
        A 臂: student 在自生成三元组 (s_t→s_{t+1}) 上训练
        B 臂: student 在原始 dataset (窗口末帧→真实下一状态) 上训练
    两臂 student 架构/优化/种子一致, 仅训练数据不同; 在**同一真实 holdout**
    上报单步 mse。若 A 未优于 B, 照实报 self_generated_not_better,
    **不宣称自产数据有效**。主预测器全程只读, 主参恒 52191。
    """

    def __init__(self, teacher, hidden: int = 32, lr: float = 1e-2,
                 epochs: int = 60, seed: int = 0):
        self.teacher = teacher
        self.teacher.eval()
        self.hidden = int(hidden)
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.seed = int(seed)

    def _fit_student(self, s: torch.Tensor, ns: torch.Tensor,
                     seed: int) -> "_StudentHead":
        st = _StudentHead(hidden=self.hidden)
        g = torch.Generator().manual_seed(seed)
        opt = torch.optim.AdamW(st.parameters(), lr=self.lr,
                               weight_decay=1e-4)
        n = s.size(0)
        # run() 外层 @torch.no_grad; 训练 student 需临时恢复梯度
        with torch.enable_grad():
            for ep in range(self.epochs):
                idx = torch.randperm(n, generator=g)
                for i in range(0, n, 32):
                    b = idx[i:i + 32]
                    opt.zero_grad()
                    loss = ((st(s[b]) - ns[b]) ** 2).mean()
                    loss.backward()
                    opt.step()
        st.eval()
        return st

    @torch.no_grad()
    def _holdout_mse(self, student: "_StudentHead",
                     dataset) -> float:
        s_last = dataset.X[:, -1, :]
        target = dataset.Y[:, 0, :]
        pred = student(s_last)
        return round(float(((pred - target) ** 2).mean()), 6)

    @torch.no_grad()
    def run(self, self_trip: SelfGeneratedTriplets,
            orig_dataset, holdout_dataset) -> Dict[str, Any]:
        """跑 A/B。

        Parameters
        ----------
        self_trip:       A 臂训练数据 (自生成三元组)
        orig_dataset:    B 臂训练数据 (真实 ParametricDynamicsDataset)
        holdout_dataset: 两臂共用的真实 holdout (.X/.Y)
        """
        if not isinstance(self_trip, SelfGeneratedTriplets):
            raise ValueError("self_trip 须为 SelfGeneratedTriplets")
        if len(self_trip) == 0:
            raise ValueError("self_trip 为空")

        # A 臂: 自生成
        st_a = self._fit_student(self_trip.states, self_trip.next_states,
                                 seed=self.seed)
        mse_a = self._holdout_mse(st_a, holdout_dataset)

        # B 臂: 原始数据 (窗口末帧 -> 真实下一状态)
        s_orig = orig_dataset.X[:, -1, :]
        ns_orig = orig_dataset.Y[:, 0, :]
        st_b = self._fit_student(s_orig, ns_orig, seed=self.seed)
        mse_b = self._holdout_mse(st_b, holdout_dataset)

        a_better = mse_a < mse_b
        return {
            "arm_A_self_generated_mse": mse_a,
            "arm_B_original_mse": mse_b,
            "delta_A_minus_B": round(mse_a - mse_b, 6),
            "self_generated_wins": bool(a_better),
            "verdict": ("自产数据更优" if a_better else
                        "自产数据未优于原始数据 (留候选账本, 不宣称有效)"),
            "student_n_params": st_a.n_params,
        }


