"""
潜在空间前向世界模型 (v3.6.0) —— PWM 物理世界模型 (阶段三·物理)
=================================================================
analogy, not reproduction —— 受 latent dynamics / world-model 思想启发的**合成低维
类比实现**, 不宣称复现 V-JEPA / Cosmos / Genie / 任何视频世界模型; 潜在空间为
predictor 中间激活的低维合成代理 (d_input=32 维), 非高维视觉潜空间。

设计纪律 (与全工程 opt-in / 零外挂优先一致):
    * 纯前向、确定性; 推理路径整体 @torch.no_grad, 只读 predictor, 不改主权重。
    * 潜在状态 z_t = predictor.obs_encoder(window)[:, -1, :] (窗口末帧嵌入),
      形状 [B, d_input]。这是对"中间激活即状态"的极简类比。
    * 前向转移 z_{t+1} = z_t + MLP([z_t, action]) 为**外挂小 MLP**:
      不入主 predictor.state_dict, 不随正式 checkpoint 主权重保存; 末层零初始化
      => 未拟合时转移恒等 (潜在空间零漂移, 数值稳定)。参数量显式可查 n_params。
    * 物理读出 decode(z): 外挂 Linear(latent -> raw), 同样不入主 state_dict。
    * imagine(window, H): 第 0 步**逐位锚定** predictor.predict_next (故 H=1 时
      与 predictor.rollout(window,1) 逐位等价); H>1 步在潜在空间用外挂转移推进,
      再经外挂读出解码回物理状态 [pos(3), vel(3)]。
    * fit(): 离线在冻结 predictor 的潜在轨迹上以极小 AdamW 拟合外挂转移+读出
      (优化器只含世界模型自身参数, predictor 权重梯度恒为 0)。
"""

from __future__ import annotations

import logging

logger = logging.getLogger("udos.world_model")


from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from .dynamics import RAW_DIM


class LatentTransitionMLP(nn.Module):
    """外挂潜在转移 MLP (不入主 state_dict)。

    z_{t+1} = z_t + MLP([z_t, action])  (残差形式)
    末层权重/偏置零初始化 => 未拟合时残差=0, 转移严格恒等, 潜在空间不漂移。
    """

    def __init__(self, latent_dim: int, action_dim: int = 0, hidden: int = 32):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = int(action_dim)
        in_dim = latent_dim + max(self.action_dim, 0)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, latent_dim),
        )
        # 零初始化末层 => 初始残差为 0 (恒等转移), 对齐 LoRA scaler_B=0 惯例
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, z: torch.Tensor,
                action: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.action_dim > 0:
            if action is None:
                raise ValueError(
                    "LatentTransitionMLP 配置了 action_dim>0, 需提供 action")
            x = torch.cat([z, action.to(z.dtype)], dim=-1)
        else:
            x = z
        return z + self.net(x)


class LatentDecoder(nn.Module):
    """外挂潜在 -> 物理状态线性读出 (不入主 state_dict)。"""

    def __init__(self, latent_dim: int, raw_dim: int = RAW_DIM):
        super().__init__()
        self.out = nn.Linear(latent_dim, raw_dim)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.out(z)


class LatentWorldModel:
    """潜在空间前向世界模型 (只读外挂, 持有 predictor 引用, 不注册其参数)。

    Parameters
    ----------
    predictor:
        已训练 PhysicsPredictor (只读)。潜在宽度取 predictor.ctm.cfg.d_input。
    action_dim:
        可选条件动作维度 (>0 时转移以 [z, action] 为输入; 0 为无动作自治转移)。
    hidden:
        外挂转移 MLP 隐藏宽。
    """

    def __init__(self, predictor, action_dim: int = 0, hidden: int = 32):
        self.predictor = predictor
        self.latent_dim = int(predictor.ctm.cfg.d_input)
        self.raw_dim = int(predictor.raw_dim)
        self.action_dim = int(action_dim)
        self.transit = LatentTransitionMLP(self.latent_dim, action_dim, hidden)
        self.decode = LatentDecoder(self.latent_dim, self.raw_dim)
        self.transit.eval()
        self.decode.eval()
        self._fitted = False

    # ------------------------------------------------------------------ #
    # 元信息
    # ------------------------------------------------------------------ #
    @property
    def n_params(self) -> int:
        """外挂世界模型自身可学参数量 (不含主 predictor 的 52191)。"""
        return (self.transit.n_params + self.decode.n_params)

    @property
    def fitted(self) -> bool:
        return self._fitted

    def describe(self) -> Dict[str, Any]:
        return {
            "latent_dim": self.latent_dim,
            "raw_dim": self.raw_dim,
            "action_dim": self.action_dim,
            "wm_params": self.n_params,
            "transit_params": self.transit.n_params,
            "decode_params": self.decode.n_params,
            "fitted": self._fitted,
            "main_predictor_params_untouched": True,
            "analogy_not_reproduction": True,
        }

    # ------------------------------------------------------------------ #
    # 输入守卫
    # ------------------------------------------------------------------ #
    @staticmethod
    def _as_window(window: torch.Tensor) -> torch.Tensor:
        w = torch.as_tensor(window, dtype=torch.float32)
        if w.dim() == 2:
            w = w.unsqueeze(0)
        if w.dim() != 3:
            raise ValueError("window 需为 [W,RAW] 或 [B,W,RAW]")
        if w.size(1) == 0:
            raise ValueError("window 时间维为空")
        return w

    # ------------------------------------------------------------------ #
    # 编码 / 转移 / 解码 (均零梯度)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def encode_latent(self, window: torch.Tensor,
                      scene_params: Optional[torch.Tensor] = None) -> torch.Tensor:
        """窗口 -> 潜在状态 z_t [B, latent_dim] (取 obs_encoder 末帧嵌入)。"""
        w = self._as_window(window)
        h = self.predictor.obs_encoder(w)          # [B, W, d_input]
        return h[:, -1, :]                          # [B, latent_dim]

    @torch.no_grad()
    def transit_step(self, z: torch.Tensor,
                     action: Optional[torch.Tensor] = None) -> torch.Tensor:
        """潜在空间单步前向转移 z_{t+1}=f(z_t, action) (外挂 MLP)。"""
        return self.transit(z, action)

    @torch.no_grad()
    def decode_latent(self, z: torch.Tensor) -> torch.Tensor:
        """潜在状态 -> 物理状态 [B, RAW_DIM] (外挂线性读出)。"""
        return self.decode(z)

    # ------------------------------------------------------------------ #
    # 多步"想象" rollout
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def imagine(self, window: torch.Tensor, horizon: int,
                scene_params: Optional[torch.Tensor] = None,
                actions: Optional[Sequence] = None) -> torch.Tensor:
        """多步潜在 rollout 并解码回物理状态。

        第 0 步逐位锚定 predictor.predict_next (H=1 时 == predictor.rollout)。
        第 h>=1 步: 滑窗 -> encode_latent -> transit_step(外挂) -> decode_latent。
        horizon 须为正整数; 想象产生 NaN/inf 时显式 ValueError (不静默污染)。
        返回 [B, horizon, RAW_DIM]。
        """
        w = self._as_window(window)
        if not isinstance(horizon, int) or horizon <= 0:
            raise ValueError("horizon 需为正整数")
        if horizon == 0:
            raise ValueError("horizon 需为正整数 (>=1)")

        outs: List[torch.Tensor] = []
        cur = w
        # 第 0 步锚定真实 predictor 单步 (H=1 逐位等价 rollout)
        nxt = self.predictor.predict_next(cur, scene_params=scene_params)
        outs.append(nxt)
        for h in range(1, horizon):
            cur = torch.cat([cur[:, 1:, :], nxt.unsqueeze(1)], dim=1)
            z = self.encode_latent(cur, scene_params=scene_params)
            act = None
            if actions is not None:
                act = torch.as_tensor(actions[h - 1], dtype=torch.float32)
                if act.dim() == 1:
                    act = act.unsqueeze(0)
            z = self.transit_step(z, act)
            nxt = self.decode_latent(z)
            if not bool(torch.isfinite(nxt).all()):
                raise ValueError(
                    f"想象 rollout 第 {h} 步产生非有限状态 (NaN/inf), 拒绝污染输出")
            outs.append(nxt)
        return torch.stack(outs, dim=1)            # [B, H, RAW]

    @torch.no_grad()
    def imagine_rollout(self, window: torch.Tensor, horizon: int,
                        scene_params: Optional[torch.Tensor] = None,
                        actions: Optional[Sequence] = None,
                        compare_real: bool = False) -> Dict[str, torch.Tensor]:
        """H 步潜在 rollout (v3.6.0.dev1): 同时产出**潜在轨迹**与**解码物理状态**。

        返回 dict:
            states  [B, H, RAW]    解码回的物理状态 (第 0 步锚定 predict_next)
            latents [B, H, latent] 每步潜在状态 z_t (第 0 步=初始窗口编码)
            real    [B, H, RAW]    compare_real=True 时: predictor.rollout 真实轨迹
            step_mse [H]           compare_real=True 时: 逐步 (states-real)^2 均值
        H=1 时 states[0] == predictor.rollout(window,1) 逐位等价。
        """
        w = self._as_window(window)
        if not isinstance(horizon, int) or horizon <= 0:
            raise ValueError("horizon 需为正整数")

        states: List[torch.Tensor] = []
        latents: List[torch.Tensor] = []
        cur = w
        nxt = self.predictor.predict_next(cur, scene_params=scene_params)
        latents.append(self.encode_latent(cur, scene_params=scene_params))
        states.append(nxt)
        for h in range(1, horizon):
            cur = torch.cat([cur[:, 1:, :], nxt.unsqueeze(1)], dim=1)
            z = self.encode_latent(cur, scene_params=scene_params)
            act = None
            if actions is not None:
                act = torch.as_tensor(actions[h - 1], dtype=torch.float32)
                if act.dim() == 1:
                    act = act.unsqueeze(0)
            z = self.transit_step(z, act)
            nxt = self.decode_latent(z)
            if not bool(torch.isfinite(nxt).all()):
                raise ValueError(
                    f"imagine_rollout 第 {h} 步产生非有限状态 (NaN/inf)")
            latents.append(z)
            states.append(nxt)

        out: Dict[str, torch.Tensor] = {
            "states": torch.stack(states, dim=1),
            "latents": torch.stack(latents, dim=1),
        }
        if compare_real:
            real = self.predictor.rollout(w, horizon, scene_params=scene_params)
            out["real"] = real
            out["step_mse"] = ((out["states"] - real) ** 2).mean(dim=(0, 2))
        return out

    # ------------------------------------------------------------------ #
    # v3.6.0.dev5: 集成式不确定性 / 置信度 / 高不确定回退
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _imagine_noisy(self, w: torch.Tensor, horizon: int,
                       scene_params: Optional[torch.Tensor],
                       g: torch.Generator, noise_scale: float) -> torch.Tensor:
        """单次带噪想象 (潜在注入 seed 控制高斯噪声), 第 0 步锚定真实 predict_next。"""
        outs = []
        cur = w
        nxt = self.predictor.predict_next(cur, scene_params=scene_params)
        outs.append(nxt)
        for h in range(1, horizon):
            cur = torch.cat([cur[:, 1:, :], nxt.unsqueeze(1)], dim=1)
            z = self.encode_latent(cur, scene_params=scene_params)
            z = self.transit_step(z)
            if noise_scale > 0.0:
                z = z + noise_scale * torch.randn(
                    z.shape, generator=g, dtype=z.dtype)
            nxt = self.decode_latent(z)
            outs.append(nxt)
        return torch.stack(outs, dim=1)                # [B,H,raw]

    @torch.no_grad()
    def imagine_uncertain(self, window: torch.Tensor, horizon: int,
                           n_samples: int = 5, noise_scale: float = 0.05,
                           fallback_threshold: Optional[float] = None,
                           scene_params: Optional[torch.Tensor] = None,
                           seed: int = 2026) -> Dict[str, Any]:
        """集成式不确定性 (v3.6.0.dev5): n_samples 次带噪想象的方差作不确定代理。

        复用 ensemble 思想但为合成单模型 + 种子受控扰动 (确定性)。返回:
            mean [B,H,raw]        n_samples 次想象的逐点均值;
            std  [B,H,raw]        逐点标准差 (不确定性代理);
            uncertainty [B,H]     每步均方不确定 (跨 raw 维);
            confidence [B,H]      exp(-uncertainty / median) ∈[0,1];
            used_fallback [H]     哪些步因不确定超阈值而回退主 predictor rollout;
            states [B,H,raw]      最终输出 (回退步用真实 rollout 值替换)。
        fallback_threshold=None 时不回退 (states==mean)。
        """
        w = self._as_window(window)
        if not isinstance(horizon, int) or horizon <= 0:
            raise ValueError("horizon 需为正整数")
        if not isinstance(n_samples, int) or n_samples < 1:
            raise ValueError("n_samples 需为 >=1 的整数")
        if noise_scale < 0:
            raise ValueError("noise_scale 须 >=0")

        g = torch.Generator().manual_seed(seed)
        trajs = torch.stack([self._imagine_noisy(w, horizon, scene_params, g,
                                                 noise_scale)
                             for _ in range(n_samples)], dim=0)   # [K,B,H,raw]
        mean = trajs.mean(dim=0)                                  # [B,H,raw]
        std = trajs.std(dim=0, unbiased=False)                    # [B,H,raw]
        uncertainty = std.mean(dim=-1)                           # [B,H]
        med = float(uncertainty.median()) + 1e-9
        confidence = torch.exp(-uncertainty / med).clamp(0.0, 1.0)

        states = mean.clone()
        used_fallback: List[int] = []
        if fallback_threshold is not None:
            real = self.predictor.rollout(w, horizon, scene_params=scene_params)
            for h in range(horizon):
                if float(uncertainty[:, h].mean()) > fallback_threshold:
                    states[:, h, :] = real[:, h, :]
                    used_fallback.append(h)
        return {
            "mean": mean, "std": std,
            "uncertainty": uncertainty, "confidence": confidence,
            "states": states, "used_fallback": used_fallback,
            "n_samples": n_samples, "noise_scale": noise_scale,
        }

    # ------------------------------------------------------------------ #
    # 离线拟合外挂转移 + 读出 (优化器只含世界模型自身参数; predictor 零梯度)
    # ------------------------------------------------------------------ #
    def fit(self, dataset, epochs: int = 30, lr: float = 1e-2,
            batch_size: int = 64, seed: int = 0) -> Dict[str, float]:
        """在冻结 predictor 的潜在轨迹上离线拟合外挂世界模型。

        教师对 (从 dataset.X / dataset.P 逐样本取, 教师强制):
            z_t   = encode_latent(window)
            s1    = predictor.predict_next(window)        # 真实下一物理状态
            z1    = encode_latent(slid_window)            # 真实下一潜在状态
        监督: decode(z_t) -> s1 ; transit(z_t) -> z1 。
        优化器只含 self.transit/self.decode 参数; predictor 全程 eval + 不反传。
        返回拟合历史 (初/末 loss)。
        """
        g = torch.Generator().manual_seed(seed)
        X = dataset.X
        P = getattr(dataset, "P", None)
        self.predictor.eval()
        with torch.no_grad():
            zs, s1s, z1s = [], [], []
            for i in range(X.size(0)):
                w = X[i:i + 1]
                sp = P[i:i + 1] if P is not None else None
                z0 = self.encode_latent(w, scene_params=sp)
                s1 = self.predictor.predict_next(w, scene_params=sp)
                w1 = torch.cat([w[:, 1:, :], s1.unsqueeze(1)], dim=1)
                z1 = self.encode_latent(w1, scene_params=sp)
                zs.append(z0)
                s1s.append(s1)
                z1s.append(z1)
            Z0 = torch.cat(zs, 0)      # [N, latent]
            S1 = torch.cat(s1s, 0)      # [N, raw]
            Z1 = torch.cat(z1s, 0)      # [N, latent]

        params = list(self.transit.parameters()) + list(self.decode.parameters())
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
        n = Z0.size(0)
        first_loss = last_loss = float("nan")
        for ep in range(epochs):
            idx = torch.randperm(n, generator=g)
            ep_loss = 0.0
            nb = 0
            for s in range(0, n, batch_size):
                b = idx[s:s + batch_size]
                zb, sb, zb1 = Z0[b], S1[b], Z1[b]
                opt.zero_grad()
                pred_s = self.decode(zb)
                pred_z = self.transit(zb)
                loss = ((pred_s - sb) ** 2).mean() \
                    + ((pred_z - zb1) ** 2).mean()
                loss.backward()
                opt.step()
                ep_loss += float(loss.detach())
                nb += 1
            ep_loss /= max(nb, 1)
            if ep == 0:
                first_loss = ep_loss
            last_loss = ep_loss
        self.transit.eval()
        self.decode.eval()
        self._fitted = True
        return {"fit_first_loss": round(first_loss, 6),
                "fit_last_loss": round(last_loss, 6),
                "fit_epochs": epochs, "n_train": int(n),
                "predictor_grad_zero": True,
                "wm_params": self.n_params}
