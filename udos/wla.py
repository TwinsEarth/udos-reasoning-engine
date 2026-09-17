"""
UnifoLM-WLA 机制类比核心 (v3.9.0.dev1..dev6)
=========================================================================
analogy, not reproduction —— 类比宇树 WLA 的"光流动态区域 / VQ-VAE / 三路 RVQ /
ER-Flow 对齐 / MMDiT flow 动作专家", **非复现 6B**: 全部 CPU 合成状态/动作代理,
外挂、零梯度、opt-in, 不改主 predictor 52191 参数。

类清单 (严格对应 VERSION_PLAN_3.9):
    * ``ChangeMask``            dev1: 相邻状态差分 -> 稀疏 change-mask (光流等价物)
    * ``ChangeMaskVQ``          dev2: 变化区小型 VQ 码本 (利用率/重构误差/坍塌)
    * ``RVQActionTokenizer``    dev4: 每分组残差矢量量化动作分词
    * ``ActionStateTaskAlign``  dev5: 动作 token-状态-任务 对齐 (轻量对齐损失+一致性)
    * ``FlowMatchingDecoder``   dev6: 冻结骨干 + 外挂少步 flow-matching 动作解码器
                                   (对标 MMDiT flow, 与直接回归 A/B; 非复现 6B)

dev3 (稀疏 vs 3.6 PWM 稠密 rollout A/B) 由脚本/测试落 JSON, 不在此。
设计纪律: 纯前向、确定性、@torch.no_grad、空/非法显式 ValueError、日志走 stderr。
"""

from __future__ import annotations

import logging

logger = logging.getLogger("udos.wla")

from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# dev1: 相邻状态差分 -> 稀疏 change-mask (光流的状态域等价物)
# --------------------------------------------------------------------------- #
class ChangeMask:
    """相邻状态差分提取"真实变化分量"稀疏掩码 (interaction-centric 类比)。

    主张 (类比 WLA 层2): 不重构整帧/整态未来, 只盯"动作会改变的分量"。
    在状态域等价物上: |s_{t+1}-s_t| 超过阈值的维 -> change=1, 其余不变(复制上一帧)。

    Parameters
    ----------
    threshold:
        差分绝对值阈值; 超过即视为"动态/变化"分量 (>0)。
    """

    def __init__(self, threshold: float = 0.05) -> None:
        if not (threshold > 0) or threshold != threshold:
            raise ValueError("threshold 须为有限正数 (>0)")
        self.threshold = float(threshold)

    @staticmethod
    def _as_window(w: torch.Tensor) -> torch.Tensor:
        t = torch.as_tensor(w, dtype=torch.float32)
        if t.dim() == 2:
            t = t.unsqueeze(0)
        if t.dim() != 3:
            raise ValueError("window 需为 [W,RAW] 或 [B,W,RAW]")
        if t.size(1) < 2:
            raise ValueError("window 至少需 2 帧以计算相邻差分")
        return t

    @torch.no_grad()
    def diff(self, window: torch.Tensor) -> torch.Tensor:
        """窗口末两帧差分 [B, RAW] (s_last - s_prev)。"""
        w = self._as_window(window)
        return w[:, -1, :] - w[:, -2, :]

    @torch.no_grad()
    def mask(self, window: torch.Tensor) -> torch.Tensor:
        """二值变化掩码 [B, RAW]: |diff| > threshold 记 1 (float)。"""
        d = self.diff(window)
        return (d.abs() > self.threshold).float()

    @torch.no_grad()
    def sparsity(self, window: torch.Tensor) -> Dict[str, float]:
        """报告变化分量占比 (sparsity = 变化维 / 总维)。"""
        m = self.mask(window)
        return {
            "changed_ratio": float(m.mean()),
            "n_changed": int(m.sum()),
            "raw_dim": int(m.size(-1)),
        }

    @torch.no_grad()
    def sparse_next(self, window: torch.Tensor,
                    predicted_next: Optional[torch.Tensor] = None) -> torch.Tensor:
        """稀疏下一状态: 变化维用 predicted_next, 未变化维复制末帧 (恒等)。

        predicted_next=None 时全维恒等复制末帧 (退化为"无变化"先验)。
        """
        w = self._as_window(window)
        last = w[:, -1, :]
        if predicted_next is None:
            return last.clone()
        p = torch.as_tensor(predicted_next, dtype=torch.float32)
        if p.dim() == 1:
            p = p.unsqueeze(0)
        if p.shape != last.shape:
            raise ValueError(
                f"predicted_next 形状 {tuple(p.shape)} != 末帧 {tuple(last.shape)}")
        m = self.mask(w)
        return m * p + (1.0 - m) * last


# --------------------------------------------------------------------------- #
# dev2: 变化区小型 VQ 码本 (利用率/重构误差/坍塌检测)
# --------------------------------------------------------------------------- #
def _kmeans_fit(x: torch.Tensor, k: int, steps: int,
                seed: int) -> torch.Tensor:
    """纯 torch 确定性 k-means, 返回 [k, dim] 码本中心。"""
    n = x.size(0)
    if k > n:
        raise ValueError(f"码本 k={k} > 样本数 {n}")
    g = torch.Generator().manual_seed(seed)
    # k-means++ 初值
    first = int(torch.randint(0, n, (1,), generator=g).item())
    centers = [x[first]]
    d2 = ((x - x[first]) ** 2).sum(dim=1)
    for _ in range(1, k):
        probs = d2 / d2.sum().clamp_min(1e-12)
        j = int(torch.multinomial(probs, 1, generator=g).item())
        centers.append(x[j])
        d2 = torch.minimum(d2, ((x - x[j]) ** 2).sum(dim=1))
    C = torch.stack(centers, 0)
    for _ in range(steps):
        dist = ((x[:, None, :] - C[None, :, :]) ** 2).sum(dim=2)  # [n,k]
        assign = dist.argmin(dim=1)
        for c in range(k):
            sel = x[assign == c]
            if sel.numel() > 0:
                C[c] = sel.mean(dim=0)
    return C


class ChangeMaskVQ:
    """对变化向量做小型 VQ 离散化 (外挂码本, 不入主 state_dict)。

    report: 码本利用率 (被用到的码占比) / 重构 MSE / 坍塌 (最大码使用占比, 接近 1 即坍塌)。
    """

    def __init__(self, codebook_size: int = 8, seed: int = 0) -> None:
        if codebook_size < 2:
            raise ValueError("codebook_size 须 >=2")
        self.k = int(codebook_size)
        self.seed = int(seed)
        self.centers: Optional[torch.Tensor] = None
        self.dim: Optional[int] = None

    @torch.no_grad()
    def fit(self, change_vectors: torch.Tensor, steps: int = 30) -> Dict[str, Any]:
        x = torch.as_tensor(change_vectors, dtype=torch.float32)
        if x.dim() != 2 or x.size(0) < self.k:
            raise ValueError(
                f"change_vectors 需 [N,dim] 且 N>=k({self.k}), 收到 {tuple(x.shape)}")
        self.dim = x.size(1)
        self.centers = _kmeans_fit(x, self.k, steps, self.seed)
        return self.report(x)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        self._fitted_guard()
        xt = torch.as_tensor(x, dtype=torch.float32)
        if xt.size(-1) != self.dim:
            raise ValueError(f"维度 {xt.size(-1)} != 码本 {self.dim}")
        dist = ((xt[..., None, :] - self.centers) ** 2).sum(dim=-1)
        return dist.argmin(dim=-1)

    @torch.no_grad()
    def decode(self, ids: torch.Tensor) -> torch.Tensor:
        self._fitted_guard()
        return self.centers[ids.long()]

    def _fitted_guard(self) -> None:
        if self.centers is None:
            raise ValueError("码本未 fit, 先调用 fit()")

    @torch.no_grad()
    def report(self, x: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        self._fitted_guard()
        ids = self.encode(x) if x is not None else torch.arange(self.k)
        onehot = torch.nn.functional.one_hot(ids.long(), self.k).float()
        usage = onehot.mean(dim=0)
        active = int((usage > 0).sum())
        util = active / self.k
        max_share = float(usage.max())
        recon = None
        if x is not None:
            recon = float(((self.decode(ids) - x) ** 2).mean())
        collapsed = bool(max_share > 0.9)
        return {
            "codebook_size": self.k, "dim": self.dim,
            "utilization": round(util, 4), "active_codes": active,
            "recon_mse": recon,
            "max_code_share": round(max_share, 4),
            "collapsed": collapsed,
        }


# --------------------------------------------------------------------------- #
# dev4: 每分组残差矢量量化 (RVQ) 动作分词
# --------------------------------------------------------------------------- #
class RVQActionTokenizer:
    """单分组残差矢量量化: L 级码本逐级拟合残差 (外挂, 不入主 state_dict)。

    类比 WLA 三分量各自一个 RVQ。report: 每级利用率 / 整体重构 MSE / 坍塌检测。
    """

    def __init__(self, codebook_size: int = 8, n_levels: int = 2,
                 seed: int = 0) -> None:
        if codebook_size < 2 or n_levels < 1:
            raise ValueError("codebook_size>=2 且 n_levels>=1")
        self.k = int(codebook_size)
        self.L = int(n_levels)
        self.seed = int(seed)
        self.codebooks: List[torch.Tensor] = []
        self.dim: Optional[int] = None

    @torch.no_grad()
    def fit(self, actions: torch.Tensor, steps: int = 30) -> Dict[str, Any]:
        x = torch.as_tensor(actions, dtype=torch.float32)
        if x.dim() != 2 or x.size(0) < self.k:
            raise ValueError(
                f"actions 需 [N,dim] 且 N>=k({self.k}), 收到 {tuple(x.shape)}")
        self.dim = x.size(1)
        self.codebooks = []
        residual = x.clone()
        level_reports = []
        for lv in range(self.L):
            C = _kmeans_fit(residual, self.k, steps, self.seed + lv * 101)
            self.codebooks.append(C)
            ids = ((residual[:, None, :] - C[None, :, :]) ** 2).sum(dim=2).argmin(dim=1)
            quant = C[ids]
            residual = residual - quant
            usage = torch.nn.functional.one_hot(ids, self.k).float().mean(dim=0)
            level_reports.append({
                "level": lv,
                "utilization": round(float((usage > 0).sum()) / self.k, 4),
                "max_code_share": round(float(usage.max()), 4),
            })
        recon = self.decode(self.encode(x))
        return {
            "dim": self.dim, "n_levels": self.L, "codebook_size": self.k,
            "recon_mse": float(((recon - x) ** 2).mean()),
            "levels": level_reports,
            "collapsed": any(l["max_code_share"] > 0.9 for l in level_reports),
        }

    def _fitted_guard(self) -> None:
        if not self.codebooks:
            raise ValueError("RVQ 未 fit, 先调用 fit()")

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """[N,dim] -> [N,L] token ids。"""
        self._fitted_guard()
        xt = torch.as_tensor(x, dtype=torch.float32)
        if xt.size(-1) != self.dim:
            raise ValueError(f"维度 {xt.size(-1)} != 码本 {self.dim}")
        residual = xt.clone()
        ids_all = []
        for C in self.codebooks:
            ids = ((residual[:, None, :] - C[None, :, :]) ** 2).sum(dim=2).argmin(dim=1)
            ids_all.append(ids)
            residual = residual - C[ids]
        return torch.stack(ids_all, dim=-1)

    @torch.no_grad()
    def decode(self, ids: torch.Tensor) -> torch.Tensor:
        """[N,L] token ids -> [N,dim] 重构动作。"""
        self._fitted_guard()
        ids = torch.as_tensor(ids).long()
        out = torch.zeros(*ids.shape[:-1], self.dim)
        for lv, C in enumerate(self.codebooks):
            out = out + C[ids[..., lv]]
        return out


# --------------------------------------------------------------------------- #
# dev5: 动作 token - 状态 - 任务 对齐 (轻量对齐损失 + 一致性测试)
# --------------------------------------------------------------------------- #
class ActionStateTaskAlign:
    """把离散动作 token / 状态 latent / 任务 id 投影到同一共享小空间并测对齐。

    analogy: ER-Flow 在单 VLM 内对齐视觉/语言/动作。这里用**确定性**投影 (无新可训
    权重) 把三者映射到统一维, 报告:
        * alignment_loss  同批次 (动作 token 嵌入 与 状态 latent 投影) 的余弦距离均值;
        * consistency     同任务样本对的对齐内积 > 异任务对 (可复算一致性)。
    """

    def __init__(self, latent_dim: int, action_dim: int, n_tasks: int,
                 share_dim: int = 8) -> None:
        if min(latent_dim, action_dim, n_tasks, share_dim) < 1:
            raise ValueError("各维须 >=1")
        self.latent_dim = int(latent_dim)
        self.action_dim = int(action_dim)
        self.n_tasks = int(n_tasks)
        self.share_dim = int(share_dim)
        # 确定性固定投影矩阵 (种子化, 无可训参数; 仅作对齐坐标系)
        g = torch.Generator().manual_seed(7)
        self.W_state = torch.randn(latent_dim, share_dim, generator=g) / (latent_dim ** 0.5)
        self.W_act = torch.randn(action_dim, share_dim, generator=g) / (action_dim ** 0.5)
        self.W_task = torch.eye(n_tasks, share_dim) if n_tasks == share_dim else \
            torch.randn(n_tasks, share_dim, generator=g) / (n_tasks ** 0.5)

    @torch.no_grad()
    def embed_state(self, z: torch.Tensor) -> torch.Tensor:
        z = torch.as_tensor(z, dtype=torch.float32)
        if z.size(-1) != self.latent_dim:
            raise ValueError(f"状态维 {z.size(-1)} != {self.latent_dim}")
        return torch.nn.functional.normalize(z @ self.W_state, dim=-1)

    @torch.no_grad()
    def embed_action(self, a: torch.Tensor) -> torch.Tensor:
        a = torch.as_tensor(a, dtype=torch.float32)
        if a.size(-1) != self.action_dim:
            raise ValueError(f"动作维 {a.size(-1)} != {self.action_dim}")
        return torch.nn.functional.normalize(a @ self.W_act, dim=-1)

    @torch.no_grad()
    def embed_task(self, task_id: torch.Tensor) -> torch.Tensor:
        t = torch.as_tensor(task_id, dtype=torch.long)
        if bool((t < 0).any()) or bool((t >= self.n_tasks).any()):
            raise ValueError("task_id 越界")
        return torch.nn.functional.normalize(self.W_task[t], dim=-1)

    @torch.no_grad()
    def alignment_loss(self, z: torch.Tensor, a: torch.Tensor) -> float:
        """同批次状态-动作对齐损失 = 1 - 余弦相似度均值。"""
        es, ea = self.embed_state(z), self.embed_action(a)
        return float((1.0 - (es * ea).sum(dim=-1)).mean())

    @torch.no_grad()
    def consistency(self, z: torch.Tensor, a: torch.Tensor,
                    task_id: torch.Tensor) -> Dict[str, Any]:
        """一致性测试: 同任务对的状态-动作对齐分 > 异任务对 (可复算 bool)。"""
        es, ea = self.embed_state(z), self.embed_action(a)
        # 成对相似度矩阵 S[N,N]: S[i,j] = 样本 i 的状态嵌入 与 样本 j 的动作嵌入
        S = es @ ea.t()                              # [N,N]
        t = torch.as_tensor(task_id)
        same_mask = (t[:, None] == t[None, :])
        diff_mask = ~same_mask
        # 排除对角 (自身) 避免偏置
        eye = torch.eye(S.size(0), dtype=torch.bool)
        same_mask = same_mask & ~eye
        diff_mask = diff_mask & ~eye
        same = S[same_mask]
        diff = S[diff_mask]
        same_m = float(same.mean()) if same.numel() else 0.0
        diff_m = float(diff.mean()) if diff.numel() else 0.0
        return {
            "same_task_alignment": round(same_m, 4),
            "cross_task_alignment": round(diff_m, 4),
            "consistent": bool(same_m >= diff_m),
        }


# --------------------------------------------------------------------------- #
# dev6: 冻结骨干 + 外挂少步 flow-matching 动作解码器 (对标 MMDiT flow)
# --------------------------------------------------------------------------- #
class FlowMatchingDecoder(nn.Module):
    """外挂少步 flow-matching 动作解码器 (MMDiT flow decoder 的极简类比)。

    机制类比 (非复现 6B MMDiT): 在冻结表征上, 用 n_steps 个 Euler 步沿预测速度场
    把"带噪动作"去噪为连续动作。速度场为外挂小 MLP [z, a_t, t] -> Δa。
    **主 predictor 零梯度、不进主 state_dict**; 与"直接线性回归动作"做 A/B。
    """

    def __init__(self, latent_dim: int, action_dim: int, hidden: int = 16,
                 n_steps: int = 3) -> None:
        super().__init__()
        if min(latent_dim, action_dim, hidden) < 1 or n_steps < 1:
            raise ValueError("各维/步数须 >=1")
        self.latent_dim = int(latent_dim)
        self.action_dim = int(action_dim)
        self.n_steps = int(n_steps)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim + 1, hidden), nn.GELU(),
            nn.Linear(hidden, action_dim),
        )
        # 直接回归对照头 (同参数量级, 用于 A/B)
        self.regress = nn.Linear(latent_dim, action_dim)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _velocity(self, z, a_t, t):
        if torch.is_tensor(t):
            tt = t.to(a_t.dtype)
        else:
            tt = torch.full((a_t.size(0), 1), float(t))
        return self.net(torch.cat([z, a_t, tt], dim=-1))

    @torch.no_grad()
    def decode(self, z: torch.Tensor, seed: int = 0) -> torch.Tensor:
        """从 z 经 n_steps Euler 去噪得连续动作 [B, action_dim]。"""
        g = torch.Generator().manual_seed(seed)
        a_t = torch.randn(z.size(0), self.action_dim, generator=g)
        for k in range(self.n_steps):
            t = k / self.n_steps
            v = self._velocity(z, a_t, t)
            a_t = a_t + v / self.n_steps
        return a_t

    @torch.no_grad()
    def decode_regress(self, z: torch.Tensor) -> torch.Tensor:
        """直接回归对照: z -> 动作 (无去噪过程)。"""
        return self.regress(z)

    def fit(self, z: torch.Tensor, a: torch.Tensor, epochs: int = 30,
            lr: float = 1e-2, seed: int = 0) -> Dict[str, float]:
        """在冻结 z/a 对上离线拟合速度场 + 回归头 (优化器只含本解码器参数)。"""
        g = torch.Generator().manual_seed(seed)
        z = torch.as_tensor(z, dtype=torch.float32)
        a = torch.as_tensor(a, dtype=torch.float32)
        opt = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-4)
        n = z.size(0)
        first = last = float("nan")
        for ep in range(epochs):
            idx = torch.randperm(n, generator=g)
            eloss = 0.0
            nb = 0
            for s in range(0, n, 64):
                b = idx[s:s + 64]
                zb, ab = z[b], a[b]
                t = torch.rand(zb.size(0), 1, generator=g)
                noise = torch.randn(zb.size(0), self.action_dim, generator=g)
                a_t = (1 - t) * noise + t * ab
                v_target = ab - noise
                v_pred = self._velocity(zb, a_t, t)
                loss = ((v_pred - v_target) ** 2).mean() \
                    + ((self.regress(zb) - ab) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                eloss += float(loss.detach())
                nb += 1
            eloss /= max(nb, 1)
            if ep == 0:
                first = eloss
            last = eloss
        self.eval()
        return {"fit_first_loss": round(first, 6), "fit_last_loss": round(last, 6),
                "n_params": self.n_params, "frozen_backbone_zero_grad": True}
