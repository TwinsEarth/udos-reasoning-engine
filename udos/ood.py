"""
OOD / 分布漂移检测核心 (v2.4.0)
================================
在不重训、零额外可学参数的前提下, 用**纯统计**手段回答两个问题:
  1. 某个输入样本离训练分布多远? (马氏距离 Mahalanobis)
  2. 一整批新样本相对训练分布是否发生了系统性漂移? (双样本 KS 检验)

设计原则 (与 v2.3 校准模块一致):
  - 全部确定性、纯前向, 可随 checkpoint 外挂序列化;
  - 对特征 X: [N, D] (任意维), 拟合时只记均值 μ 与精度矩阵 Σ⁻¹;
  - 马氏距离用岭正则化协方差 (Σ + ridge·I) 求 Cholesky 逆, 避免病态;
  - 阈值取训练集马氏距离的 (1-alpha) 保守上分位 (与 conformal 同口径, 向上取整),
    ID 样本超过阈值即判 OOD; 该阈值是经验分位而非 χ² 理论值, 不做分布假设;
  - KS 双样本统计量 D=sup|F_n-F_m| 纯 torch 实现, 不依赖 scipy。

术语: 第二组件一律 GPM。
"""



from __future__ import annotations

import logging

logger = logging.getLogger("udos.ood")


from typing import Any, Dict, Optional

import torch


# --------------------------------------------------------------------------- #
# 双样本 KS 统计量 (零 scipy)
# --------------------------------------------------------------------------- #
def ks_two_sample(x: torch.Tensor, y: torch.Tensor) -> float:
    """
    一维双样本 Kolmogorov-Smirnov 统计量 D = sup_t |F_n(t) - F_m(t)|。
    x/y 任意形状, 展平为 1D; 在合并样本点上用 searchsorted 计算经验 CDF 差。
    纯前向、确定性。
    """
    a = torch.sort(x.reshape(-1).float()).values
    b = torch.sort(y.reshape(-1).float()).values
    n, m = a.numel(), b.numel()
    if n == 0 or m == 0:
        raise ValueError("ks_two_sample 需要两个非空样本")
    pts = torch.sort(torch.cat([a, b])).values
    fa = torch.searchsorted(a, pts, right=True).float() / n
    fb = torch.searchsorted(b, pts, right=True).float() / m
    return float((fa - fb).abs().max())


class DistributionDriftDetector:
    """
    训练分布参考 -> 马氏距离打分 + 经验阈值判异 + KS 漂移诊断。

    典型用法:
        det = DistributionDriftDetector().fit(train_features)
        scores = det.score(test_features)          # 每样本马氏距离
        flags  = det.is_ood(test_features)         # 布尔掩码
        drift  = det.ks_drift(test_features)       # 整批漂移诊断 dict
    """

    def __init__(self, ridge: float = 1e-3, alpha: float = 0.05) -> None:
        """
        ridge: 协方差对角加正项, 保证正定可逆 (D 维较大/样本少时尤需)。
        alpha: OOD 误报率名义水平; 阈值 = 训练分位 (1-alpha) 保守上分位。
        """
        self.ridge = float(ridge)
        self.alpha = float(alpha)
        self.mean_: Optional[torch.Tensor] = None       # [D]
        self.precision_: Optional[torch.Tensor] = None  # [D,D]
        self.dim: Optional[int] = None
        self.n_fit: int = 0
        self.threshold_: float = float("nan")
        self.train_score_mean_: float = float("nan")
        self.train_score_std_: float = float("nan")

    # ---------------- 拟合 ---------------- #
    def fit(self, X: torch.Tensor) -> "DistributionDriftDetector":
        """X: [N, D] (或可展平为 [N, D] 的任意前导维张量)。"""
        X = self._flatten(X)
        if X.size(0) < 2:
            raise ValueError("OOD 检测器至少需要 2 个样本")
        n, d = X.shape
        self.dim = d
        mu = X.mean(dim=0)
        centered = X - mu
        # 样本协方差 (无偏), 岭正则化保证正定
        cov = (centered.T @ centered) / max(n - 1, 1)
        cov = cov + self.ridge * torch.eye(d, dtype=torch.float32)
        try:
            L = torch.linalg.cholesky(cov)
            eye = torch.eye(d, dtype=torch.float32)
            prec = torch.cholesky_solve(eye, L)
        except Exception:
            # 极端病态时退化为对角精度 (仅用方差), 不静默崩溃
            logger.warning(
                "ood cholesky 失败, 退化为对角精度 (仅用方差) dim=%d", d)
            prec = torch.diag(1.0 / cov.diagonal().clamp_min(1e-8)).contiguous()
        self.mean_ = mu
        self.precision_ = prec
        self.n_fit = int(n)
        train_scores = self._mahal(X)
        # 经验阈值: 保守 (1-alpha) 上分位, 与 conformal 同口径向上取整
        self.threshold_ = float(
            self._conservative_quantile(train_scores, 1.0 - self.alpha))
        self.train_score_mean_ = float(train_scores.mean())
        self.train_score_std_ = float(train_scores.std())
        return self

    @property
    def fitted(self) -> bool:
        return self.mean_ is not None and self.precision_ is not None

    # ---------------- 打分 ---------------- #
    @staticmethod
    def _flatten(X: torch.Tensor) -> torch.Tensor:
        X = X.float()
        if X.dim() == 1:
            X = X.unsqueeze(0)
        if X.dim() > 2:
            X = X.reshape(X.size(0), -1)
        return X

    def _mahal(self, X: torch.Tensor) -> torch.Tensor:
        centered = X - self.mean_
        m2 = ((centered @ self.precision_) * centered).sum(dim=-1)  # d^2
        return m2.clamp_min(0.0).sqrt()                             # d (>=0)

    @torch.no_grad()
    def score(self, X: torch.Tensor) -> torch.Tensor:
        """每样本马氏距离 d = sqrt((x-μ)ᵀ Σ⁻¹ (x-μ)); 越大越 OOD。"""
        if not self.fitted:
            raise RuntimeError("OOD 检测器尚未拟合")
        return self._mahal(self._flatten(X))

    @torch.no_grad()
    def is_ood(self, X: torch.Tensor,
               threshold: Optional[float] = None) -> torch.Tensor:
        """布尔掩码: 马氏距离 > (拟合阈值或显式 threshold) 判为 OOD。"""
        thr = self.threshold_ if threshold is None else float(threshold)
        return self.score(X) > thr

    @staticmethod
    def _conservative_quantile(vals: torch.Tensor, q: float) -> torch.Tensor:
        """与 conformal 一致的保守上分位: 排序后向上取整, 不系统性偏窄。"""
        s = torch.sort(vals.flatten().float()).values
        n = s.numel()
        pos = (n - 1) * float(q)
        idx = int(pos) if float(pos).is_integer() else int(pos) + 1
        return s[min(max(idx, 0), n - 1)]

    # ---------------- KS 漂移诊断 ---------------- #
    @torch.no_grad()
    def ks_drift(self, X: torch.Tensor) -> Dict[str, Any]:
        """
        一批新样本的漂移摘要: 用拟合阈值统计其马氏距离分布与 OOD 命中率,
        并给出 d_max/d_mean 相对训练分布的倍数。逐维严格双样本 KS 请用
        ks_two_sample_vs_reference(ref_features, X)。
        """
        if not self.fitted:
            raise RuntimeError("OOD 检测器尚未拟合")
        Xt = self._flatten(X)
        if Xt.size(1) != self.dim:
            raise ValueError(
                f"特征维 {Xt.size(1)} 与拟合维 {self.dim} 不一致")
        scores = self._mahal(Xt)
        flags = scores > self.threshold_
        if bool(flags.any()):
            logger.warning(
                "ood drift detected n_test=%d ood_rate=%.4f "
                "score_mean_x_train=%.3f threshold=%.4f",
                int(Xt.size(0)), float(flags.float().mean()),
                float(scores.mean()) / max(self.train_score_mean_, 1e-8),
                float(self.threshold_))
        ref_mu = max(self.train_score_mean_, 1e-8)
        return {
            "n_ref": self.n_fit, "n_test": int(Xt.size(0)),
            "threshold": round(self.threshold_, 6),
            "train_score_mean": round(self.train_score_mean_, 6),
            "train_score_std": round(self.train_score_std_, 6),
            "test_score_mean": round(float(scores.mean()), 6),
            "test_score_max": round(float(scores.max()), 6),
            "score_mean_x_train": round(float(scores.mean()) / ref_mu, 3),
            "ood_rate": round(float(flags.float().mean()), 4),
            "note": "逐维严格 KS 请用 ks_two_sample_vs_reference(ref, x)",
        }

    @torch.no_grad()
    def ks_two_sample_vs_reference(self, ref_features: torch.Tensor,
                                   test_features: torch.Tensor) -> Dict[str, Any]:
        """
        严格双样本 KS: ref_features/test_features 均为 [N, D]。
        逐维计算 KS 统计量, 聚合 d_max (最漂移维) 与 d_mean。
        """
        ref = self._flatten(ref_features)
        test = self._flatten(test_features)
        if ref.size(1) != test.size(1):
            raise ValueError("参考与测试特征维不一致")
        ds = [ks_two_sample(ref[:, d], test[:, d])
              for d in range(ref.size(1))]
        ds_t = torch.tensor(ds)
        return {
            "ks_d_max": round(float(ds_t.max()), 6),
            "ks_d_mean": round(float(ds_t.mean()), 6),
            "n_ref": int(ref.size(0)), "n_test": int(test.size(0)),
        }

    # ---------------- 序列化 (外挂, 纯张量/标量) ---------------- #
    def state_dict(self) -> Dict[str, Any]:
        if not self.fitted:
            raise RuntimeError("未拟合无法序列化")
        return {
            "mean": self.mean_.tolist(),
            "precision": self.precision_.tolist(),
            "dim": self.dim, "n_fit": self.n_fit,
            "ridge": self.ridge, "alpha": self.alpha,
            "threshold": self.threshold_,
            "train_score_mean": self.train_score_mean_,
            "train_score_std": self.train_score_std_,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> "DistributionDriftDetector":
        self.mean_ = torch.tensor(state["mean"], dtype=torch.float32)
        self.precision_ = torch.tensor(state["precision"], dtype=torch.float32)
        self.dim = int(state["dim"])
        self.n_fit = int(state["n_fit"])
        self.ridge = float(state.get("ridge", 1e-3))
        self.alpha = float(state.get("alpha", 0.05))
        self.threshold_ = float(state.get("threshold", float("nan")))
        self.train_score_mean_ = float(state.get("train_score_mean", float("nan")))
        self.train_score_std_ = float(state.get("train_score_std", float("nan")))
        return self


# --------------------------------------------------------------------------- #
# 在线流式漂移检测 (v2.4.14): 固定窗口在线均值/方差 + 与离线检测器同构接口
# --------------------------------------------------------------------------- #
class StreamingDriftDetector(DistributionDriftDetector):
    """
    流式漂移检测器 (opt-in, 不改变离线检测器任何行为)。
    继承 DistributionDriftDetector 的参考分布拟合 (μ + 岭精度矩阵) 与马氏打分,
    额外维护一个**固定长度 W 的滑动窗口**:
      - fit(X):    拟合参考分布 (与离线逐位同口径, 含经验阈值);
      - update(x): 推入单条 [D] 样本; 窗口满则弹出最旧 (环形缓冲, 充分统计量精确);
      - score(X)/is_ood(X): 与离线同语义, 相对参考分布的马氏距离;
      - reset():   仅清空滑动窗口, 不重置参考分布/阈值。
    窗口均值/方差 = 对窗口内样本批计算的 mean/var(无偏=False), 逐位一致 (流式==批处理)。
    """

    def __init__(self, window: int = 256, ridge: float = 1e-3,
                 alpha: float = 0.05) -> None:
        super().__init__(ridge=ridge, alpha=alpha)
        if window < 1:
            raise ValueError("window 必须 >= 1")
        self.window = int(window)
        self._buf: list = []

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        """推入单条特征 x ([D] 或 [1,D]); 窗口满时弹出最旧样本。"""
        xt = self._flatten(x)                       # [1,D]
        if self.dim is not None and xt.size(1) != self.dim:
            raise ValueError(
                f"特征维 {xt.size(1)} 与参考维 {self.dim} 不一致")
        vec = xt.reshape(-1).detach().to(torch.float32).clone()
        if len(self._buf) >= self.window:
            self._buf.pop(0)
        self._buf.append(vec)

    @property
    def n_window(self) -> int:
        return len(self._buf)

    def window_mean(self) -> Optional[torch.Tensor]:
        """当前滑动窗口逐维均值 [D]; 空窗口返回 None。"""
        if not self._buf:
            return None
        return torch.stack(self._buf, dim=0).mean(dim=0)

    def window_var(self) -> Optional[torch.Tensor]:
        """当前滑动窗口逐维总体方差 [D] (var unbiased=False); 空窗口返回 None。"""
        if len(self._buf) < 2:
            return None
        return torch.stack(self._buf, dim=0).var(dim=0, unbiased=False)

    @torch.no_grad()
    def window_drift_score(self) -> float:
        """当前窗口均值相对参考分布的马氏距离 (窗口整体漂移程度); 空窗口 nan。"""
        if not self.fitted:
            raise RuntimeError("检测器尚未 fit")
        if not self._buf:
            return float("nan")
        return float(self.score(self.window_mean().unsqueeze(0)))

    def reset(self) -> None:
        """清空滑动窗口 (保留参考分布与阈值)。"""
        self._buf.clear()

