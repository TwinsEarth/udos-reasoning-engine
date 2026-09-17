"""
置信度校准与经验预测区间 (v2.3.0)
================================
回应 v2.2.1 的实测短板 C1：CTM 的 certainty=1−归一化同步熵只度量内部同步,
并非与误差对齐的预测置信 (2.2.1 上最高置信档误差反而更高)。本模块在**不重训、
零额外可学参数**的前提下, 用确定性后处理把原始置信校准到经验精度:

  - ConfidenceCalibrator: 保序回归 (PAVA, 零第三方依赖) 拟合 原始置信 -> 经验精度,
    单调非降、可序列化、随 checkpoint 持久化;
  - 回归式可靠性/ECE: 连续回归先以校准集误差均值 scale 把逐样本误差映射为 (0,1] 的
    经验精度 a=exp(-e/scale), 再等频分桶算 ECE 与置信-误差 Spearman 秩相关;
  - fit_predictor_calibration: 在独立校准集上一次拟合并产出 校准前/后对照诊断,
    同时统计每步自由 rollout 的残差分位, 供 F3 split-conformal 预测区间使用。

设计原则: 全部确定性、纯前向; 空/退化输入显式报错或降级, 不静默给出假结论。
"""



from __future__ import annotations

import logging

logger = logging.getLogger("udos.calibration")


from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch


# v2.4.4 多名义水平 split-conformal 允许的 alpha (对应 80%/90%/95%); 默认 0.1
ALLOWED_ALPHAS = (0.2, 0.1, 0.05)


# --------------------------------------------------------------------------- #
# 基础统计: 平均排名 Spearman (零 scipy 依赖)
# --------------------------------------------------------------------------- #
def average_ranks(x: torch.Tensor) -> torch.Tensor:
    """1-D 张量的平均排名 (1..n), 并列取平均, 结果形状同输入。"""
    x = x.reshape(-1)
    n = x.numel()
    order = torch.argsort(x)
    sorted_v = x[order]
    ranks = torch.empty(n, dtype=torch.float32)
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_v[j] == sorted_v[i]:
            j += 1
        ranks[i:j] = (i + 1 + j) / 2.0  # 并列段平均排名 [i+1..j]
        i = j
    out = torch.empty(n, dtype=torch.float32)
    out[order] = ranks
    return out


def spearman(x: torch.Tensor, y: torch.Tensor) -> float:
    """Spearman 秩相关; 任一列为常量时相关无定义, 返回 0.0 并由调用方知悉。"""
    rx, ry = average_ranks(x), average_ranks(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = float(torch.sqrt((rx ** 2).sum() * (ry ** 2).sum()))
    if denom <= 1e-12:
        return 0.0
    return float((rx * ry).sum() / denom)


# --------------------------------------------------------------------------- #
# 保序回归 (PAVA, 单调非降)
# --------------------------------------------------------------------------- #
@dataclass
class _Block:
    sx: float
    sy: float
    w: int

    @property
    def xbar(self) -> float:
        return self.sx / self.w

    @property
    def ybar(self) -> float:
        return self.sy / self.w


def _pava_nondec(xs: Sequence[float], ys: Sequence[float]
                 ) -> List[Tuple[float, float]]:
    """Pool Adjacent Violators, 返回单调非降的 (块x均值, 块y均值) 序列。"""
    blocks: List[_Block] = []
    for x, y in zip(xs, ys):
        blk = _Block(float(x), float(y), 1)
        while blocks and blocks[-1].ybar > blk.ybar + 1e-12:
            prev = blocks.pop()          # 违反非降: 合并后继续回溯
            blk = _Block(prev.sx + blk.sx, prev.sy + blk.sy, prev.w + blk.w)
        blocks.append(blk)
    return [(b.xbar, b.ybar) for b in blocks]


class ConfidenceCalibrator:
    """
    单调校准: 在校准集上学习 原始置信 c -> 经验精度 a 的保序映射。
    transform 用块代表点分段线性插值, 区间外 clip 到端点 (仍单调非降)。
    """

    def __init__(self) -> None:
        self.x_levels: List[float] = []
        self.y_levels: List[float] = []
        self.scale: float = float("nan")  # 误差 -> 精度 的尺度 (校准集误差均值)
        self.n_fit: int = 0
        self.pava_blocks: int = 0          # PAVA 真实块数 (<=1 即原始置信无有效单调信息)

    # ---- 拟合 ---- #
    def fit(self, raw_conf: torch.Tensor, sample_err: torch.Tensor
            ) -> "ConfidenceCalibrator":
        conf = raw_conf.reshape(-1).to(torch.float32)
        err = sample_err.reshape(-1).to(torch.float32)
        if conf.numel() < 2:
            raise ValueError("校准至少需要 2 个样本")
        if conf.shape != err.shape:
            raise ValueError("置信与误差形状不一致")
        scale = float(err.mean())
        if not (scale > 0):
            # 零误差 -> 精度恒为 1, 校准无意义; 用极小正尺度退化为恒 1 映射并保留痕迹
            scale = 1e-8
        self.scale = scale
        acc = torch.exp(-err / scale)                     # (0,1] 经验精度
        order = torch.argsort(conf)
        xs = conf[order].tolist()
        ys = acc[order].tolist()
        levels = _pava_nondec(xs, ys)
        self.pava_blocks = len(levels)
        # 相同块代表 x 去重 (PAVA 已合并 y 违规, 这里理论上 x 严格递增)
        self.x_levels = [float(v) for v, _ in levels]
        self.y_levels = [float(v) for _, v in levels]
        self.n_fit = int(conf.numel())
        if len(self.x_levels) < 2:
            # 全部塌缩一块 (原始置信与精度无单调一致关系): 退化为近似全局平均精度,
            # 由 pava_blocks<=1 让调用方标注"原始置信无有效排序信息"
            ym = self.y_levels[0] if self.y_levels else float(acc.mean())
            self.x_levels = [float(conf.min()), float(conf.max())]
            self.y_levels = [ym, ym]
        return self

    @property
    def fitted(self) -> bool:
        return len(self.x_levels) >= 2 and self.n_fit >= 2

    @property
    def num_segments(self) -> int:
        # 真实信息量以 PAVA 块数计 (退化补点不增加信息量)
        return max(self.pava_blocks, 1)

    @property
    def is_degenerate(self) -> bool:
        """原始置信与经验精度不具备单调一致关系 (保序被压成常量)。"""
        return self.pava_blocks <= 1

    # ---- 应用 ---- #
    @torch.no_grad()
    def transform(self, raw_conf: torch.Tensor) -> torch.Tensor:
        if not self.fitted:
            raise RuntimeError("校准器尚未拟合")
        c = raw_conf.to(torch.float32)
        xt = torch.tensor(self.x_levels, dtype=torch.float32)
        yt = torch.tensor(self.y_levels, dtype=torch.float32)
        flat = c.reshape(-1).contiguous()
        idx = torch.searchsorted(xt, flat).clamp_(1, len(xt) - 1)
        x0, x1 = xt[idx - 1], xt[idx]
        y0, y1 = yt[idx - 1], yt[idx]
        denom = (x1 - x0)
        t = torch.where(denom.abs() <= 1e-12, torch.zeros_like(flat),
                        (flat - x0) / denom.clamp_min(1e-12))
        out = (y0 + t * (y1 - y0)).clamp_(0.0, 1.0)
        # 区间外 clip 到端点
        out = torch.where(flat <= xt[0], yt[0], out)
        out = torch.where(flat >= xt[-1], yt[-1], out)
        return out.reshape_as(c)

    # ---- 序列化 (纯 python 标量, 可入 torch.save bundle) ---- #
    def state_dict(self) -> Dict[str, Any]:
        return {"x_levels": list(self.x_levels), "y_levels": list(self.y_levels),
                "scale": self.scale, "n_fit": self.n_fit,
                "pava_blocks": self.pava_blocks}

    def load_state_dict(self, state: Dict[str, Any]) -> "ConfidenceCalibrator":
        self.x_levels = [float(v) for v in state["x_levels"]]
        self.y_levels = [float(v) for v in state["y_levels"]]
        self.scale = float(state.get("scale", float("nan")))
        self.n_fit = int(state.get("n_fit", len(self.x_levels)))
        self.pava_blocks = int(state.get("pava_blocks", len(self.x_levels)))
        return self


# --------------------------------------------------------------------------- #
# 温度缩放校准 (v2.4.2): 单参数 T, 单调, 网格搜索最优
# --------------------------------------------------------------------------- #
class TemperatureScaling:
    """
    把原始置信 c ∈ (0,1) 视为 logit 分数: z=log(c/(1-c)), 经温度 T 缩放后再
    sigmoid 回到 [0,1]:  c' = sigmoid(z/T)。
      - T=1 => 恒等变换 (逐位等价原始置信);
      - T>1 => 向 0.5 收缩 (过置信校正); T<1 => 向两端锐化。
    单调保序、零额外参数 (仅 T), 网格搜索使校准集 ECE 最小的 T。
    与 ConfidenceCalibrator 鸭子类型同构: 暴露 scale/transform/fitted/序列化,
    供 calibration_report 与 persistence 复用同一管线。
    """

    KIND = "temperature"

    def __init__(self, temperature: float = 1.0) -> None:
        self.temperature = float(temperature)
        self.scale: float = float("nan")
        self.n_fit: int = 0

    @property
    def fitted(self) -> bool:
        return self.n_fit >= 2 and self.temperature > 0

    @property
    def num_segments(self) -> int:
        return 1  # 温度缩放为单调单参数映射

    @property
    def is_degenerate(self) -> bool:
        return False  # 恒单调, 不退化

    # ---- 拟合: 网格搜索使 ECE 最小的 T ---- #
    def fit(self, raw_conf: torch.Tensor, sample_err: torch.Tensor,
            t_grid: Optional[Sequence[float]] = None) -> "TemperatureScaling":
        conf = raw_conf.reshape(-1).to(torch.float32)
        err = sample_err.reshape(-1).to(torch.float32)
        if conf.numel() < 2:
            raise ValueError("校准至少需要 2 个样本")
        if conf.shape != err.shape:
            raise ValueError("置信与误差形状不一致")
        scale = float(err.mean())
        if not (scale > 0):
            scale = 1e-8
        self.scale = scale
        self.n_fit = int(conf.numel())   # 先标记, 使网格搜索内部 transform 可用
        if t_grid is None:
            # 对数网格覆盖 0.2..5.0, 含 T=1
            t_grid = [round(0.2 * (5.0 ** (i / 24.0)), 4) for i in range(25)]
            if 1.0 not in t_grid:
                t_grid = sorted(set(t_grid) | {1.0})
        best_t, best_ece = 1.0, float("inf")
        for t in t_grid:
            if t <= 0:
                continue
            self.temperature = float(t)
            tr = reliability(self.transform(conf), err, self.scale, n_bins=5)
            if tr["ece"] < best_ece:
                best_ece, best_t = tr["ece"], float(t)
        self.temperature = best_t
        return self

    @torch.no_grad()
    def transform(self, raw_conf: torch.Tensor) -> torch.Tensor:
        if not self.fitted:
            raise RuntimeError("温度缩放尚未拟合")
        c = raw_conf.to(torch.float32).clamp(1e-6, 1.0 - 1e-6)
        z = torch.log(c / (1.0 - c))
        return torch.sigmoid(z / self.temperature)

    # ---- 序列化 ---- #
    def state_dict(self) -> Dict[str, Any]:
        return {"kind": self.KIND, "temperature": self.temperature,
                "scale": self.scale, "n_fit": self.n_fit}

    def load_state_dict(self, state: Dict[str, Any]) -> "TemperatureScaling":
        self.temperature = float(state.get("temperature", 1.0))
        self.scale = float(state.get("scale", float("nan")))
        self.n_fit = int(state.get("n_fit", 0))
        return self


# --------------------------------------------------------------------------- #
# 回归可靠性 / ECE
# --------------------------------------------------------------------------- #
@torch.no_grad()
def reliability(conf: torch.Tensor, err: torch.Tensor, scale: float,
                n_bins: int = 5) -> Dict[str, Any]:
    """
    等频分桶的回归可靠性: a=exp(-err/scale) 为经验精度, ECE=Σ w|acc-conf|。
    返回 ece / bins / spearman_conf_err (期望为负: 置信越高误差越低)。
    """
    conf = conf.reshape(-1).to(torch.float32)
    err = err.reshape(-1).to(torch.float32)
    n = conf.numel()
    if n == 0:
        raise ValueError("reliability 需要非空样本")
    acc = torch.exp(-err / max(scale, 1e-8))
    order = torch.argsort(conf)
    edges = torch.linspace(0, n, min(n_bins, n) + 1).long()
    bins: List[Dict[str, float]] = []
    ece = 0.0
    nb = int(edges.numel() - 1)
    for i in range(nb):
        lo, hi = int(edges[i]), max(int(edges[i + 1]), int(edges[i]) + 1)
        idx = order[lo:hi]
        if idx.numel() == 0:
            continue
        mc = float(conf[idx].mean())
        ma = float(acc[idx].mean())
        bins.append({"bin": i, "n": int(idx.numel()),
                     "mean_confidence": round(mc, 4),
                     "mean_accuracy": round(ma, 4),
                     "gap": round(abs(ma - mc), 4)})
        ece += (idx.numel() / n) * abs(ma - mc)
    return {"ece": round(ece, 6), "bins": bins,
            "spearman_conf_err": round(spearman(conf, err), 4)}


@torch.no_grad()
def calibration_report(conf: torch.Tensor, err: torch.Tensor,
                        calibrator: ConfidenceCalibrator,
                        n_bins: int = 5) -> Dict[str, Any]:
    """同一 scale、同一批样本上的 校准前 vs 校准后 对照 (唯一变量是置信映射)。"""
    raw = reliability(conf, err, calibrator.scale, n_bins)
    cal_conf = calibrator.transform(conf)
    cal = reliability(cal_conf, err, calibrator.scale, n_bins)
    return {
        "scale": round(calibrator.scale, 6),
        "n_samples": int(conf.numel()),
        "num_segments": calibrator.num_segments,
        "ranking_informative": not calibrator.is_degenerate,
        "raw": raw,
        "calibrated": cal,
        "ece_reduction_x": round(raw["ece"] / max(cal["ece"], 1e-9), 3),
    }


# --------------------------------------------------------------------------- #
# 在数据集上拟合预测器 (校准 + F3 每步残差分位)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def collect_predictor_outputs(model, dataset, use_scene: bool = True
                              ) -> Tuple[torch.Tensor, torch.Tensor,
                                         torch.Tensor, torch.Tensor]:
    """
    前向收集 原始最终置信 c[N]、单步逐元素误差[N,RAW]、真值、以及自由 rollout
    逐步逐元素残差 [H,N,RAW]。供校准与区间共用同一前向口径。
    """
    model.eval()
    X, P, Y = dataset.X, dataset.P, dataset.Y
    scene = P if (use_scene and model.scene_encoder is not None) else torch.zeros_like(P)
    _, certs, single, _ = model(X, scene_params=scene)
    conf = certs[:, 1, -1]
    step_err = ((single - Y[:, 0, :]) ** 2).mean(dim=-1)        # [N]
    H = Y.size(1)
    roll = model.rollout(X, H, scene_params=scene)              # [N,H,RAW]
    resid = (roll - Y)                                          # 有符号残差 [N,H,RAW]
    return conf, step_err, Y[:, 0, :], resid


def _conservative_upper_quantile(vals: torch.Tensor, p: float) -> torch.Tensor:
    """
    split-conformal 有限样本**保守上分位**: 排序后向上取整到最近样本 (相对线性插值
    向"更宽"一侧取), 使经验区间在小校准集上不系统性偏窄。标量返回。
    """
    s, _ = torch.sort(vals.flatten().float())
    n = s.numel()
    pos = (n - 1) * float(p)
    idx = int(pos) if float(pos).is_integer() else int(pos) + 1   # ceil
    return s[min(max(idx, 0), n - 1)]


@torch.no_grad()
def compute_conformal_halfwidths(resid: torch.Tensor,
                                 alphas=ALLOWED_ALPHAS) -> Dict[float, List[torch.Tensor]]:
    """
    v2.4.4 一次计算多名义水平的对称半宽。resid: 自由 rollout 有符号残差 [N,H,R]。
    返回 {alpha: [H] 个 [R] 逐维 |残差| 的 (1-alpha) 保守上分位}。
    """
    H, R = resid.size(1), resid.size(2)
    out: Dict[float, List[torch.Tensor]] = {}
    for a in alphas:
        cov = 1.0 - float(a)
        out[float(a)] = [torch.stack([
            _conservative_upper_quantile(resid[:, h, r].abs(), cov)
            for r in range(R)], dim=0) for h in range(H)]
    return out


@torch.no_grad()
def fit_predictor_calibration(model, dataset, use_scene: bool = True,
                              interval_coverage: float = 0.9,
                              n_bins: int = 5,
                              method: str = "pava"
                              ) -> Tuple[Any, Dict[str, Any],
                                         List[torch.Tensor]]:
    """
    返回 (calibrator, report, half_widths[h]=每 raw 维 |残差| 的 (1-alpha) 上分位 [RAW])。
    采用标准回归 split-conformal: 对自由 rollout 的**绝对残差**|y-yhat| 逐步、逐维取
    (1-alpha) 上分位作为对称半宽, 区间 [yhat-q, yhat+q]。不用有符号双侧分位, 因为
    自回归残差有偏/重尾时双侧分位会系统性 undercover (实测 0.78 vs 对称式 0.90)。

    method:
      - "pava"        (默认): ConfidenceCalibrator 保序映射, 与 v2.3 逐位等价;
      - "temperature": TemperatureScaling 单参数温度缩放;
      - "none":        不做置信校准, calibrator=None (仍返回 conformal 半宽)。
    """
    conf, step_err, _, resid = collect_predictor_outputs(model, dataset, use_scene)
    if method == "pava":
        calibrator = ConfidenceCalibrator().fit(conf, step_err)
    elif method == "temperature":
        calibrator = TemperatureScaling().fit(conf, step_err)
    elif method == "none":
        calibrator = None
    else:
        raise ValueError(
            f"未知校准 method={method!r}, 可选 pava/temperature/none")
    if calibrator is None:
        report = {"method": "none", "ece_before": None,
                  "ece_after": None}
    else:
        report = calibration_report(conf, step_err, calibrator, n_bins)
        report["method"] = method
    H, R = resid.size(1), resid.size(2)
    half_widths = [torch.stack([
        _conservative_upper_quantile(resid[:, h, r].abs(), interval_coverage)
        for r in range(R)], dim=0)               # [RAW] 非负对称半宽
        for h in range(H)]
    # v2.4.4: 同时计算多名义水平半宽 (供 predict_interval(alpha=...) 切换),
    # 默认 0.1 路径仍用上面的 half_widths, 逐位不变。
    by_alpha = compute_conformal_halfwidths(resid)
    if isinstance(report, dict):
        report["conformal_by_alpha"] = {
            str(a): [q.tolist() for q in hw] for a, hw in by_alpha.items()}
        raw_ece = (report.get("raw") or {}).get("ece")
        cal_ece = (report.get("calibrated") or {}).get("ece")
        logger.info(
            "fit_predictor_calibration method=%s n_samples=%s ece_before=%s "
            "ece_after=%s reduction_x=%s",
            report.get("method"), report.get("n_samples"),
            None if raw_ece is None else round(float(raw_ece), 6),
            None if cal_ece is None else round(float(cal_ece), 6),
            report.get("ece_reduction_x"))
    return calibrator, report, half_widths
