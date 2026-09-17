"""
CTM 物理学习闭环 (v2.0.0)
========================
一代引擎权重随机、只验证数据流; 二代提供可训练的物理预测器与训练器,
在合成动力学上用自监督"历史窗口 -> 下一状态"真正把损失降下来。

PhysicsPredictor = 观测编码器 + CTM(内部时间轴) + 物理解码头
CTMTrainer       = 下一状态 MSE + 置信度正则(鼓励最终 tick 低熵高置信)
                   + 可选逐 tick 深度监督(鼓励随内部时间轴收敛)
"""



from __future__ import annotations

import logging

logger = logging.getLogger("udos.training")


from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from .ctm_engine import CTMConfig, CTMPhysicsEngine
from .dynamics import RAW_DIM, DynamicsDataset, kinematic_residual, noise_augment


def set_seed(seed: int = 0, deterministic: bool = True) -> None:
    """v2.2.0 统一确定性入口: 固定 torch 种子 (CPU) 与算法确定性。"""
    torch.manual_seed(seed)
    if deterministic:
        torch.use_deterministic_algorithms(False)  # CPU 上前向确定性; 不强制可能无实现的算子


class PhysicsPredictor(nn.Module):
    """原始运动学序列 -> CTM 内部推演 -> 下一时刻运动学状态。"""

    def __init__(self, ctm_config: CTMConfig, raw_dim: int = RAW_DIM,
                 scene_param_dim: Optional[int] = None):
        super().__init__()
        self.raw_dim = raw_dim
        self.scene_param_dim = scene_param_dim
        self.obs_encoder = nn.Sequential(
            nn.Linear(raw_dim, ctm_config.d_input),
            nn.LayerNorm(ctm_config.d_input),
            nn.GELU(),
            nn.Linear(ctm_config.d_input, ctm_config.d_input),
            nn.LayerNorm(ctm_config.d_input),
        )
        # v2.1: 场景隐藏物理参数 -> CTM 场景条件 (联合训练, 真正激活 scene_gate)
        if scene_param_dim is not None:
            if ctm_config.scene_dim is None:
                ctm_config.scene_dim = 32
            sd = ctm_config.scene_dim
            self.scene_encoder = nn.Sequential(
                nn.Linear(scene_param_dim, sd), nn.LayerNorm(sd), nn.GELU(),
                nn.Linear(sd, sd))
        else:
            self.scene_encoder = None
        # 训练时关闭基于熵的提前停止, 保证每个 batch 都跑满 ticks 以做深度监督
        ctm_config.certainty_threshold = 0.0
        self.ctm = CTMPhysicsEngine(ctm_config)
        self.state_decoder = nn.Sequential(
            nn.Linear(ctm_config.out_dims, ctm_config.out_dims),
            nn.GELU(),
            nn.Linear(ctm_config.out_dims, raw_dim),
        )
        # v2.3 校准与经验区间为外挂后处理 (非 nn.Module 参数, 不进 state_dict)
        self.calibrator = None              # ConfidenceCalibrator
        self.residual_quantiles = None      # List[Tensor(2,)] 每步有符号残差分位
        # v2.4.4 多名义水平 conformal 半宽: {alpha: [H] 个 [R] 张量}
        self.conformal_by_alpha = None      # Dict[float, List[Tensor]]
        # v2.4.0 OOD/漂移检测器为外挂后处理 (纯统计, 不进 state_dict)
        self.ood_detector = None            # DistributionDriftDetector
        # v2.4.5 退化守卫 (外挂, 默认不激活)
        self.guard = None                   # PredictionGuard
        # v2.6.0 混合物理修正 (外挂可学模块, 默认 None => 走旧路径逐位一致)
        self.hybrid = None                  # HybridPhysicsCorrector

    def _resolve_context(self, scene_context, scene_params, scene_bias=None):
        ctx = scene_context
        if ctx is None and scene_params is not None \
                and self.scene_encoder is not None:
            ctx = self.scene_encoder(scene_params)
        # v5.4.7 GPM 记忆桥: 加性场景偏置。零偏置短路 => 零初始化桥接入瞬间
        # 不改变 ctx (尤其场景盲时保持 None, 避免零张量经 scene_proj 偏置失真)。
        if scene_bias is not None and float(scene_bias.abs().sum()) > 0.0:
            ctx = scene_bias if ctx is None else ctx + scene_bias
        return ctx

    def forward(self, raw_seq: torch.Tensor,
                scene_context: Optional[torch.Tensor] = None,
                scene_params: Optional[torch.Tensor] = None,
                scene_bias: Optional[torch.Tensor] = None):
        """
        raw_seq: [B, W, RAW_DIM]
        scene_bias (v5.4.7): 可选 [B, scene_dim] 加性场景记忆 (来自 GPM 桥);
            零张量/None 时与旧版逐位一致。
        返回:
            decoded_ticks [B, RAW_DIM, T] 每个内部 tick 对应的物理预测
            certainties    [B, 2, T]
            last           [B, RAW_DIM]   最终 tick 预测
            info
        """
        scene_context = self._resolve_context(
            scene_context, scene_params, scene_bias)
        h = self.obs_encoder(raw_seq)
        preds, certs, _, info = self.ctm(h, scene_context=scene_context)
        # preds [B, out, T] -> 逐 tick 解码: [B, T, out] -> decoder -> [B, T, RAW]
        B, out, T = preds.shape
        per_tick = preds.transpose(1, 2).reshape(B * T, out)
        decoded = self.state_decoder(per_tick).view(B, T, self.raw_dim)
        decoded_ticks = decoded.transpose(1, 2)  # [B, RAW, T]
        return decoded_ticks, certs, decoded_ticks[..., -1], info

    @torch.no_grad()
    def predict_next(self, raw_seq: torch.Tensor,
                     scene_context: Optional[torch.Tensor] = None,
                     scene_params: Optional[torch.Tensor] = None,
                     scene_bias: Optional[torch.Tensor] = None,
                     guard: bool = False,
                     hybrid: bool = False,
                     dt: float = 0.5
                     ) -> torch.Tensor:
        """
        v2.4.5: guard=True 时对输出过 PredictionGuard (NaN/inf 回退 + 越界截断);
        默认 guard=False 与旧版逐位一致 (透传)。
        v2.6.0: hybrid=True 且已挂载 HybridPhysicsCorrector 时, 用一阶欧拉骨架+学习残差
        校正输出 (last_state=窗口末帧); 默认 hybrid=False 与旧版逐位一致。
        v2.6.1:
          * scene_params 含 NaN/inf 时显式 ValueError (不静默传播 inf);
          * guard 在 hybrid 之后执行 => guard 清洗的是 hybrid 修正后的最终输出
            (hybrid=False 时与旧路径逐位一致)。
        v5.4.7: scene_bias (GPM 记忆桥) 透传, 含 NaN/inf 同样显式拒绝。
        """
        self.eval()
        if scene_params is not None:
            sp = torch.as_tensor(scene_params, dtype=torch.float32)
            if not bool(torch.isfinite(sp).all()):
                raise ValueError(
                    "scene_params 含 NaN/inf 非有限值 (拒绝静默 inf 传播); "
                    "请提供有限场景参数或显式处置")
        if scene_bias is not None:
            sb = torch.as_tensor(scene_bias, dtype=torch.float32)
            if not bool(torch.isfinite(sb).all()):
                raise ValueError(
                    "scene_bias 含 NaN/inf 非有限值 (拒绝静默 inf 传播)")
        out = self(raw_seq, scene_context=scene_context,
                    scene_params=scene_params,
                    scene_bias=scene_bias)[2]
        if hybrid and self.hybrid is not None:
            out = self.hybrid(out, raw_seq[:, -1, :], dt, enabled=True)
        if guard:
            g = self.guard
            if g is None:
                from .guard import PredictionGuard
                g = PredictionGuard()
                self.guard = g
            out = g.sanitize(out)
        return out

    def attach_hybrid(self, corrector) -> None:
        """v2.6.0 挂载混合物理修正模块 (可参与训练/state_dict)。"""
        self.hybrid = corrector

    def detach_hybrid(self) -> None:
        """v2.6.0 卸载混合物理修正模块 => 回到旧输出路径。"""
        self.hybrid = None

    @torch.no_grad()
    def predict_next_adaptive(self, raw_seq: torch.Tensor,
                              scene_params: Optional[torch.Tensor] = None,
                              stopper=None) -> torch.Tensor:
        """
        v2.6.0+dev3 自适应 iterations: wrapper 方式跑满全部 CTM ticks, 再依据
        AdaptiveStopper 在已解码逐 tick 输出里选最早收敛的 tick。
        stopper=None 时返回最终 tick, 与旧 predict_next 逐位一致 (不真截断 CTM)。
        """
        self.eval()
        decoded_ticks, certs, last, _ = self(raw_seq, scene_params=scene_params)
        if stopper is None:
            return last
        cert_traj = certs[:, 1, :].mean(dim=0).detach().cpu().tolist()
        chosen = decoded_ticks.shape[-1] - 1
        for t in range(len(cert_traj)):
            if stopper.should_stop(cert_traj[:t + 1]):
                chosen = t
                break
        return decoded_ticks[..., chosen]

    def attach_guard(self, guard) -> None:
        """挂载退化守卫 (后处理, 不参与训练)。"""
        self.guard = guard

    @torch.no_grad()
    def predict_batch(self, sequences,
                      scene_params: Optional[torch.Tensor] = None,
                      guard: bool = False,
                      cache: Optional[Any] = None,
                      max_shard: int = 64) -> torch.Tensor:
        """
        v2.5.0 批量下一状态预测 (委托 BatchPredictor)。
        支持 List[Tensor(W_i,RAW_DIM)] 变长序列 或 Tensor [B,W,RAW_DIM] 等长批量;
        结果与逐笔 predict_next 逐位一致 (atol=1e-5)。可选 InferenceCache (opt-in)。
        """
        from .batch import BatchPredictor
        return BatchPredictor(self, max_shard=max_shard).predict(
            sequences, scene_params=scene_params, guard=guard, cache=cache)

    @torch.no_grad()
    def rollout(self, raw_window: torch.Tensor, horizon: int,
                scene_context: Optional[torch.Tensor] = None,
                scene_params: Optional[torch.Tensor] = None,
                scene_bias: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        """
        v2.1 多步自由滚动推演: 把每步预测拼回窗口递归预测 H 步。
        raw_window [B,W,RAW_DIM] -> [B,H,RAW_DIM]
        v5.4.7: scene_bias (GPM 记忆桥) 在 ctx 解析时一次性并入, 多步复用。
        """
        self.eval()
        ctx = self._resolve_context(scene_context, scene_params, scene_bias)
        window, outs = raw_window, []
        for _ in range(horizon):
            nxt = self(window, scene_context=ctx)[2]
            outs.append(nxt)
            window = torch.cat([window[:, 1:, :], nxt.unsqueeze(1)], dim=1)
        return torch.stack(outs, dim=1)

    # ---------------- v2.3 校准置信与 split-conformal 区间 ---------------- #
    def attach_calibration(self, calibrator, residual_quantiles=None,
                           conformal_by_alpha=None) -> None:
        """挂载 F1 校准器与 F3 每步残差分位 (后处理, 不参与训练/state_dict)。
        v2.4.4 可选传入 {alpha: [H] 个 [R]} 多名义水平半宽。"""
        self.calibrator = calibrator
        if residual_quantiles is not None:
            self.residual_quantiles = [
                q.detach().to(torch.float32).flatten() for q in residual_quantiles]
        if conformal_by_alpha is not None:
            self.conformal_by_alpha = {
                float(a): [q.detach().to(torch.float32).flatten()
                           for q in hw]
                for a, hw in conformal_by_alpha.items()}

    @property
    def is_calibrated(self) -> bool:
        return getattr(self.calibrator, "fitted", False)

    # ---------------- v2.4.0 OOD/漂移检测 (外挂, opt-in) ---------------- #
    def attach_ood_detector(self, detector) -> None:
        """挂载 OOD 检测器 (后处理, 不参与训练/state_dict)。"""
        self.ood_detector = detector

    @property
    def has_ood_detector(self) -> bool:
        return getattr(self.ood_detector, "fitted", False)

    @torch.no_grad()
    def ood_score(self, raw_window: torch.Tensor) -> torch.Tensor:
        """
        v2.4.0 对历史窗口 [B,W,RAW] 展平为特征, 返回每样本马氏距离 ood_score。
        未挂载检测器时显式报错 (与 predict_interval 同纪律: 不静默退化)。
        """
        if not self.has_ood_detector:
            raise RuntimeError("尚未挂载 OOD 检测器, 请先 attach_ood_detector")
        self.eval()
        return self.ood_detector.score(raw_window)

    @torch.no_grad()
    def predict_interval(self, raw_window: torch.Tensor, horizon: int,
                         scene_context: Optional[torch.Tensor] = None,
                         scene_params: Optional[torch.Tensor] = None,
                         alpha: float = 0.1) -> Dict:
        """
        v2.3 marginal split-conformal 经验预测区间: 中值=自由 rollout 点预测,
        每步、**每个 raw 维**用校准集**绝对残差** |y-yhat| 的 (1-alpha) 上分位 q 作
        对称半宽: [mid-q, mid+q] (各物理量尺度不同, 逐维取半宽; 用绝对残差对称式
        而非有符号双侧分位, 避免自回归残差有偏/重尾时系统性 undercover)。
        v2.4.4: alpha∈{0.2,0.1,0.05}; 默认 alpha=0.1 与 v2.3 逐位等价。
        返回 median/lower/upper/half_width_by_step/alpha。
        """
        from .calibration import ALLOWED_ALPHAS
        if alpha not in ALLOWED_ALPHAS:
            raise ValueError(
                f"alpha={alpha} 不在允许集 {ALLOWED_ALPHAS}")
        # 选半宽: 默认 0.1 优先用 residual_quantiles (旧路径逐位不变);
        # 其他水平用多水平字典; 0.1 未挂旧分位时也可从字典取。
        quantiles = None
        if abs(alpha - 0.1) < 1e-12 and self.residual_quantiles is not None:
            quantiles = self.residual_quantiles
        elif self.conformal_by_alpha is not None \
                and float(alpha) in self.conformal_by_alpha:
            quantiles = self.conformal_by_alpha[float(alpha)]
        elif self.residual_quantiles is not None:
            quantiles = self.residual_quantiles
        else:
            raise RuntimeError("尚未挂载残差半宽, 请先 attach_calibration 拟合")
        mid = self.rollout(raw_window, horizon, scene_context=scene_context,
                           scene_params=scene_params)            # [B,H,R]
        B, H, R = mid.shape
        lowers, uppers, widths = [], [], []
        for h in range(H):
            q = quantiles[min(h, len(quantiles) - 1)]
            q = q.view(-1)                              # [R] 逐维非负对称半宽
            lowers.append(mid[:, h, :] - q)
            uppers.append(mid[:, h, :] + q)
            widths.append(float(q.mean()))             # 跨维平均半宽代表该步宽度
        return {
            "median": mid,
            "lower": torch.stack(lowers, dim=1),
            "upper": torch.stack(uppers, dim=1),
            "half_width_by_step": [round(w, 6) for w in widths],
            "alpha": alpha,
        }

    @torch.no_grad()
    def per_step_confidence(self, raw_window: torch.Tensor, horizon: int,
                            scene_context: Optional[torch.Tensor] = None,
                            scene_params: Optional[torch.Tensor] = None
                            ) -> torch.Tensor:
        """
        v2.4.10 逐步逐维置信矩阵 [B, H, RAW_DIM] (可解释性, opt-in 后处理, 不改旧输出)。
        步置信 = 该 rollout 步 CTM 标量 certainty (certs[:,1,-1]); 若已挂载校准器则先经
        calibrator.transform 校准。逐维: 若已挂载 conformal 半宽 (residual_quantiles),
        用 exp(-q / median_q) 作逐维可靠性因子调制 (区间越窄置信越高); 未挂半宽时逐维
        广播步置信 (无逐维信息时诚实退化为均匀, 此时 mean(dim=-1) 与标量置信逐位一致)。
        所有输出裁剪到 [0,1]。纯前向、确定性。
        """
        self.eval()
        ctx = self._resolve_context(scene_context, scene_params)
        B, W, R = raw_window.shape
        window = raw_window
        step_confs: List[torch.Tensor] = []
        for _ in range(horizon):
            _, certs, nxt, _ = self(window, scene_context=ctx)
            step_confs.append(certs[:, 1, -1])          # [B]
            window = torch.cat([window[:, 1:, :], nxt.unsqueeze(1)], dim=1)
        step_conf = torch.stack(step_confs, dim=1)       # [B,H]
        if getattr(self, "calibrator", None) is not None and self.is_calibrated:
            flat = step_conf.reshape(-1).contiguous()
            step_conf = self.calibrator.transform(flat).view(B, horizon)
        conf3d = step_conf.unsqueeze(-1).expand(B, horizon, R).clone()
        q = getattr(self, "residual_quantiles", None)
        if q is not None and len(q) > 0:
            qall = torch.cat([t.reshape(-1).float() for t in q], dim=0)
            med = float(qall.median()) + 1e-12
            for h in range(horizon):
                qh = q[min(h, len(q) - 1)].reshape(-1).float()     # [R]
                rel = torch.exp(-qh / med)                          # [R] in (0,1]
                conf3d[:, h, :] = step_conf[:, h].unsqueeze(-1) * rel.unsqueeze(0)
        return conf3d.clamp_(0.0, 1.0)


@dataclass
class TrainConfig:
    epochs: int = 60
    batch_size: int = 64
    lr: float = 3e-3
    weight_decay: float = 1e-4
    cert_weight: float = 0.02     # 置信度正则权重
    deep_supervision: bool = True  # 逐 tick 衰减加权监督
    phys_weight: float = 0.0       # v2.1 运动学一致性正则权重 (0=关闭, 同 v2.0)
    seed: int = 0
    # ---- v2.2.0 Scheduled Sampling (默认 ss_max=0 ⇒ 纯 teacher-forcing, 同 2.1) ----
    ss_max: float = 0.0            # 自回归喂入概率上限, 0 关闭
    ss_start: int = 0              # 开始爬坡的 epoch
    ss_warmup: int = 10            # 从 0 线性升到 ss_max 的 epoch 跨度
    # ---- v2.2.0 早停 (默认 patience=None ⇒ 不早停, 跑满 epochs, 同 2.1) ----
    patience: Optional[int] = None
    min_delta: float = 1e-4
    # ---- v2.3.0 多时域均衡损失 (默认 front=linspace(1,0.5), 与 2.2.1 逐元素等价) ----
    # front(默认,旧行为)/uniform(等权)/back(远期加权) 均可显式选择。v2.3 两组同合同 A/B:
    # uniform 在"无早停+特定种子"口径 3/3 远期更优, 但在"早停+build 口径"被 front 反超
    # (scripts/sensitivity_step_weight_pipeline.py), 优劣对训练口径敏感、不稳健,
    # 故沿用 SS 纪律: 不设为默认, 三方案均 opt-in, 默认 front 逐位复现 2.2.1。
    step_weight_scheme: str = "front"
    # ---- v2.4.3 训练输入高斯噪声增强 (默认 0.0 等价旧版; >0 注入 sigma 噪声) ----
    noise_sigma: float = 0.0
    # ---- v2.6.0 混合物理修正损失权重 (默认 0.0 关闭, 走旧路径逐位一致) ----
    # >0 且 model.hybrid 已挂载时, 对 hybrid 输出 (euler+residual) 按此权重加 MSE。
    hybrid_weight: float = 0.0
    # ---- v3.3.0.dev5 内存优化 (默认 False 关闭, 不改旧训练路径) ----
    # True: 训练时对每个 CTM 前向用 torch.utils.checkpoint 重算换显存 (CPU 上省激活内存)。
    gradient_checkpointing: bool = False

    STEP_WEIGHT_SCHEMES = ("front", "uniform", "back")

    def ss_prob(self, epoch: int) -> float:
        """第 epoch 轮的自回归喂入概率, 在 [0, ss_max] 间线性爬坡。"""
        if self.ss_max <= 0:
            return 0.0
        if self.ss_warmup <= 0:
            return float(self.ss_max)
        prog = min(1.0, max(0.0, (epoch - self.ss_start) / self.ss_warmup))
        return float(self.ss_max * prog)


@dataclass
class TrainHistory:
    train_loss: List[float] = field(default_factory=list)
    eval_mse: List[float] = field(default_factory=list)
    cert: List[float] = field(default_factory=list)
    phys_violation: List[float] = field(default_factory=list)  # v2.1
    ss_prob: List[float] = field(default_factory=list)         # v2.2 每轮 SS 概率
    best_epoch: Optional[int] = None                           # v2.2 早停最佳轮
    best_eval: Optional[float] = None
    stopped_early: bool = False

    def final(self) -> Dict[str, float]:
        out = {"train_loss": self.train_loss[-1],
               "first_train_loss": self.train_loss[0],
               "epochs_run": len(self.train_loss)}
        if self.eval_mse:
            out["eval_mse"] = self.eval_mse[-1]
            out["first_eval_mse"] = self.eval_mse[0]
        if self.best_eval is not None:
            out["best_epoch"] = self.best_epoch
            out["best_eval"] = self.best_eval
        return out


class CTMTrainer:
    def __init__(self, predictor: PhysicsPredictor,
                 config: Optional[TrainConfig] = None):
        self.model = predictor
        self.cfg = config or TrainConfig()
        self.opt = torch.optim.AdamW(predictor.parameters(),
                                    lr=self.cfg.lr,
                                    weight_decay=self.cfg.weight_decay)

    def _forward_cell(self, window: torch.Tensor, ctx):
        """v3.3.0.dev5: gradient_checkpointing 关时直接前向 (旧路径逐位一致);
        开时用 torch.utils.checkpoint 重算换激活内存。"""
        if self.cfg.gradient_checkpointing:
            import torch.utils.checkpoint as cp
            return cp.checkpoint(self.model, window, scene_context=ctx,
                                 use_reentrant=False)
        return self.model(window, scene_context=ctx)

    def _tick_weights(self, T: int, device) -> torch.Tensor:
        # 后期 tick 权重更高 (线性递增归一), 引导思考随时间收敛
        w = torch.linspace(0.2, 1.0, T, device=device)
        return w / w.sum()

    def _loss(self, decoded_ticks, certs, y):
        B, R, T = decoded_ticks.shape
        target = y.unsqueeze(-1).expand(-1, -1, T)             # [B,R,T]
        se = ((decoded_ticks - target) ** 2).mean(dim=1)      # [B,T]
        if self.cfg.deep_supervision:
            w = self._tick_weights(T, se.device)
            mse = (se * w.unsqueeze(0)).sum(dim=1).mean()
        else:
            mse = se[:, -1].mean()
        # 最终 tick 的归一化熵 (certs[:,0]=entropy), 鼓励高置信
        cert_pen = certs[:, 0, -1].mean()
        return mse, cert_pen

    @staticmethod
    def _is_parametric(ds) -> bool:
        return hasattr(ds, "Y") and hasattr(ds, "P")

    def _step_weights(self, H: int, device) -> torch.Tensor:
        # 多步权重归一为 1; v2.3 起可配曲线 (默认 front 与 2.2.1 逐元素相同)
        scheme = self.cfg.step_weight_scheme
        if scheme == "front":
            w = torch.linspace(1.0, 0.5, H, device=device)
        elif scheme == "uniform":
            w = torch.ones(H, device=device)
        elif scheme == "back":
            w = torch.linspace(0.5, 1.0, H, device=device)
        else:
            raise ValueError(
                f"未知 step_weight_scheme={scheme!r}, 可选 "
                f"{self.cfg.STEP_WEIGHT_SCHEMES}")
        return w / w.sum()

    @torch.no_grad()
    def evaluate(self, dataset) -> float:
        """单步(首步)测试 MSE; 兼容 v2.0 DynamicsDataset 与 v2.1 参数化数据集。"""
        self.model.eval()
        errs, n = 0.0, 0
        parametric = self._is_parametric(dataset)
        for batch in dataset.batches(self.cfg.batch_size, shuffle=False):
            if parametric:
                xb, pb, yb = batch
                pred = self.model.predict_next(xb, scene_params=pb)
                target = yb[:, 0, :]
            else:
                xb, target = batch
                pred = self.model.predict_next(xb)
            errs += ((pred - target) ** 2).sum().item()
            n += target.numel()
        return errs / max(n, 1)

    def _parametric_loss(self, xb, pb, yb, dt, epoch: int = 0):
        """
        多步 MSE + 运动学一致性 + 置信度正则, v2.2 支持 Scheduled Sampling。
        维护滚动窗口: 第 h 步预测后, 以概率 p_ss(epoch) 把模型自身上一步预测
        (截断梯度) 拼回窗口, 否则拼真值 (teacher-forcing)。ss_max=0 时与 2.1 等价。
        """
        H = yb.size(1)
        w = self._step_weights(H, xb.device)
        p_ss = self.cfg.ss_prob(epoch)
        ctx = None
        if self.model.scene_encoder is not None:
            ctx = self.model.scene_encoder(pb)
        mse_total = xb.new_zeros(())
        phys_total = xb.new_zeros(())
        cert_pen = xb.new_zeros(())
        hyb_total = xb.new_zeros(())
        window = xb
        use_hybrid = self.cfg.hybrid_weight > 0 and self.model.hybrid is not None
        for h in range(H):
            _, certs, last, _ = self._forward_cell(window, ctx)
            mse_total = mse_total + w[h] * ((last - yb[:, h, :]) ** 2).mean()
            if use_hybrid:
                # v2.6.0: 一阶欧拉骨架 + 学习残差, 对 hybrid 输出加权 MSE
                last_state = window[:, -1, :]
                euler = self.model.hybrid.euler_prediction(last_state, dt)
                inp = torch.cat([last, euler, last_state], dim=-1)
                hyb_out = euler + self.model.hybrid.mlp(inp)
                hyb_total = (hyb_total
                             + w[h] * ((hyb_out - yb[:, h, :]) ** 2).mean())
            if self.cfg.phys_weight > 0:
                prev = window[:, -1, :]   # 用上一帧速度做一阶欧拉诊断
                phys_total = (phys_total
                              + w[h] * kinematic_residual(last, prev, dt).mean())
            cert_pen = certs[:, 0, -1].mean()
            if h + 1 < H:
                teacher = yb[:, h, :]
                if p_ss > 0.0:
                    keep = (torch.rand(window.size(0), 1, device=window.device)
                            >= p_ss).to(teacher.dtype)  # 1=真值, 0=自身预测
                    feed = keep * teacher + (1.0 - keep) * last.detach()
                else:
                    feed = teacher
                window = torch.cat([window[:, 1:, :], feed.unsqueeze(1)], dim=1)
        # v2.6.0: 保持 3 元组返回签名 (旧测试/旧调用方逐位兼容), hybrid 损失经属性透出。
        self._last_hyb_loss = hyb_total
        return mse_total, phys_total, cert_pen

    def train(self, train_ds, eval_ds=None, verbose: bool = False) -> TrainHistory:
        set_seed(self.cfg.seed)
        hist = TrainHistory()
        parametric = self._is_parametric(train_ds)
        dt = getattr(train_ds, "dt", 0.5)
        # v2.2 早停状态 (仅在给 eval_ds 且 patience 非空时启用, 默认关闭=同 2.1)
        use_es = self.cfg.patience is not None and eval_ds is not None
        best_state, bad = None, 0
        for ep in range(self.cfg.epochs):
            self.model.train()
            ep_loss, ep_cert, ep_phys, nb = 0.0, 0.0, 0.0, 0
            hist.ss_prob.append(self.cfg.ss_prob(ep))
            for batch in train_ds.batches(self.cfg.batch_size, shuffle=True,
                                          seed=self.cfg.seed + ep):
                self.opt.zero_grad()
                if parametric:
                    xb, pb, yb = batch
                    xb = noise_augment(xb, self.cfg.noise_sigma)
                    mse, phys, cert_pen = self._parametric_loss(
                        xb, pb, yb, dt, epoch=ep)
                    loss = mse + self.cfg.phys_weight * phys \
                        + self.cfg.cert_weight * cert_pen \
                        + self.cfg.hybrid_weight * getattr(self, "_last_hyb_loss", 0.0)
                    ep_phys += float(phys.detach())
                else:
                    xb, yb = batch
                    xb = noise_augment(xb, self.cfg.noise_sigma)
                    decoded, certs, _, _ = self.model(xb)
                    mse, cert_pen = self._loss(decoded, certs, yb)
                    loss = mse + self.cfg.cert_weight * cert_pen
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.opt.step()
                ep_loss += float(mse.detach())
                ep_cert += (1.0 - float(cert_pen.detach()))
                nb += 1
            hist.train_loss.append(ep_loss / max(nb, 1))
            hist.cert.append(ep_cert / max(nb, 1))
            hist.phys_violation.append(ep_phys / max(nb, 1))
            if eval_ds is not None:
                cur = self.evaluate(eval_ds)
                hist.eval_mse.append(cur)
                if use_es:
                    if hist.best_eval is None or \
                            hist.best_eval - cur > self.cfg.min_delta:
                        hist.best_eval, hist.best_epoch, bad = cur, ep, 0
                        best_state = {k: v.detach().clone()
                                      for k, v in self.model.state_dict().items()}
                    else:
                        bad += 1
            if verbose and (ep % 10 == 0 or ep == self.cfg.epochs - 1):
                ev = hist.eval_mse[-1] if hist.eval_mse else float("nan")
                # verbose 训练进度仍走 stdout (既有契约 test_v22_robustness 断言);
                # 运行期诊断日志统一走 logger -> stderr, 二者职责分离。
                print(f"epoch {ep:3d} train_mse={hist.train_loss[-1]:.5f} "
                      f"eval_mse={ev:.5f} cert={hist.cert[-1]:.3f} "
                      f"ss={hist.ss_prob[-1]:.2f} "
                      f"phys_resid={hist.phys_violation[-1]:.4f}")
            if use_es and bad > self.cfg.patience:
                hist.stopped_early = True
                break
        # 早停启用时恢复最佳权重; 否则保持末轮 (同 2.1)
        if use_es and best_state is not None:
            self.model.load_state_dict(best_state)
        return hist


# ---------------- v3.3.0.dev5 推理时激活 FP16 压缩缓存 (opt-in, 默认关) ---------------- #
class ActivationFp16Cache:
    """
    推理时中间激活 FP16 压缩缓存 (analogy, not reproduction):
    对 obs_encoder 输出的中间嵌入按输入哈希缓存, 缓存以 float16 存储 (内存减半),
    命中时直接取回并上采样回 float32, 跳过重复编码。纯推理外挂, 不改模型权重,
    不调用时旧路径逐位不变。
    """

    def __init__(self) -> None:
        self._store: dict = {}
        self.hits = 0
        self.misses = 0

    @torch.no_grad()
    def encode_cached(self, model, raw_seq: torch.Tensor) -> torch.Tensor:
        h = hash(raw_seq.detach().cpu().numpy().tobytes())
        if h in self._store:
            self.hits += 1
            return self._store[h].float()
        self.misses += 1
        enc = model.obs_encoder(raw_seq)
        self._store[h] = enc.detach().half()     # FP16 压缩存储
        return enc

    def memory_bytes_est(self) -> int:
        """缓存内 FP16 张量的近似字节数 (不含 dict 开销)。"""
        return sum(t.numel() * 2 for t in self._store.values())

    def clear(self) -> None:
        self._store.clear()
        self.hits = self.misses = 0
