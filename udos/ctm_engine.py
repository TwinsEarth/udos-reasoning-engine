"""
CTM 物理推演引擎 (机制对齐 SakanaAI/continuous-thought-machines)
================================================================
对齐上游 models/ctm.py 的三大核心机制:

1. 内部时间轴 (internal recurrence):
   推理不是一次前向, 而是在 `iterations` 个内部 tick 上循环展开,
   每个 tick 都产出一版预测与 certainty —— "thought takes time"。

2. 神经元级时序模型 (Neuron-Level Models, NLMs):
   每个神经元维护长度 memory_length 的 pre-activation 历史 trace,
   并使用该神经元 **私有** 的权重处理自身历史 (上游 SuperLinear,
   分组大小=1; 这里用 einsum 的 per-neuron 权重等价实现)。

3. 神经同步作为表示 (synchronisation as representation):
   选取成对神经元, 用其激活时间序列的 (带指数衰减的) 累积点积作为
   表示向量; random-pairing 选点策略、alpha/beta 递推、
   r=exp(-decay) 衰减均与上游 compute_synchronisation 对齐。

另对齐: certainty = 1 - 归一化熵, 用于自适应计算时长 (early-stop)。

与上游差异: 上游面向图像 (ResNet backbone), 本实现为 backbone-free,
直接吃 PCE 物理 Token 嵌入序列 [B, S, d_input], 适配物理因果推演。
"""



from __future__ import annotations

import logging

logger = logging.getLogger("udos.ctm_engine")


import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalized_entropy(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """归一化熵 ∈ [0,1], 对齐上游 models/utils.compute_normalized_entropy。"""
    probs = F.softmax(logits, dim=dim)
    ent = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=dim)
    n = logits.size(dim)
    return ent / math.log(n)


class NeuronLevelModel(nn.Module):
    """
    神经元级时序模型 (对齐上游 get_neuron_level_models + SuperLinear)。

    每个神经元 d 拥有私有权重 W1[d]: (M -> 2H), W2[d]: (H -> 2),
    经两次 GLU 后输出该神经元的 post-activation。
    输入 trace: [B, D, M] (每个神经元各自的历史窗口), 输出 [B, D]。
    """

    def __init__(self, num_neurons: int, memory_length: int, hidden: int,
                 deep: bool = True):
        super().__init__()
        self.D = num_neurons
        self.M = memory_length
        self.deep = deep
        if deep:
            self.w1 = nn.Parameter(torch.empty(num_neurons, memory_length, 2 * hidden))
            self.b1 = nn.Parameter(torch.zeros(num_neurons, 2 * hidden))
            self.w2 = nn.Parameter(torch.empty(num_neurons, hidden, 2))
            self.b2 = nn.Parameter(torch.zeros(num_neurons, 2))
            nn.init.xavier_uniform_(self.w1)
            nn.init.xavier_uniform_(self.w2)
        else:
            self.w1 = nn.Parameter(torch.empty(num_neurons, memory_length, 2))
            self.b1 = nn.Parameter(torch.zeros(num_neurons, 2))
            nn.init.xavier_uniform_(self.w1)

    def forward(self, trace: torch.Tensor) -> torch.Tensor:
        # trace: [B, D, M]
        h = torch.einsum("bdm,dmo->bdo", trace, self.w1) + self.b1
        h = F.glu(h, dim=-1)  # [B, D, *]
        if self.deep:
            h = torch.einsum("bdh,dho->bdo", h, self.w2) + self.b2
            h = F.glu(h, dim=-1)  # [B, D, 1]
            h = h.squeeze(-1)
        else:
            h = h.squeeze(-1)
        return h  # [B, D]


class SynapseMixer(nn.Module):
    """突触混合层 (对齐上游 synapse_depth=1 的 synapses: Linear->GLU->LayerNorm)。"""

    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * d_model, 2 * d_model),
            nn.GLU(),
            nn.LayerNorm(d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class CTMConfig:
    iterations: int = 32           # T: 内部 tick 数
    d_model: int = 256             # D: 内部神经群体规模
    d_input: int = 256             # 物理 Token 嵌入维度
    heads: int = 4                 # 数据交互注意力头数
    n_synch_out: int = 64          # D_out: 输出同步神经元对数
    n_synch_action: int = 32       # D_action: 动作同步神经元对数
    memory_length: int = 16        # M: NLM 历史窗口
    nlm_hidden: int = 32
    deep_nlms: bool = True
    out_dims: int = 256            # 每 tick 预测维度
    dropout: float = 0.0
    # random-pairing 自配对数 (对齐上游 n_random_pairing_self)
    n_random_pairing_self: int = 4
    certainty_threshold: float = 0.95  # 自适应停止阈值; <=0 表示关闭
    # v2.0.0 GPM->CTM 场景条件化: 给定则建立场景投影, None 保持一代行为
    scene_dim: Optional[int] = None
    # ---- v3.3.0 架构精炼 (全部 opt-in, 默认全关与旧架构逐位等价) ----
    # residual:    在循环单元上对 activated 做残差连接 (new_act = f(prev) + prev)
    # act_norm:    在每次更新 activated 后追加 LayerNorm(d_model)
    # init_mode:   "legacy" 完全沿用既有初始化顺序 (逐位等价);
    #              "xavier"/"he" 在构造完成后对全部 nn.Linear 权重重初始化
    residual: bool = False
    act_norm: bool = False
    init_mode: str = "legacy"

    INIT_MODES = ("legacy", "xavier", "he")


class CTMPhysicsEngine(nn.Module):
    """
    CTM 物理推演引擎。

    forward(physical_tokens) ->
        predictions: [B, out_dims, T_used]  每个内部 tick 的预测
        certainties: [B, 2, T_used]         (归一化熵, 1-熵)
        sync_out:    [B, n_synch_out]       最终同步表示
        trace_info:  (可选) 各 tick 的同步/激活轨迹
    """

    def __init__(self, config: Optional[CTMConfig] = None, **overrides):
        super().__init__()
        cfg = config or CTMConfig()
        for k, v in overrides.items():
            setattr(cfg, k, v)
        self.cfg = cfg
        D, M = cfg.d_model, cfg.memory_length

        # --- 输入交互 (对齐 q_proj/kv_proj + MultiheadAttention) ---
        self.kv_proj = nn.Sequential(nn.Linear(cfg.d_input, cfg.d_input),
                                     nn.LayerNorm(cfg.d_input))
        self.q_proj = nn.Linear(cfg.n_synch_action, cfg.d_input)
        self.attention = nn.MultiheadAttention(
            cfg.d_input, cfg.heads, cfg.dropout, batch_first=True)
        # 注意力输出宽度 d_input -> 神经群体宽度 d_model, 保证与
        # activated_state 拼接后为 2*d_model (对齐上游 LazyLinear 自适应)
        self.attn_to_state = nn.Linear(cfg.d_input, cfg.d_model)

        # --- CTM 核心 ---
        self.synapses = SynapseMixer(D)
        self.trace_processor = NeuronLevelModel(
            D, M, cfg.nlm_hidden, deep=cfg.deep_nlms)

        # --- 起始状态 (可学习, 对齐 start_activated_state / start_trace) ---
        self.start_activated_state = nn.Parameter(
            torch.empty(D).uniform_(-1 / math.sqrt(D), 1 / math.sqrt(D)))
        self.start_trace = nn.Parameter(
            torch.empty(D, M).uniform_(-1 / math.sqrt(D + M), 1 / math.sqrt(D + M)))

        # --- 神经元配对 (random-pairing, 注册为 buffer 以随 state_dict 保存) ---
        g = torch.Generator().manual_seed(0)
        self._build_pairing("out", cfg.n_synch_out, D, cfg.n_random_pairing_self, g)
        self._build_pairing("action", cfg.n_synch_action, D, 0, g)
        # 可学习指数衰减 (对齐 decay_params_*, clamp 到 [0,15])
        self.decay_params_out = nn.Parameter(torch.zeros(cfg.n_synch_out))
        self.decay_params_action = nn.Parameter(torch.zeros(cfg.n_synch_action))

        # --- 输出投影 (从同步表示投影, 对齐 output_projector) ---
        self.output_projector = nn.Linear(cfg.n_synch_out, cfg.out_dims)

        # --- v2.0.0 场景条件化 (GPM 场景嵌入 -> 每个 tick 的状态偏置) ---
        # gate 零初始化: 开启条件化时初始等价于无条件, 不破坏一代已训练权重
        if cfg.scene_dim is not None:
            self.scene_proj = nn.Linear(cfg.scene_dim, D)
            self.scene_gate = nn.Parameter(torch.zeros(1))
        else:
            self.scene_proj = None
            self.scene_gate = None

        # --- v3.3.0 act_norm: 仅在 opt-in 时注册 LayerNorm (默认关 => 不增参数,
        # 旧 state_dict / checkpoint 逐位不变) ---
        self.act_norm = nn.LayerNorm(D) if cfg.act_norm else None

        # --- v3.3.0 可配置初始化: 仅 init_mode != "legacy" 时重初始化 Linear,
        # legacy 路径不触碰既有 RNG 消耗顺序, 与旧架构逐位等价 ---
        if cfg.init_mode not in CTMConfig.INIT_MODES:
            raise ValueError(
                f"未知 init_mode={cfg.init_mode!r}, 可选 {CTMConfig.INIT_MODES}")
        if cfg.init_mode != "legacy":
            self._apply_configurable_init(cfg.init_mode)

    def _apply_configurable_init(self, mode: str) -> None:
        """对全部 nn.Linear.weight 施加可配置初始化 (opt-in, legacy 不调用)。"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if mode == "xavier":
                    nn.init.xavier_uniform_(m.weight)
                elif mode == "he":
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _build_pairing(self, kind: str, n: int, D: int, n_self: int,
                       g: torch.Generator) -> None:
        left = torch.randint(0, D, (n,), generator=g)
        rest = torch.randint(0, D, (n - n_self,), generator=g)
        right = torch.cat([left[:n_self], rest]) if n_self > 0 else rest
        self.register_buffer(f"{kind}_left", left)
        self.register_buffer(f"{kind}_right", right)

    # ---------- 神经同步 (对齐 compute_synchronisation, random-pairing 分支) ----------
    def _synchronise(self, activated: torch.Tensor, alpha, beta, r,
                     kind: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        left = getattr(self, f"{kind}_left")
        right = getattr(self, f"{kind}_right")
        pairwise = activated[:, left] * activated[:, right]  # [B, n]
        if alpha is None:
            alpha = pairwise
            beta = torch.ones_like(pairwise)
        else:
            alpha = r * alpha + pairwise
            beta = r * beta + 1.0
        sync = alpha / torch.sqrt(beta)
        return sync, alpha, beta

    def reset(self) -> None:
        """无状态前向下无需清理; 接口保留以对齐上游会话重置语义。"""
        pass

    def forward(self, physical_tokens: torch.Tensor,
                track: bool = False,
                scene_context: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        assert physical_tokens.dim() == 3, "需要 [B, seq_len, d_input]"
        if scene_context is not None:
            assert self.scene_proj is not None, \
                "需在 CTMConfig 配置 scene_dim 才能传入 scene_context"
            scene_bias = torch.tanh(self.scene_proj(scene_context))  # [B,D]
        B, device = physical_tokens.size(0), physical_tokens.device
        cfg = self.cfg

        kv = self.kv_proj(physical_tokens)  # [B, S, d_input]

        activated = self.start_activated_state.unsqueeze(0).expand(B, -1).contiguous()
        trace = self.start_trace.unsqueeze(0).expand(B, -1, -1).contiguous().clone()

        # 衰减系数 r = exp(-decay), clamp 对齐上游
        self.decay_params_action.data.clamp_(0, 15)
        self.decay_params_out.data.clamp_(0, 15)
        r_act = torch.exp(-self.decay_params_action).unsqueeze(0).expand(B, -1)
        r_out = torch.exp(-self.decay_params_out).unsqueeze(0).expand(B, -1)

        alpha_a = beta_a = None
        sync_out, alpha_o, beta_o = self._synchronise(
            activated, None, None, r_out, "out")

        preds: List[torch.Tensor] = []
        certs: List[torch.Tensor] = []
        sync_hist: List[torch.Tensor] = []
        act_hist: List[torch.Tensor] = []
        stop_tick = cfg.iterations

        for tick in range(cfg.iterations):
            # 1) 动作同步 -> 与物理数据做注意力交互
            sync_act, alpha_a, beta_a = self._synchronise(
                activated, alpha_a, beta_a, r_act, "action")
            q = self.q_proj(sync_act).unsqueeze(1)  # [B,1,d_input]
            attn_out, _ = self.attention(q, kv, kv, need_weights=False)
            attn_out = self.attn_to_state(attn_out.squeeze(1))  # [B,d_model]

            # 2) 突触混合 (跨神经元通信) + GPM 场景条件偏置
            state = self.synapses(torch.cat([attn_out, activated], dim=-1))
            if scene_context is not None:
                state = state + self.scene_gate * scene_bias

            # 3) 历史窗口滚动 + 神经元级时序模型
            # v3.3.0: residual/act_norm 默认关, 守卫分支不执行 => 旧路径逐位等价
            prev_activated = activated
            trace = torch.cat([trace[:, :, 1:], state.unsqueeze(-1)], dim=-1)
            activated = self.trace_processor(trace)
            if cfg.residual:
                activated = activated + prev_activated
            if self.act_norm is not None:
                activated = self.act_norm(activated)

            # 4) 输出同步 -> 预测 + certainty
            sync_out, alpha_o, beta_o = self._synchronise(
                activated, alpha_o, beta_o, r_out, "out")
            pred = self.output_projector(sync_out)
            ent = normalized_entropy(pred, dim=-1)
            cert = torch.stack([ent, 1.0 - ent], dim=-1)  # [B,2]
            preds.append(pred)
            certs.append(cert)
            if track:
                sync_hist.append(sync_out.detach())
                act_hist.append(activated.detach())

            # 5) 自适应计算时长: 批次平均置信度达阈值即收敛
            if cfg.certainty_threshold > 0 and (1.0 - ent).mean().item() >= cfg.certainty_threshold:
                stop_tick = tick + 1
                break

        predictions = torch.stack(preds, dim=-1)   # [B, out, T_used]
        certainties = torch.stack(certs, dim=-1)   # [B, 2, T_used]
        info = {
            "ticks_used": stop_tick,
            "sync_history": torch.stack(sync_hist, dim=1) if track else None,
            "activation_history": torch.stack(act_hist, dim=1) if track else None,
        }
        return predictions, certainties, sync_out, info
