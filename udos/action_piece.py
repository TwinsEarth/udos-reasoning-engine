"""
ActionPiece 离散动作 token 化 (v3.1.0)
==========================================
analogy, not reproduction —— 受 PhysBrain "ActionPiece" 把连续动作量化为离散 token
序列思想启发的**轻量化类比实现**, 非复现:

UDOS 是 CPU-only、~52k 参数的合成参数化动力学小模型。这里在**合成动作向量**上
用 k-means 码本把连续动作量化为离散 token, 仅验证以下工程事实:

    1. 码本学习收敛 (k-means++ 初始化, 确定性 seed);
    2. encode(连续动作 -> token id) / decode(token id -> 连续动作) 往返有界误差;
    3. token 覆盖率统计 (哪些码本项被实际用到);
    4. 空 / 非法输入守卫;
    5. 码本 state_dict 保存 / 加载。

设计纪律 (与全工程一致):
    * **纯前向、确定性、不修改主模型权重** (推理时外挂; 主 CTM 仍为 52191 参数);
    * k-means 用纯 torch 实现, 不引入 sklearn / numpy 以外的重依赖 (numpy 随 torch 已装);
    * 码本仅在合成动作向量上拟合, 不碰真机动作 / 视频 / VLM;
    * 第二引擎一律称 GPM。
"""



from __future__ import annotations

import logging

logger = logging.getLogger("udos.action_piece")


import json
from typing import Any, Dict, Optional, Sequence

import torch


# --------------------------------------------------------------------------- #
# k-means 工具 (纯 torch, CPU-only, 确定性)
# --------------------------------------------------------------------------- #
def _as_action_matrix(x: torch.Tensor, action_dim: int) -> torch.Tensor:
    """把动作输入规整为 [N, action_dim] float32; 空/非法显式 ValueError。"""
    t = torch.as_tensor(x, dtype=torch.float32)
    if t.numel() == 0:
        raise ValueError("动作输入为空, 无法量化")
    if t.dim() == 1:
        t = t.unsqueeze(0)
    if t.dim() != 2:
        raise ValueError(f"动作需为 [action_dim] 或 [N, action_dim], 收到 {tuple(t.shape)}")
    if t.size(1) != action_dim:
        raise ValueError(
            f"动作维度 {t.size(1)} 与 tokenizer action_dim {action_dim} 不符")
    if not bool(torch.isfinite(t).all()):
        raise ValueError("动作含 NaN/inf 非有限值")
    return t


def _kmeans_pp_init(x: torch.Tensor, k: int, generator: torch.Generator
                    ) -> torch.Tensor:
    """k-means++ 初始化: 首个中心均匀抽取, 后续按到最近中心距离平方概率抽取。

    确定性: 给定 generator 种子, 结果可复现。k <= N 时不重复抽样。
    """
    n = x.size(0)
    if k > n:
        raise ValueError(f"码本大小 k={k} 不能大于样本数 N={n}")
    idx = torch.randint(0, n, (1,), generator=generator).item()
    centers = [x[idx]]
    # 到最近中心的平方距离
    d2 = ((x - centers[0]) ** 2).sum(dim=1)
    for _ in range(1, k):
        probs = d2 / d2.sum().clamp_min(1e-12)
        choice = torch.multinomial(probs, 1, generator=generator).item()
        centers.append(x[choice])
        new_d2 = ((x - centers[-1]) ** 2).sum(dim=1)
        d2 = torch.minimum(d2, new_d2)
    return torch.stack(centers, dim=0)


def _uniform_grid_init(x: torch.Tensor, k: int, action_dim: int) -> torch.Tensor:
    """均匀网格初始化: 每维在数据 min..max 上线性等分 (1-D 网格后逐维展开)。

    当 k 不是 action_dim 的完美幂时, 用最近整数网格边长并截断到 k 个中心。
    """
    lo = x.min(dim=0).values
    hi = x.max(dim=0).values
    # 取每维网格边长 g, 使 g**action_dim 接近 k (g>=1)
    g = max(1, int(round(k ** (1.0 / max(1, action_dim)))))
    lin = [torch.linspace(float(lo[d]), float(hi[d]), g)
           for d in range(action_dim)]
    mesh = torch.meshgrid(*lin, indexing="ij")
    grid = torch.stack([m.reshape(-1) for m in mesh], dim=1).float()
    if grid.size(0) >= k:
        sel = torch.linspace(0, grid.size(0) - 1, k).round().long()
        grid = grid[sel]
    else:  # 网格不够 k 个: 用首行复制补足 (罕见)
        pad = grid[0:1].repeat(k - grid.size(0), 1)
        grid = torch.cat([grid, pad], dim=0)
    return grid


# --------------------------------------------------------------------------- #
# ActionPieceTokenizer (node 31 / v3.1.0)
# --------------------------------------------------------------------------- #
class ActionPieceTokenizer:
    """k-means 码本把连续动作向量量化为离散 token。

    Parameters
    ----------
    action_dim:
        动作向量维度 (默认 6, 与 RAW_DIM 位/速向量同口径)。
    codebook_size:
        码本项数 K (>=1; =1 时退化为全局均值, 见 node 40 边界)。
    init:
        码本初始化策略: "kmeans++" (默认) / "random" / "uniform_grid"。
    seed:
        确定性随机种子 (k-means++ / random 抽样用)。
    """

    INIT_STRATEGIES = ("kmeans++", "random", "uniform_grid")

    def __init__(self, action_dim: int = 6, codebook_size: int = 16,
                 init: str = "kmeans++", seed: int = 42) -> None:
        if action_dim < 1:
            raise ValueError("action_dim 必须 >= 1")
        if codebook_size < 1:
            raise ValueError("codebook_size 必须 >= 1")
        if init not in self.INIT_STRATEGIES:
            raise ValueError(
                f"init 需为 {self.INIT_STRATEGIES}, 收到 {init!r}")
        self.action_dim = int(action_dim)
        self.codebook_size = int(codebook_size)
        self.init = str(init)
        self.seed = int(seed)
        self.centroids_: Optional[torch.Tensor] = None  # [K, action_dim]
        self.usage_: Optional[torch.Tensor] = None        # [K] 计数
        self.inertia_history_: list = []
        self.fitted_ = False

    # ------------------------------------------------------------------ #
    # 码本学习
    # ------------------------------------------------------------------ #
    def _init_centers(self, x: torch.Tensor) -> torch.Tensor:
        g = torch.Generator().manual_seed(self.seed)
        k = self.codebook_size
        if self.init == "kmeans++":
            return _kmeans_pp_init(x, k, g)
        if self.init == "random":
            perm = torch.randperm(x.size(0), generator=g)
            sel = perm[:k]
            return x[sel].clone()
        return _uniform_grid_init(x, k, self.action_dim)

    def fit(self, actions: torch.Tensor, n_iter: int = 30,
            tol: float = 1e-7) -> "ActionPieceTokenizer":
        """在合成动作矩阵 [N, action_dim] 上拟合 k-means 码本。

        迭代直到 inertia 变化 < tol 或达到 n_iter; 记录 inertia_history_ 供收敛验证。
        """
        x = _as_action_matrix(actions, self.action_dim)
        k = self.codebook_size
        if k > x.size(0):
            raise ValueError(
                f"码本大小 {k} 不能大于训练动作数 {x.size(0)}")
        centers = self._init_centers(x)
        inertia_hist: list = []
        for _ in range(int(n_iter)):
            # 分配: 每样本到最近中心的平方距离
            d2 = torch.cdist(x, centers) ** 2          # [N, K]
            assign = d2.argmin(dim=1)                    # [N]
            inertia = float(d2.gather(1, assign.unsqueeze(1)).mean())
            inertia_hist.append(round(inertia, 10))
            # 更新: 每簇取均值; 空簇重初始化到随机点 (确定性 seed 派生)
            new_centers = centers.clone()
            for j in range(k):
                mask = assign == j
                if bool(mask.any()):
                    new_centers[j] = x[mask].mean(dim=0)
                else:
                    g = torch.Generator().manual_seed(self.seed + j + 1)
                    new_centers[j] = x[
                        torch.randint(0, x.size(0), (1,), generator=g)].squeeze(0)
            shift = float((new_centers - centers).abs().max())
            centers = new_centers
            if len(inertia_hist) >= 2 and abs(inertia_hist[-2] - inertia_hist[-1]) < tol:
                break
        self.centroids_ = centers.detach().clone()
        self.inertia_history_ = inertia_hist
        self.usage_ = torch.zeros(k, dtype=torch.long)
        self.fitted_ = True
        return self

    # ------------------------------------------------------------------ #
    # 量化 / 反量化
    # ------------------------------------------------------------------ #
    def _require_fit(self) -> None:
        if not self.fitted_ or self.centroids_ is None:
            raise RuntimeError("tokenizer 未拟合, 请先 fit()")

    def encode(self, actions: torch.Tensor) -> torch.Tensor:
        """连续动作 [N, action_dim] -> token id (long tensor [N])。"""
        self._require_fit()
        x = _as_action_matrix(actions, self.action_dim)
        d2 = torch.cdist(x, self.centroids_) ** 2
        ids = d2.argmin(dim=1).long()
        # 更新用法统计 (纯只读外挂下的记账)
        if self.usage_ is not None:
            for j in ids.tolist():
                self.usage_[j] += 1
        return ids

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """token id [N] -> 连续动作 [N, action_dim] (最近码本项)。"""
        self._require_fit()
        t = torch.as_tensor(tokens, dtype=torch.long).reshape(-1)
        if t.numel() == 0:
            raise ValueError("token 序列为空, 无法反量化")
        if bool((t < 0).any()) or bool((t >= self.codebook_size).any()):
            raise ValueError(
                f"token id 须在 [0,{self.codebook_size}) 内, 收到 {t.tolist()}")
        return self.centroids_[t].clone()

    # ------------------------------------------------------------------ #
    # 质量 / 统计
    # ------------------------------------------------------------------ #
    def roundtrip_error(self, actions: torch.Tensor) -> float:
        """量化往返 MSE: mean((a - decode(encode(a)))^2)。"""
        x = _as_action_matrix(actions, self.action_dim)
        ids = self.encode(x)
        recon = self.decode(ids)
        return float(((x - recon) ** 2).mean())

    def token_coverage(self, tokens: Optional[torch.Tensor] = None
                       ) -> Dict[str, float]:
        """token 覆盖率: 每个 token 被使用的频率 (基于 encode 记账或传入序列)。

        返回 {token_id: fraction}; 未用到的 token 频率为 0。
        """
        self._require_fit()
        if tokens is not None:
            t = torch.as_tensor(tokens, dtype=torch.long).reshape(-1)
            counts = torch.bincount(t, minlength=self.codebook_size).float()
        elif self.usage_ is not None:
            counts = self.usage_.float()
        else:
            counts = torch.zeros(self.codebook_size)
        total = float(counts.sum())
        if total <= 0:
            return {str(j): 0.0 for j in range(self.codebook_size)}
        return {str(j): round(float(counts[j] / total), 6)
                for j in range(self.codebook_size)}

    def utilization(self) -> float:
        """码本利用率: 非空码本项占比 [0,1]。"""
        self._require_fit()
        if self.usage_ is None or float(self.usage_.sum()) == 0:
            return 0.0
        return float((self.usage_ > 0).float().mean())

    # ------------------------------------------------------------------ #
    # 序列化
    # ------------------------------------------------------------------ #
    def state_dict(self) -> Dict[str, Any]:
        return {
            "kind": "ActionPieceTokenizer",
            "action_dim": self.action_dim,
            "codebook_size": self.codebook_size,
            "init": self.init,
            "seed": self.seed,
            "centroids": (self.centroids_.tolist()
                          if self.centroids_ is not None else None),
            "usage": (self.usage_.tolist() if self.usage_ is not None else None),
            "inertia_history": self.inertia_history_,
            "fitted": self.fitted_,
        }

    def load_state_dict(self, d: Dict[str, Any]) -> "ActionPieceTokenizer":
        if d.get("kind", "ActionPieceTokenizer") != "ActionPieceTokenizer":
            raise ValueError(f"非法 tokenizer state_dict: {d!r}")
        self.action_dim = int(d["action_dim"])
        self.codebook_size = int(d["codebook_size"])
        self.init = str(d.get("init", "kmeans++"))
        self.seed = int(d.get("seed", 42))
        c = d.get("centroids")
        self.centroids_ = (torch.tensor(c, dtype=torch.float32)
                            if c is not None else None)
        u = d.get("usage")
        self.usage_ = (torch.tensor(u, dtype=torch.long) if u is not None
                       else None)
        self.inertia_history_ = list(d.get("inertia_history", []))
        self.fitted_ = bool(d.get("fitted", self.centroids_ is not None))
        return self

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.state_dict(), f)

    @classmethod
    def load(cls, path: str) -> "ActionPieceTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            return cls().load_state_dict(json.load(f))


# --------------------------------------------------------------------------- #
# 合成 token 序列 (Markov 链, 供自回归 next-token 验证)
# --------------------------------------------------------------------------- #
def markov_token_sequences(n_seqs: int = 200, seq_len: int = 12,
                           vocab_size: int = 8, step: int = 1,
                           stickiness: float = 0.9, seed: int = 42):
    """生成 "准周期" token 序列: next ≈ (prev + step) mod vocab 以 stickiness 概率。

    确定性 (给定 seed); 返回 list[LongTensor], 每条长 seq_len。
    n-gram 模型应能学到该结构 (精度 >> 随机 1/vocab)。
    """
    if not (0.0 <= stickiness <= 1.0):
        raise ValueError("stickiness 需在 [0,1]")
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n_seqs):
        toks = [int(torch.randint(0, vocab_size, (1,), generator=g).item())]
        for _ in range(seq_len - 1):
            if torch.rand((1,), generator=g).item() < stickiness:
                nxt = (toks[-1] + step) % vocab_size
            else:
                nxt = int(torch.randint(0, vocab_size, (1,), generator=g).item())
            toks.append(nxt)
        out.append(torch.tensor(toks, dtype=torch.long))
    return out


# --------------------------------------------------------------------------- #
# TokenizedActionPredictor (node 33 / v3.1.0.dev2): n-gram next-token
# --------------------------------------------------------------------------- #
class TokenizedActionPredictor:
    """给定历史动作 token 序列, n-gram 自回归预测下一个 token。

    纯计数模型 (无权重、确定性、CPU-only); 短历史时向低阶回退。
    **推理外挂, 不替换主 CTM**; token -> 连续动作解码走传入的 ActionPieceTokenizer。

    Parameters
    ----------
    vocab_size: 词表大小 (= codebook_size)。
    order: n-gram 阶数 (>=1); 历史不足时回退到可用低阶。
    """

    def __init__(self, vocab_size: int, order: int = 2) -> None:
        if vocab_size < 1:
            raise ValueError("vocab_size 必须 >= 1")
        if order < 1:
            raise ValueError("order 必须 >= 1")
        self.vocab_size = int(vocab_size)
        self.order = int(order)
        # counts_: dict[tuple(context_tokens), FloatTensor[vocab]]
        self.counts_: Dict[tuple, torch.Tensor] = {}
        self.unigram_: Optional[torch.Tensor] = None
        self.fitted_ = False

    def fit(self, sequences: Sequence[torch.Tensor]) -> "TokenizedActionPredictor":
        """在 token 序列列表上统计 (context -> next) 共现。"""
        seqs = [torch.as_tensor(s, dtype=torch.long).reshape(-1) for s in sequences]
        if not seqs:
            raise ValueError("序列列表为空, 无法拟合")
        counts: Dict[tuple, torch.Tensor] = {}
        uni = torch.zeros(self.vocab_size)
        for s in seqs:
            if bool((s < 0).any()) or bool((s >= self.vocab_size).any()):
                raise ValueError("序列含越界 token id")
            uni += torch.bincount(s, minlength=self.vocab_size).float()
            for t in range(1, s.numel()):
                ctx = tuple(s[max(0, t - self.order):t].tolist())
                bucket = counts.setdefault(
                    ctx, torch.zeros(self.vocab_size))
                bucket[int(s[t])] += 1.0
        self.counts_ = counts
        self.unigram_ = uni
        self.fitted_ = True
        return self

    def _require_fit(self) -> None:
        if not self.fitted_:
            raise RuntimeError("predictor 未拟合, 请先 fit()")

    def predict_logits(self, history: torch.Tensor) -> torch.Tensor:
        """历史 token [L] -> 下一个 token 的 logits [vocab] (计数为对数概率代理)。"""
        self._require_fit()
        h = torch.as_tensor(history, dtype=torch.long).reshape(-1)
        if h.numel() == 0:
            raise ValueError("历史 token 序列为空")
        if bool((h < 0).any()) or bool((h >= self.vocab_size).any()):
            raise ValueError("历史含越界 token id")
        # 从最长可用 context 逐步回退到 unigram
        for o in range(min(self.order, h.numel()), 0, -1):
            ctx = tuple(h[-o:].tolist())
            if ctx in self.counts_:
                return self.counts_[ctx].clone()
        return self.unigram_.clone() if self.unigram_ is not None else \
            torch.ones(self.vocab_size)

    def predict_next(self, history: torch.Tensor) -> int:
        """历史 token -> argmax 下一个 token id。"""
        return int(self.predict_logits(history).argmax().item())

    @torch.no_grad()
    def generate(self, seed_tokens: torch.Tensor, n_steps: int) -> torch.Tensor:
        """自回归生成 n_steps 个新 token (给定种子历史)。返回 [len(seed)+n_steps]。"""
        self._require_fit()
        if n_steps < 0:
            raise ValueError("n_steps 必须 >= 0")
        out = torch.as_tensor(seed_tokens, dtype=torch.long).reshape(-1).tolist()
        if not out:
            raise ValueError("seed_tokens 不能为空")
        for _ in range(n_steps):
            nxt = self.predict_next(torch.tensor(out, dtype=torch.long))
            out.append(nxt)
        return torch.tensor(out, dtype=torch.long)

    def heldout_accuracy(self, sequences: Sequence[torch.Tensor]) -> float:
        """在序列上前一步测 next-token 命中率 (argmax)。"""
        self._require_fit()
        correct = total = 0
        for s in sequences:
            s = torch.as_tensor(s, dtype=torch.long).reshape(-1)
            for t in range(1, s.numel()):
                pred = self.predict_next(s[:t])
                correct += int(pred == int(s[t]))
                total += 1
        return float(correct) / max(1, total)

    @staticmethod
    def decode_to_actions(tokens: torch.Tensor,
                          tokenizer: "ActionPieceTokenizer") -> torch.Tensor:
        """token 序列 -> 连续动作 [N, action_dim] (委托 tokenizer.decode)。"""
        return tokenizer.decode(torch.as_tensor(tokens, dtype=torch.long).reshape(-1))


# --------------------------------------------------------------------------- #
# TokenActionDecoder (node 34 / v3.1.0.dev3): token 序列 -> 连续动作 + 平滑
# --------------------------------------------------------------------------- #
class TokenActionDecoder:
    """token 序列 -> 连续动作, 支持线性插值平滑与关节限位后处理。

    * ``hard_decode``: 逐 token 取码本中心 (分段常数, 步间可能跳变);
    * ``smooth_decode``: 在相邻 token 解码之间做线性插值, 步间跳变更小 (更连续);
    * 输出一律经关节限位 clamp (若提供 joint_limits), 不可越界。

    Parameters
    ----------
    tokenizer:
        已拟合的 ActionPieceTokenizer (提供 token -> 连续动作中心)。
    joint_limits:
        可选 [action_dim, 2] 的 (lo, hi); 提供后输出严格落在限位内。
    """

    def __init__(self, tokenizer: "ActionPieceTokenizer",
                 joint_limits: Optional[torch.Tensor] = None) -> None:
        if not getattr(tokenizer, "fitted_", False):
            raise ValueError("tokenizer 必须先 fit()")
        self.tokenizer = tokenizer
        if joint_limits is not None:
            lim = torch.as_tensor(joint_limits, dtype=torch.float32)
            if lim.dim() != 2 or lim.size(1) != 2:
                raise ValueError("joint_limits 需为 [action_dim, 2]")
            if bool((lim[:, 0] > lim[:, 1]).any()):
                raise ValueError("joint_limits lo > hi")
            self.low: Optional[torch.Tensor] = lim[:, 0]
            self.high: Optional[torch.Tensor] = lim[:, 1]
        else:
            self.low = self.high = None

    def _clamp(self, a: torch.Tensor) -> torch.Tensor:
        if self.low is None:
            return a
        return torch.maximum(torch.minimum(a, self.high), self.low)

    def hard_decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """token 序列 -> 分段常数连续动作 [N, action_dim] (经限位 clamp)。"""
        raw = self.tokenizer.decode(tokens)
        return self._clamp(raw)

    def smooth_decode(self, tokens: torch.Tensor,
                      interp_steps: int = 3) -> torch.Tensor:
        """相邻 token 解码间线性插值平滑, 返回更密的轨迹 [M, action_dim]。

        interp_steps: 每对相邻 token 之间插入的段数 (>=1); =1 等价硬解码。
        输出端点与硬解码一致, 但步间最大跳变 = 硬解码跳变 / interp_steps。
        """
        if interp_steps < 1:
            raise ValueError("interp_steps 必须 >= 1")
        a = self.hard_decode(tokens)          # [N, D]
        if a.size(0) < 2 or interp_steps == 1:
            return a
        outs = [a[0:1]]
        for i in range(a.size(0) - 1):
            for s in range(1, interp_steps + 1):
                alpha = s / interp_steps
                outs.append((1 - alpha) * a[i:i+1] + alpha * a[i+1:i+2])
        return torch.cat(outs, dim=0)

    @staticmethod
    def continuity(actions: torch.Tensor) -> float:
        """动作连续性: 相邻步 L2 跳变的均值 (越小越连续)。"""
        a = torch.as_tensor(actions, dtype=torch.float32)
        if a.size(0) < 2:
            return 0.0
        return float((a[1:] - a[:-1]).norm(dim=1).mean())

    def within_limits(self, actions: torch.Tensor) -> bool:
        """校验动作是否严格落在关节限位内。"""
        a = torch.as_tensor(actions, dtype=torch.float32)
        if self.low is None:
            return True
        return bool((a >= self.low - 1e-6).all() and (a <= self.high + 1e-6).all())


# --------------------------------------------------------------------------- #
# ActionPieceCodec (node 32 / v3.1.0.dev1): 多尺度粗+细两级码本
# --------------------------------------------------------------------------- #
def token_perplexity(tokens: torch.Tensor, n_vocab: int) -> float:
    """离散 token 分布困惑度 exp(熵); 均匀分布时 = n_vocab, 退化 = 1。"""
    t = torch.as_tensor(tokens, dtype=torch.long).reshape(-1)
    if t.numel() == 0:
        raise ValueError("token 序列为空, 无法计算困惑度")
    counts = torch.bincount(t, minlength=n_vocab).float()
    p = counts / counts.sum().clamp_min(1e-12)
    nz = p[p > 0]
    ent = float(-(nz * nz.log()).sum())
    return float(torch.tensor(ent).exp().item())


class ActionPieceCodec:
    """多尺度两级量化码本 (粗粒度 + 细粒度残差)。

    思路 (analogy, not reproduction): 先用粗码本把动作分到粗簇, 再用细码本量化
    "动作 - 粗簇中心" 残差, 两级联合解码 = 粗中心 + 残差中心, 误差应 <= 粗级单独误差。

    Parameters
    ----------
    action_dim / coarse_size / fine_size:
        动作维度与两级码本大小 (>=1)。
    init:
        两级共用初始化策略 "kmeans++" / "random" / "uniform_grid"。
    seed: 确定性种子。
    """

    def __init__(self, action_dim: int = 6, coarse_size: int = 8,
                 fine_size: int = 8, init: str = "kmeans++",
                 seed: int = 42) -> None:
        if coarse_size < 1 or fine_size < 1:
            raise ValueError("coarse_size / fine_size 必须 >= 1")
        self.action_dim = int(action_dim)
        self.coarse_size = int(coarse_size)
        self.fine_size = int(fine_size)
        self.init = init
        self.seed = int(seed)
        self.coarse_ = ActionPieceTokenizer(action_dim, coarse_size, init, seed)
        self.fine_ = ActionPieceTokenizer(action_dim, fine_size, init, seed + 1)
        self.fitted_ = False

    def fit(self, actions: torch.Tensor, n_iter: int = 30) -> "ActionPieceCodec":
        x = _as_action_matrix(actions, self.action_dim)
        if self.coarse_size > x.size(0):
            raise ValueError(f"coarse_size {self.coarse_size} > 样本 {x.size(0)}")
        if self.fine_size > x.size(0):
            raise ValueError(f"fine_size {self.fine_size} > 样本 {x.size(0)}")
        self.coarse_.fit(x, n_iter=n_iter)
        coarse_ids = self.coarse_.encode(x)
        coarse_recon = self.coarse_.decode(coarse_ids)
        residuals = x - coarse_recon
        self.fine_.fit(residuals, n_iter=n_iter)
        self.fitted_ = True
        return self

    def _require_fit(self) -> None:
        if not self.fitted_:
            raise RuntimeError("codec 未拟合, 请先 fit()")

    def encode(self, actions: torch.Tensor):
        """连续动作 -> (coarse_ids[N], fine_ids[N])。"""
        self._require_fit()
        x = _as_action_matrix(actions, self.action_dim)
        c_ids = self.coarse_.encode(x)
        residual = x - self.coarse_.decode(c_ids)
        f_ids = self.fine_.encode(residual)
        return c_ids, f_ids

    def decode(self, coarse_ids: torch.Tensor,
               fine_ids: torch.Tensor) -> torch.Tensor:
        """(coarse_ids, fine_ids) -> 连续动作 [N, action_dim]。"""
        self._require_fit()
        c = torch.as_tensor(coarse_ids, dtype=torch.long).reshape(-1)
        f = torch.as_tensor(fine_ids, dtype=torch.long).reshape(-1)
        if c.numel() != f.numel():
            raise ValueError("coarse/fine token 序列长度不一致")
        return self.coarse_.decode(c) + self.fine_.decode(f)

    def roundtrip_error(self, actions: torch.Tensor) -> float:
        x = _as_action_matrix(actions, self.action_dim)
        c, f = self.encode(x)
        return float(((x - self.decode(c, f)) ** 2).mean())

    def coarse_error(self, actions: torch.Tensor) -> float:
        """仅用粗级的往返误差 (用于证明两级更优)。"""
        x = _as_action_matrix(actions, self.action_dim)
        return self.coarse_.roundtrip_error(x)

    def quality(self, actions: torch.Tensor) -> Dict[str, float]:
        """码本质量: 两级往返 MSE / 粗级 MSE / 粗&细困惑度 / 粗&细利用率。"""
        c, f = self.encode(actions)
        return {
            "two_level_mse": round(self.roundtrip_error(actions), 6),
            "coarse_mse": round(self.coarse_error(actions), 6),
            "coarse_perplexity": round(token_perplexity(c, self.coarse_size), 3),
            "fine_perplexity": round(token_perplexity(f, self.fine_size), 3),
            "coarse_utilization": round(self.coarse_.utilization(), 4),
            "fine_utilization": round(self.fine_.utilization(), 4),
            "gain_vs_coarse": round(
                max(0.0, self.coarse_error(actions) - self.roundtrip_error(actions)), 6),
        }

    def state_dict(self) -> Dict[str, Any]:
        return {
            "kind": "ActionPieceCodec",
            "action_dim": self.action_dim,
            "coarse_size": self.coarse_size,
            "fine_size": self.fine_size,
            "init": self.init, "seed": self.seed,
            "coarse": self.coarse_.state_dict(),
            "fine": self.fine_.state_dict(),
            "fitted": self.fitted_,
        }

    def load_state_dict(self, d: Dict[str, Any]) -> "ActionPieceCodec":
        if d.get("kind", "ActionPieceCodec") != "ActionPieceCodec":
            raise ValueError(f"非法 codec state_dict: {d!r}")
        self.action_dim = int(d["action_dim"])
        self.coarse_size = int(d["coarse_size"])
        self.fine_size = int(d["fine_size"])
        self.init = str(d.get("init", "kmeans++"))
        self.seed = int(d.get("seed", 42))
        self.coarse_ = ActionPieceTokenizer().load_state_dict(d["coarse"])
        self.fine_ = ActionPieceTokenizer().load_state_dict(d["fine"])
        self.fitted_ = bool(d.get("fitted", True))
        return self

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.state_dict(), f)

    @classmethod
    def load(cls, path: str) -> "ActionPieceCodec":
        with open(path, "r", encoding="utf-8") as f:
            return cls().load_state_dict(json.load(f))


# --------------------------------------------------------------------------- #
# SequenceCurriculum (node 36 / v3.1.0.dev5): easy->hard 纯数据调度
# --------------------------------------------------------------------------- #
class SequenceCurriculum:
    """按轨迹复杂度 (默认长度) 排序训练样本, easy->hard 纯数据调度。

    **纯数据排序, 不改模型 / 不改正式件训练口径**; 只决定样本喂入顺序。
    """

    def __init__(self, sequences: Sequence[torch.Tensor],
                 complexity=None) -> None:
        self.sequences = [torch.as_tensor(s, dtype=torch.long).reshape(-1)
                          for s in sequences]
        if not self.sequences:
            raise ValueError("序列集为空, 无法建课程")
        self._complexity = complexity or (lambda s: int(s.numel()))

    def complexities(self) -> list:
        return [self._complexity(s) for s in self.sequences]

    def easy_to_hard(self) -> list:
        """按复杂度升序 (easy 在前, hard 在后)。"""
        order = sorted(range(len(self.sequences)),
                       key=lambda i: self._complexity(self.sequences[i]))
        return [self.sequences[i] for i in order]

    def random_order(self, seed: int = 0) -> list:
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(len(self.sequences), generator=g).tolist()
        return [self.sequences[i] for i in idx]

    def split_easy_hard(self, frac: float = 0.5) -> tuple:
        """按复杂度把样本切成 (easy, hard) 两段 (frac=easy 占比)。"""
        if not (0.0 < frac < 1.0):
            raise ValueError("frac 需在 (0,1)")
        ordered = self.easy_to_hard()
        k = max(1, int(round(len(ordered) * frac)))
        return ordered[:k], ordered[k:]

    @staticmethod
    def _tiny_next_token_model(vocab: int, hidden: int = 16):
        """最小 next-token 线性模型 (one-hot prev -> logits vocab)。"""
        return torch.nn.Sequential(
            torch.nn.Linear(vocab, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, vocab))

    @staticmethod
    def _train_epoch(model, sequences, vocab: int, lr: float,
                     g: torch.Generator):
        opt = torch.optim.SGD(model.parameters(), lr=lr)
        loss_sum = n = 0.0
        for s in sequences:
            for t in range(1, s.numel()):
                x = torch.nn.functional.one_hot(s[t - 1], vocab).float()
                y = s[t]
                logits = model(x)
                loss = torch.nn.functional.cross_entropy(
                    logits.unsqueeze(0), y.unsqueeze(0))
                opt.zero_grad(); loss.backward(); opt.step()
                loss_sum += float(loss.detach()); n += 1
        return loss_sum / max(1, n)

    @staticmethod
    def eval_accuracy(model, sequences, vocab: int) -> float:
        correct = total = 0
        for s in sequences:
            for t in range(1, s.numel()):
                x = torch.nn.functional.one_hot(s[t - 1], vocab).float()
                pred = int(model(x).argmax())
                correct += int(pred == int(s[t])); total += 1
        return correct / max(1, total)

    @classmethod
    def convergence_ab(cls, seed: int = 42, n_easy: int = 60,
                       n_hard: int = 60, epochs: int = 12,
                       vocab: int = 8):
        """课程 vs 随机顺序在合成 next-token 任务上的收敛对比。

        easy = 短(长 4)高 stickiness(0.95); hard = 长(长 12)低 stickiness(0.6)。
        课程: 先训 easy 再训 hard; 随机: 全部打乱同总步数。
        返回 {curriculum_final_acc, random_final_acc,
              curriculum_better, ...}。
        """
        easy = markov_token_sequences(n_seqs=n_easy, seq_len=4, vocab_size=vocab,
                                      stickiness=0.95, seed=seed)
        hard = markov_token_sequences(n_seqs=n_hard, seq_len=12, vocab_size=vocab,
                                     stickiness=0.6, seed=seed + 1)
        cur = cls(easy + hard)
        easy_set, hard_set = cur.split_easy_hard(0.5)

        # 课程顺序: easy -> hard
        torch.manual_seed(seed)
        m_cur = cls._tiny_next_token_model(vocab)
        per = max(1, epochs // 2)
        for _ in range(per):
            cls._train_epoch(m_cur, easy_set, vocab, 0.1,
                             torch.Generator().manual_seed(0))
        for _ in range(epochs - per):
            cls._train_epoch(m_cur, hard_set, vocab, 0.1,
                             torch.Generator().manual_seed(0))
        cur_acc = cls.eval_accuracy(m_cur, hard, vocab)

        # 随机顺序: 同样总 epoch, 全部样本打乱
        torch.manual_seed(seed)
        m_ran = cls._tiny_next_token_model(vocab)
        for _ in range(epochs):
            cls._train_epoch(m_ran, cur.random_order(seed=seed), vocab, 0.1,
                             torch.Generator().manual_seed(0))
        ran_acc = cls.eval_accuracy(m_ran, hard, vocab)
        return {
            "vocab": vocab, "epochs": epochs,
            "curriculum_hard_acc": round(cur_acc, 4),
            "random_hard_acc": round(ran_acc, 4),
            "curriculum_better": bool(cur_acc >= ran_acc - 1e-9),
        }


# --------------------------------------------------------------------------- #
# InContextActionPrompter (node 37 / v3.1.0.dev6): few-shot 上下文
# --------------------------------------------------------------------------- #
class InContextActionPrompter:
    """few-shot 示例轨迹 + 查询 -> 拼接上下文, 用 tokenized predictor 预测后续。

    analogy, not reproduction —— 受 PhysBrain 长上下文 / in-context learning 启发,
    在合成 token 序列上做**轻量类比**: 不预训练大模型, 而是把 few-shot 示例作为
    n-gram 的即时上下文 (fit 在示例上), 再对查询后缀预测下一个 token。

    Parameters
    ----------
    vocab_size / order: 词表与 n-gram 阶。
    """

    def __init__(self, vocab_size: int, order: int = 2) -> None:
        self.vocab_size = int(vocab_size)
        self.order = int(order)

    def build_context(self, examples: Sequence[torch.Tensor],
                      query: torch.Tensor) -> torch.Tensor:
        """拼接 few-shot 示例 + 查询为单一 token 序列 (上下文)。"""
        ex = [torch.as_tensor(e, dtype=torch.long).reshape(-1) for e in examples]
        q = torch.as_tensor(query, dtype=torch.long).reshape(-1)
        if q.numel() == 0:
            raise ValueError("查询序列为空")
        parts = ex + [q]
        return torch.cat(parts, dim=0)

    def predict_next(self, query: torch.Tensor,
                     examples: Sequence[torch.Tensor]) -> int:
        """在 few-shot 示例上即时建 n-gram, 对查询后缀预测下一个 token。"""
        if not examples:
            raise ValueError("few-shot 示例为空")
        # 在示例上即时拟合 (in-context, 不持久化权重)
        pred = TokenizedActionPredictor(self.vocab_size, self.order).fit(
            list(examples))
        return pred.predict_next(query)

    def few_shot_accuracy(self, n_shots: int, seed: int = 0,
                          n_queries: int = 40) -> float:
        """在合成 "周期+小噪" 任务上测 n-shot next-token 精度。

        任务: next ≈ (cur + 1) mod vocab。示例序列展示该规则; 查询为短前缀。
        n_shots 越多 -> n-gram 计数越完整 -> 精度越高。
        """
        if n_shots < 1:
            raise ValueError("n_shots 必须 >= 1")
        g = torch.Generator().manual_seed(seed)
        vocab = self.vocab_size
        # 示例池: 周期序列 (展示规则)
        bank = markov_token_sequences(n_seqs=20, seq_len=8, vocab_size=vocab,
                                      step=1, stickiness=0.95, seed=seed)
        correct = total = 0
        for i in range(n_queries):
            # 取 n_shots 个示例
            start = int(torch.randint(0, len(bank), (1,), generator=g))
            examples = [bank[(start + j) % len(bank)] for j in range(n_shots)]
            # 查询前缀 (4 token), 目标 = 按规则下一个
            cur = int(torch.randint(0, vocab, (1,), generator=g).item())
            qseq = [cur]
            for _ in range(3):
                qseq.append((qseq[-1] + 1) % vocab)
            query = torch.tensor(qseq, dtype=torch.long)
            target = (qseq[-1] + 1) % vocab
            pred = self.predict_next(query, examples)
            correct += int(pred == target)
            total += 1
        return correct / max(1, total)


__all__ = ["ActionPieceTokenizer", "ActionPieceCodec",
           "TokenizedActionPredictor", "markov_token_sequences",
           "TokenActionDecoder", "SequenceCurriculum",
           "InContextActionPrompter"]
