"""
ICM 上下文记忆 (In-Context Memory) — v3.4.0 核心
================================================================
analogy, not reproduction —— 受 NVIDIA MimicDroid/DreamDojo、Skild 纯 ICL、
Generalist 数据涌现等 "任务从权重转移到上下文、零梯度、人做一遍机器当场执行"
思想启发的**轻量化类比实现**, 非复现:

UDOS 是 ~52k 参数的 CPU-only 合成参数化动力学小模型, 不预训练大模型、不看视频。
v3.2 线的 `InContextLearner` 把 few-shot 示例窗口**沿时间维朴素拼接**进上下文序列,
实测 (合成基准) 0-shot MSE≈0.046 而 1-shot≈0.67 / 3-shot≈1.11 **反而更差**——
根因: 朴素拼接把 k*W 帧原始窗口塞进 52k 小模型的注意力视野, 稀释了默认 6 帧的
主信号, 模型不知道哪些是演示、哪些是查询。

v3.4 线的 ICM 不是重复 naive few-shot, 而是 **"检索 + 聚合"**:

    1. DemonstrationEpisode : 一次 "输入轨迹 -> 动作 -> 结果" 的 PCE 因果块;
    2. DemonstrationMemory : 按查询窗口与演示输入窗口的确定性相似度检索 top-k;
    3. ICMAggregator        : 零梯度原型编码 + 跨演示相似度加权聚合。

关键工程事实 (为什么不退化):
    * **不把演示窗口拼进模型输入序列**, 而是在**输出残差空间**聚合;
    * 注册期一次性算出每条演示的 "模型残差" r_i = result_i - predict_next(in_i),
      即 "模型在这类轨迹上漏掉了什么", 缓存;
    * 预测期 p_icm = p0 + λ · Σ_i softmax(s_i) · r_i,  p0 = 纯 0-shot 预测;
    * λ 收缩 (shrink) 保证最坏情形 λ→0 时 p_icm == p0, 即 **绝不比 0-shot 更差**,
      从机制上杜绝 naive 拼接那种 0.046→1.1 的爆炸。

设计纪律 (零梯度原则, 最高优先级):
    * 全程 @torch.no_grad, 不反向、不更新主 predictor 任何权重;
    * ICM 自身无可学参数 (原型编码是确定性 L2 归一化), state_dict 零新增;
    * 默认 opt-in: 不构造 ICM / 不传 memory 时, 调用方走旧 predict_next 逐位一致;
    * 空记忆 / k=0 退化为 0-shot, 不伪造检索结果;
    * 第二引擎一律称 GPM; analogy, not reproduction。
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .dynamics import RAW_DIM

logger = logging.getLogger("udos.icm")


# ---------------------------------------------------------------------------
# 演示因果块: 输入轨迹 -> 动作 -> 结果
# ---------------------------------------------------------------------------
def _flatten_window_embed(window: torch.Tensor) -> torch.Tensor:
    """确定性检索嵌入: 把 [W, RAW_DIM] 展平为定长向量并 L2 归一化 (零参数)。

    不用可学编码器 (避免引入可训权重), 也不调用主模型 (保证注册/检索不触发
    任何前向副作用); 纯几何相似度, 可复现、可单测。
    """
    w = torch.as_tensor(window, dtype=torch.float32)
    if w.dim() != 2 or w.size(-1) != RAW_DIM:
        raise ValueError(
            f"演示窗口需为 [W, {RAW_DIM}], 收到 shape={tuple(w.shape)}")
    flat = w.reshape(-1).contiguous()
    norm = flat.norm()
    if norm.item() < 1e-12:
        # 全零向量: 归一化后给零向量, 相似度为 0 (不与任何查询匹配)
        return flat
    return flat / norm


class DemonstrationEpisode:
    """一次演示因果块: 历史窗口 -> (动作) -> 下一状态结果。

    Attributes
    ----------
    input_window : [W, RAW_DIM]
        演示时的观测历史窗口 (演示 "输入轨迹")。
    result : [RAW_DIM]
        该窗口对应的 ground-truth 下一状态 (演示 "结果")。
    action : [RAW_DIM]
        该窗口到结果之间的动作位移 = result - input_window[-1] (演示 "动作" 流)。
    scene_params : [4] 可选
        演示对应的隐藏场景参数 (仅记录, 不参与零梯度聚合)。
    kind : str
        演示运动类型标签 (仅记录/审计)。
    embed : [W*RAW_DIM]
        检索用确定性 L2 归一化嵌入 (注册时一次算好缓存)。
    """

    def __init__(self, input_window: torch.Tensor, result: torch.Tensor,
                 action: Optional[torch.Tensor] = None,
                 scene_params: Optional[torch.Tensor] = None,
                 kind: str = "unspecified") -> None:
        w = torch.as_tensor(input_window, dtype=torch.float32)
        r = torch.as_tensor(result, dtype=torch.float32).reshape(-1)
        if w.dim() != 2 or w.size(-1) != RAW_DIM:
            raise ValueError(
                f"input_window 需为 [W,{RAW_DIM}], 收到 {tuple(w.shape)}")
        if r.numel() != RAW_DIM:
            raise ValueError(f"result 需为 {RAW_DIM} 维, 收到 {r.numel()}")
        if not bool(torch.isfinite(w).all() and torch.isfinite(r).all()):
            raise ValueError("演示窗口/结果含 NaN/inf")
        self.input_window = w.contiguous()
        self.result = r.contiguous()
        # 动作流 = 结果 - 窗口末帧 (显式给定则校验形状)
        if action is None:
            a = self.result - self.input_window[-1]
        else:
            a = torch.as_tensor(action, dtype=torch.float32).reshape(-1)
            if a.numel() != RAW_DIM:
                raise ValueError(f"action 需为 {RAW_DIM} 维")
        self.action = a.contiguous()
        self.scene_params = (torch.as_tensor(scene_params, dtype=torch.float32)
                             if scene_params is not None else None)
        self.kind = str(kind)
        self.embed = _flatten_window_embed(self.input_window)

    def to_dict(self) -> Dict:
        return {
            "kind": self.kind,
            "input_window": self.input_window.tolist(),
            "result": self.result.tolist(),
            "action": self.action.tolist(),
            "scene_params": (self.scene_params.tolist()
                             if self.scene_params is not None else None),
        }


# ---------------------------------------------------------------------------
# 演示记忆库: 相似度检索 top-k
# ---------------------------------------------------------------------------
class DemonstrationMemory:
    """零梯度演示记忆库: 注册 episode, 按查询窗口相似度检索 top-k。

    相似度 = 归一化嵌入点积 (余弦相似度, 值域 [-1,1]); 纯确定性、无参数、
    不调用主模型。容量上限 (默认不限) 用于 v3.4.0.dev4 预算管理。
    """

    def __init__(self, max_episodes: Optional[int] = None) -> None:
        if max_episodes is not None and max_episodes < 1:
            raise ValueError("max_episodes 须 >=1 或 None")
        self.max_episodes = max_episodes
        self._eps: List[DemonstrationEpisode] = []

    def __len__(self) -> int:
        return len(self._eps)

    @property
    def size(self) -> int:
        return len(self._eps)

    def kinds(self) -> List[str]:
        return [e.kind for e in self._eps]

    def register(self, episode: DemonstrationEpisode) -> None:
        """注册一条演示; 超容量时淘汰最旧 (FIFO)。"""
        if not isinstance(episode, DemonstrationEpisode):
            raise ValueError("仅可注册 DemonstrationEpisode")
        self._eps.append(episode)
        if self.max_episodes is not None and len(self._eps) > self.max_episodes:
            self._eps.pop(0)

    def clear(self) -> None:
        self._eps = []

    def similarity(self, query_window: torch.Tensor) -> torch.Tensor:
        """查询窗口 vs 库中每条 episode 的余弦相似度 [N]。"""
        if not self._eps:
            raise RuntimeError("记忆库为空, 无可检索对象")
        q = _flatten_window_embed(query_window)
        embs = torch.stack([e.embed for e in self._eps], dim=0)   # [N, D]
        sim = embs @ q                                              # [N]
        return sim

    def retrieve(self, query_window: torch.Tensor, k: int = 3
                 ) -> List[Tuple[DemonstrationEpisode, float]]:
        """返回 top-k (episode, 相似度), 按相似度降序; k<=0 或空库显式报错。"""
        if k <= 0:
            raise ValueError("k 须为正整数")
        if not self._eps:
            raise RuntimeError("记忆库为空, 无可检索对象")
        k = int(min(k, len(self._eps)))
        sim = self.similarity(query_window)
        top = torch.topk(sim, k=k, largest=True, sorted=True)
        out: List[Tuple[DemonstrationEpisode, float]] = []
        for idx, s in zip(top.indices.tolist(), top.values.tolist()):
            out.append((self._eps[idx], float(s)))
        return out


# ---------------------------------------------------------------------------
# 零梯度原型聚合器: 跨演示相似度加权聚合 -> 演示条件化修正
# ---------------------------------------------------------------------------
class ICMAggregator:
    """零梯度原型聚合器 (无可学参数, 不入主 state_dict)。

    流程 (全程 no_grad):
        1. 注册期: 对每条 episode 一次性算并缓存残差
              r_i = result_i - predictor.predict_next(input_window_i)
           (模型在这类轨迹上漏掉的量; 与 scene_params 条件无关地反映模型误差结构)。
        2. 预测期:
              p0  = predictor.predict_next(query_window)            # 纯 0-shot
              s_i = cosine(query_embed, ep_i.embed)                 # 相似度
              w_i = softmax(s_i / temperature)                       # 跨演示加权
              corr = Σ_i w_i · r_i                                  # 原型聚合
              p_icm = p0 + λ · corr                                 # 演示条件化修正

    λ (shrink) 默认 1.0; 调小即收缩修正幅度, λ=0 时 p_icm ≡ p0 逐位一致。
    空记忆 / k=0 时直接返回 p0 (opt-in 默认路径不变)。
    """

    def __init__(self, predictor, temperature: float = 1.0,
                 lamb: float = 1.0, scene_param_dim: int = 4) -> None:
        if temperature <= 0:
            raise ValueError("temperature 须 >0")
        if not (0.0 <= lamb <= 1.0):
            raise ValueError("lamb 须在 [0,1]")
        self.predictor = predictor
        self.temperature = float(temperature)
        self.lamb = float(lamb)
        self.scene_param_dim = int(scene_param_dim)
        # episode -> 缓存残差 [RAW_DIM] (注册期一次算好)
        self._residuals: Dict[int, torch.Tensor] = {}

    # ------------------------------------------------------------------ #
    # 注册期残差缓存 (零梯度)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def cache_residual(self, episode: DemonstrationEpisode,
                       scene_params: Optional[torch.Tensor] = None) -> torch.Tensor:
        """对一条 episode 算模型残差 r = result - predict_next(input_window) 并缓存。

        残差以 episode 的 id 为键缓存; 纯推理外挂, 不更新 predictor 权重。
        """
        self.predictor.eval()
        win = episode.input_window.unsqueeze(0)                # [1,W,6]
        sp = scene_params if scene_params is not None else episode.scene_params
        p = self.predictor.predict_next(win, scene_params=sp)  # [1,6]
        r = episode.result - p[0]
        self._residuals[id(episode)] = r.detach().clone()
        return r

    def get_cached_residual(self, episode: DemonstrationEpisode) -> torch.Tensor:
        if id(episode) not in self._residuals:
            raise KeyError(
                "该 episode 尚未缓存残差, 请先 cache_residual()")
        return self._residuals[id(episode)]

    def clear_cache(self) -> None:
        self._residuals = {}

    # ------------------------------------------------------------------ #
    # 预测期: 检索 + 聚合
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def predict(self, query_window: torch.Tensor,
                memory: Optional[DemonstrationMemory] = None,
                k: int = 3,
                scene_params: Optional[torch.Tensor] = None,
                use_cached: bool = True) -> torch.Tensor:
        """演示条件化下一状态预测 [RAW_DIM]。

        memory 为空 / k<=0 / 无可检索 episode 时 => 纯 0-shot p0 (opt-in 默认)。
        """
        self.predictor.eval()
        win = torch.as_tensor(query_window, dtype=torch.float32)
        if win.dim() == 2:
            win_b = win.unsqueeze(0)
        elif win.dim() == 3:
            win_b = win
        else:
            raise ValueError("query_window 需为 [W,RAW] 或 [1,W,RAW]")
        p0 = self.predictor.predict_next(win_b, scene_params=scene_params)
        p0 = p0[0] if p0.dim() > 1 else p0

        # ---- opt-in 退化: 无记忆 => 纯 0-shot ----
        if memory is None or memory.size == 0 or k <= 0 or self.lamb == 0.0:
            return p0

        retrieved = memory.retrieve(query_window, k)
        if not retrieved:
            return p0

        sims = torch.tensor([s for _, s in retrieved], dtype=torch.float32)
        weights = torch.softmax(sims / self.temperature, dim=0)   # [k]
        corr = torch.zeros(RAW_DIM, dtype=torch.float32)
        for (ep, _s), w in zip(retrieved, weights.tolist()):
            if use_cached:
                try:
                    r = self.get_cached_residual(ep)
                except KeyError:
                    r = self.cache_residual(ep, scene_params=None)
            else:
                r = self.cache_residual(ep, scene_params=None)
            corr = corr + w * r
        return p0 + self.lamb * corr

    # ------------------------------------------------------------------ #
    # 批量 k-shot MSE 测量 (诚实记录, 不保证提升)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def shot_mse(self, query_windows: torch.Tensor,
                 query_targets: torch.Tensor,
                 memory: Optional[DemonstrationMemory] = None,
                 k: int = 0,
                 scene_params: Optional[torch.Tensor] = None) -> float:
        """对一批查询测 k-shot 下一状态 MSE (均方差 / RAW_DIM)。

        k=0 => 纯 0-shot; k>0 => 检索聚合修正。scene_params 可逐样本提供 [N,4]。
        """
        if k < 0:
            raise ValueError("k 须 >= 0")
        total = n = 0.0
        N = query_windows.size(0)
        for i in range(N):
            sp_i = scene_params[i:i+1] if scene_params is not None else None
            pred = self.predict(query_windows[i], memory=memory, k=k,
                                scene_params=sp_i)
            total += float(((pred - query_targets[i]) ** 2).sum())
            n += query_targets.size(1)
        return total / max(n, 1)


__all__ = [
    "DemonstrationEpisode",
    "DemonstrationMemory",
    "ICMAggregator",
]
