# DreamerV3 (RSSM) — 架构卡片

## 1. 身份与证据等级
- 对象：DreamerV3（基于世界模型的 model-based RL，RSSM）。
- 证据等级：**inferred**（公开 `danijarh/dreamerv3`；RSSM 结构为论文标准；本次未逐文件复核）。
- UDOS 分级：L1（**仅 RSSM 状态结构 schema**，不下载权重、不跑世界模型前向）。

## 2. 入口与调用链（inferred）
- RSSM 状态 = 离散随机项 `stoch`（categorical）+ 连续确定性隐状态 `deter`（GRU 隐向量）。
- 模型在想象 rollout 中用该状态演化；UDOS 只镜像其状态张量结构。

## 3. 数据 / 动作格式
- `stoch: List[float]`（类别概率，归一化）+ `deter: List[float]`（隐向量）。

## 4. 运行时装配
- L1：`WorldModelRSSMConnector.to_udos({stoch, deter})` → 归一化类别分布 + deter 维度。无权重。

## 5. 模块责任（UDOS 映射）
- → `WorldModelRSSMConnector`；对照 UDOS 自身 world_model.py 的想象机制（analogy not reproduction）。

## 6. 最小反例 / 不可证项
- 反例：缺 stoch/deter → ValueError；概率和归一为 1。
- 不可证：未下载权重、未跑 RSSM 前向/想象；仅状态表示 schema。
