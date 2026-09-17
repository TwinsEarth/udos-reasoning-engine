# Diffusion Policy — 架构卡片

## 1. 身份与证据等级
- 对象：Diffusion Policy（扩散/flow-matching 动作头，Stanford AI-R，MIT）。
- 证据等级：**inferred**（公开仓库 `real-stanford/diffusion_policy`；本次未逐文件复核）。
- UDOS 分级：L1（动作块/扩散动作表示适配，不跑扩散权重）。

## 2. 入口与调用链（inferred）
- policy 以条件扩散从噪声去噪出 `[horizon, action_dim]` 动作序列。
- 输入：观测条件；输出：动作块（与 ACT 同构的 chunk）。

## 3. 数据 / 动作格式
- 动作块 `[H, action_dim]`；UDOS 复用 ActionChunking 适配。

## 4. 运行时装配
- L1：`ActionChunkingConnector`，无权重；去噪过程不实现。

## 5. 模块责任（UDOS 映射）
- → `ActionChunkingConnector`；对照 wla flow 动作专家。

## 6. 最小反例 / 不可证项
- 不可证：未装 diffusers/未跑去噪；仅做 chunk 格式归一。
