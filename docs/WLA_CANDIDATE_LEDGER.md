# UDOS v3.9 WLA 类比线 — 候选账本（被否决/降级/采纳裁决）

> 规则：每个机制同合同 A/B 证据落 `benchmarks/results/*.json`；收益不稳或被反证照实
> **默认 opt-in**，并在此留账（不删代码、不删候选，仅记录裁决依据）。

| 机制 | A/B 证据（JSON） | 结果 | 裁决 |
|---|---|---|---|
| 稀疏 change-mask vs 3.6 PWM 稠密 rollout | `wla_sparse_vs_dense_ab.json` | 逐步 MSE 稀疏 0.034/0.059/0.112/0.211 **低于** 稠密 0.0/0.155/0.425/0.889（稠密末步潜在漂移累积）；成本稀疏 0.85 vs 稠密 9.24 单位 | **采纳**（稀疏更省且合成基准上误差不累积）；仍 opt-in 外挂 |
| change-mask 小型 VQ 码本 | `training_v3.9.0.json` → change_vq | 利用率 1.0、未坍塌、recon_mse 如实记录 | **采纳**（外挂零梯度） |
| 三路 RVQ 动作分词（每分组） | `training_v3.9.0.json` → rvq_action_tokenizer | 两级利用率均 1.0、max_share≈0.19、未坍塌 | **采纳** |
| 动作 token-状态-任务对齐 | `training_v3.9.0.json` → alignment | same_task 0.054 > cross_task -0.076，consistent=true | **采纳** |
| **flow-matching 少步解码器 vs 直接回归** | `wla_flow_vs_regress_ab.json` | flow MSE 0.264 **高于** 直接回归 0.169 | **被反证 → opt-in（默认关）留候选**：合成小数据上少步去噪未优于线性回归；不删，保留为 MMDiT 类比接口，待更大数据再验 |

## 备注
- 全部外挂模块**不进主 state_dict**，主 predictor 恒 52191、eval_mse 恒 0.045556。
- analogy, not reproduction：均为 CPU 合成状态/动作代理，非复现 6B / 真机 / VLM。

## v4.1 自规划自监督线候选裁决

| 机制 | A/B 证据（JSON） | 结果 | 裁决 |
|---|---|---|---|
| 自生成课程 vs 固定课程 | `curriculum_self_vs_fixed_v4.1.0.json` | 同探针预算可解率均 1.0（delta 0.0）；自生成额外探明难度前沿 horizon=8 | **采纳**（自适应暴露难度前沿）；opt-in 外挂 |
| PWM rollout 一致性伪标签 | `pseudolabel_gain_v4.1.0.json` | 伪标签一致性均值 0.48，有区分度 | **采纳**为自监督信号（opt-in） |
| **守恒+多视角硬门叠加** | `pseudolabel_gain_v4.1.0.json` | 对加速/振动轨迹过度降权：均值 final_weight=0.03，144/144 被压，有效一致性反降（-0.466） | **被反证 → opt-in（默认关）留候选**：非匀速动力学本就不守恒，硬门错杀；不删，待调 scale/分运动类再验 |
