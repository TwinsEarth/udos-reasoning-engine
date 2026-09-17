# UDOS v5.3.0 类脑「物理 Token 微观编译器」— 树突区室机制原型

> 全部为 CPU 合成 analogy, not reproduction；真 GPU/NEURON/DeepDendrite/生物数据 ENV_BLOCKED。

## 模块
- `udos/dendrite/compartment.py`：apical/basal 区室，距离→被动衰减 exp(-leak·distance)。
- `dendritic_compute.py`：重合检测(时间窗)、NMDA 式非线性放大、时序性抑制 Δt 门控曲线。
- `multimodal_sync.py`：多模态合成时间戳对齐（微秒级为生物/硬件口径，标 analogy）。
- `dhs_scheduler.py`：区室 DAG 分层调度 vs 朴素串行；数值一致+关键路径步数下降。

## 文献引用（区分引用/自测）
- Beniaguev/Segev/London, Neuron 109(17):2727–2739, 2021：L5PC 需 5–8 层时序卷积 DNN（R²>0.95），深度源于 NMDA×树突形态，可学 XOR。【引用】
- DeepDendrite, Nature Communications 14, 2023, PMC10507119：DHS 理论证明计算最优且精确，GPU 相比经典串行 Hines(CPU) 快 2–3 个数量级（约100–1000×，1000×为上界、基线为 Hines/CPU）。【引用】
- Continuous Thought Machines, NeurIPS 2025, arXiv 2505.05522：神经元级时序+同步矩阵 S=Z·Zᵀ。【引用】
- "O(N³)→O(2N)" 与 "最多16线程"：未在 PMC 原文证实 → **[UNVERIFIED]**，不写入自测结论。worker_count 参数化。
- 澎湃 thepaper 29902805：二手科普，不作证据。

## 自测（analogy）
- DHS A/B：benchmarks/results/dhs_ab.json，数值一致+关键路径步数下降，verdict=ACCEPT。
- 鲁棒对照：dendrite_robustness.json，合成假设值，不声称复现生物效应量。

## 端点（UDOS_BRAIN=on）
POST /brain/sim/run、/brain/dhs/benchmark、/brain/robustness/run；GET /intel/brain 始终 200。
