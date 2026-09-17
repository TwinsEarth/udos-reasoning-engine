# UDOS Reasoning Engine v5.4.3

首个开源版本。双引擎（CTM 连续思维机 + GPM 场景内化）认知架构内核，纯 CPU 可跑，采用 Apache-2.0 许可。

## 本版本重点：精细生物物理数值核 `udos/finesim/`

- **被动电缆（Rall 1959）**：衰减长度 λ、时间常数 τ、解析解与离散数值自对照（默认 λ=5.0 mm、τ=10 ms、拟合误差 0）。
- **Hodgkin–Huxley 动作电位（1952）**：确定性 Euler（dt=0.05 ms）、过零计 spike；I=10 µA → 4 个 spike、峰值 42.72 mV；f-I 曲线 I=2/5/10/20 → 0/1/4/5 Hz。
- **NMDA 镁阻滞与时序性抑制（可证伪）**：Jahr–Stevens 电导；有镁阻滞时平台电压随时序差 Δt 敏感（range=0.106956），**移除镁阻滞后立刻不敏感（range=0）**，构成可跑反的对照（Du 2017 / Doron 2017）。
- **Payeur 四类树突信息处理**：时空滤波 / 信息选择 / 信息路由 / 信息多路复用（固定 seed 合成演示）。
- **突触位置鲁棒性**：远端前馈 spread=0.0563 显著小于近端胞体 0.1878，方向与“远端被动衰减抗噪”一致。
- **Hines 串行 vs DHS 层级并行**：5 节点树 serial=5 步 / DHS=4 层，逐节点电压差 < 1e-9。
- **NGRAD**：仅作外挂假设（`analogy hypothesis, not trained`），不进 `state_dict`。

开关：`UDOS_FINESIM=on` 启用 `/finesim/*`（关闭返回 503）；`GET /intel/finesim` 始终 200 并在无 GPU/NEURON 时如实报告 `ENV_BLOCKED`。

## 能力总览

双引擎核心（v0.1）之上，按“外挂、零梯度、opt-in、可证伪”叠加：可信推演（v2.3–v2.4）、效率服务（v2.5）、因果决策（v2.6–v2.7）、具身闭环与世界模型（v2.8–v3.9）、自进化与多智能体/隐式思考（v4.x）、安全治理与 AGI/ASI 情报（v5.0.1–v5.0.2）、KV Cache 分层类比（v5.1）、类脑树突（v5.3）、精细生物物理核（v5.4.3）。完整时间线见 [CHANGELOG.md](CHANGELOG.md) 与 `docs/VERIFICATION_v*.md`。

## 验证

- 测试：**1572 个用例全部通过**（204 个测试文件），覆盖率约 **93%**。
- 恒定锚点：主模型 `PhysicsPredictor` **52191 个可学习参数**、`eval_mse = 0.045556`；`checkpoints/` 内 33 个自训练 checkpoint 全部可加载且主参一致。
- 依赖极轻：`torch`、`numpy`、`pytest`，可选 `huggingface_hub`（仅真实上游适配器）。

## 能力边界

`finesim / dendrite / kvcache` 为 CPU 合成机制原型，**不是** NEURON/CoreNEURON/GPU/QAT 复现；“16 线程 / 10× / 100–1000× / 5 万神经元”等为论文或厂商口径，标 `[UNVERIFIED]`；MuJoCo 因果虚拟小鼠、在线多 LLM 打分、GPU vLLM / NEURON 对拍尚未实现或未启动。详见 README“能力边界”与 [NOTICE](NOTICE)。

## 许可与归属

Apache License 2.0。仓库不包含上游或第三方源代码/预训练权重，第三方元数据与唯一内置资产（Apache ECharts）的归属见 [NOTICE](NOTICE)。
