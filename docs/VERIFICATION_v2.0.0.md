# UDOS v2.0.0（第二代）验证报告 — 能学习 · 真耦合 · 可持久化 · 可解释

运行环境：Ubuntu、Python 3.12、torch 2.14.0+cpu、2 线程。所有数字为本沙箱实跑，
原始训练曲线存于 `benchmarks/results/training_v2.0.0.json`，可用
`python -m demos.demo4_learn_to_predict` / `make train` 复现。

---

## 一、代际目标与一代短板

| 一代 v0.2 自述短板 | 二代 v2.0 交付能力 | 新增模块 |
|---|---|---|
| 权重随机、只验证数据流，**不能学习** | 合成动力学 + 自监督训练闭环，真实 loss 下降 | `udos/dynamics.py`、`udos/training.py` |
| GPM 注入基座、CTM 自跑，**两引擎表示不通** | GPM 场景嵌入条件化 CTM 每个内部 tick（真耦合） | `ctm_engine.scene_*`、`gpm.scene_embedding`、`reasoning` |
| 训练成果无法保存 | checkpoint 存/载，载入逐元素一致 | `udos/persistence.py` |
| 只输出抽象向量 | 输出下一时刻 position/velocity；服务新增 `/train` | `reasoning`、`udos/server.py` |

环境说明：本轮运行时第三次被重置（torch/pytest 丢失），已重装 CPU 依赖闭环，非代码缺陷。

---

## 二、学习闭环（第二代核心证据）

合成四类可解析运动（匀速 / 匀加速 / 弹簧谐振 / 一维弹性碰撞），自监督构造
"6 步历史窗口 → 下一时刻 [pos(3),vel(3)]" 样本，PhysicsPredictor =
观测编码器 + CTM(8 内部 tick) + 物理解码头；损失 = 逐 tick 深度监督 MSE
（后期 tick 权重更高）+ 置信度正则。

### 2.1 默认训练（服务 `/train`，45 epoch × 40/类，960 样本，48,798 参数，21.9s）

| 指标 | MSE | 说明 |
|---|---:|---|
| 未训练网络 | 6.0513 | 随机权重基线 |
| “状态不变”朴素外推 | 0.2095 | naive baseline |
| **训练后** | **0.0841** | test 集 |
| 较未训练下降 | **71.96×** | 证明网络确实在学习 |
| 较朴素基线 | **2.49×** | 优于无模型外推 |

- train loss：epoch0 `7.001` → epoch15 `0.259` → epoch30 `0.145` → epoch44 `0.147`，
  单调快速下降后收敛。
- certainty（1−归一化熵）：`0.005 → 0.833`，训练后模型从近乎无置信变为高置信，
  与 MSE 改善相互印证。
- 独立 sanity（50 epoch、48/类）：test MSE `6.04 → 0.044`（降 136×、胜 naive 4.3×）。

### 2.2 自动化锁（tests/test_training.py，7 项）
数据集形状/四类覆盖、滑窗切分、前向形状、**梯度回传到 CTM 核心**、eval 确定性，
以及 `test_trains_and_beats_baselines`：固定种子训练后必须同时满足
`final < 0.5×初始` 且 `final < naive`（不可学习的冻结/错误解码器会变红）。

---

## 三、GPM → CTM 真耦合（tests/test_conditioning.py，7 项）

- **向后兼容**：场景门控 `scene_gate` 零初始化，开启条件化时初始前向与一代无条件
  **逐元素一致**（atol=1e-6）——不破坏一代任何权重与行为。
- 同一 scene_context 两次结果一致；门控打开后**不同场景条件产出不同推演**
  （反例：若 ctx 被忽略则此测试变红）。
- 未配置 `scene_dim` 却传 context 会明确报错；GPM `scene_embedding` 形状正确且
  因持久编码器而可复现。
- 协同引擎默认条件化，`reason` 结果带 `scene_conditioned=true`；挂载训练好的
  预测器后返回可解释 `predicted_next_state={position,velocity}`。

---

## 四、持久化（tests/test_persistence.py，3 项）

- `save_predictor/load_predictor` 载入后前向与保存前**逐元素一致**（atol=1e-7），
  bundle 含版本 `2.0.0`、CTMConfig（含 scene_dim）、raw_dim、训练指标。
- 通用 `save_module/load_module_state` 对整套 UDOSReasoningEngine 存/载，
  同名参数逐元素一致。

---

## 五、服务与部署

- 新增 `POST /train`：在合成动力学上短训并挂载预测器，返回三条 MSE 对照、
  loss/eval/certainty 曲线与下降倍数；参数越界返回 400，不崩进程。
- `/reason` 新增 `scene_conditioned`，训练后返回 `predicted_next_state`；
  `/health` 暴露 `version=2.0.0` 与 `predictor_trained`。
- 端到端冒烟（真实起 HTTP 进程）**10/10 通过**：health/internalize/reason 条件化/
  train 学习闭环/health 挂载/reason 物理预测/reset/400×2/demo。
- 容器：镜像 tag 与 compose 升至 2.0.0，compose 内存上限调至 1536M（训练需要）。
  沙箱仍无 docker daemon，镜像构建命令随工程交付（`make docker-build`）。

---

## 六、回归与性能（不退化）

- **测试 55 passed** = 一代 36（原样全绿，向后兼容）+ 二代新增 19
  （training 7 / conditioning 7 / persistence 3 / service+2，service 原 5 项保留）。
- 覆盖率：dynamics **100%**、persistence **100%**、training 96%、ctm 96%、
  gpm 99%、reasoning 94%、server 85%、**总 90%**（一代为 87%）。
- 性能回归守卫 `--guard` **通过**；预测器前向中位数 **2.84 ms**（48,798 参数）；
  进程峰值 RSS **285 MB**（一代 288 MB，持平）。

## 七、复现实验命令
```bash
make test      # 55 项全量回归
make cov       # 覆盖率
make train     # 学习闭环 demo (默认40 epoch, 可 EPOCHS=45)
make guard     # 性能守卫
make smoke     # HTTP 10 项冒烟
# 服务: python -m udos.server 后  curl -XPOST localhost:8000/train -d '{}'
```

## 八、剩余缺口（如实说明）
1. 训练数据为**可解析合成动力学**，用于证明学习能力与闭环；真实物理精度需接入
   实测/仿真数据训练，当前不代表真实场景精度。
2. 场景条件化的 `scene_gate` 从零开始学，当前协同演示中门控仍接近初始；要体现
   "场景改变推演"需对端到端联合训练（当前 GPM 仍零反向、预测器单独训练）。
3. 容器镜像未在本沙箱实建（无 daemon），同 v0.2 以等价文件集 + 真实进程冒烟替代。
