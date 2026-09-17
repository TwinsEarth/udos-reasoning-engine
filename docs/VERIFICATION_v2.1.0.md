# UDOS 推演引擎 v2.1.0 验证报告

- 版本：v2.1.0（在 v2.0.0 基础上迭代，向后兼容）
- 环境：Python 3.12.11 / torch 2.14.0+cpu / CPU 2 线程 / 无 GPU
- 复现入口：`python scripts/build_v21_checkpoint.py`（训练+评估+落 checkpoint，约 1 分钟）
- 产物：`benchmarks/results/evaluation_v2.1.0.json`、`checkpoints/predictor_v2.1.0.pt`（222 KB）

---

## 1. 本轮目标：补齐 v2.0 自陈短板

v2.0 验证报告自陈三项短板：①仅合成数据；②**场景门控零初始化、预测器单独训练，
`scene_gate` 始终≈0，"场景改变推演"从未被联合训练激活**；③容器无 daemon 未实建。
v2.1 主攻短板②，并把"只预测下一帧"升级为"多步滚动推演"，同时补物理一致性诊断。

| 代际能力 | 含义 | 主要落点 |
|---|---|---|
| M1 多步滚动推演 | 单步 → teacher-forcing 多步训练 + 自由 rollout H 步 | `dynamics.ParametricDynamicsDataset`、`training.PhysicsPredictor.rollout`、`reasoning.reason(horizon=)` |
| M2 场景条件联合训练（消融实证） | 隐藏物理参数经 `scene_encoder` 进入训练回路，门控真正被学出来 | `training`（scene_encoder/`_parametric_loss`）、`evaluation.evaluate_predictor` |
| M3 物理一致性 | 一阶欧拉运动学残差，分运动类型诊断 | `dynamics.kinematic_residual`、评估体系 |
| 工程化 | 评估体系、服务多步、`--checkpoint` 开箱预加载 | `evaluation.py`、`server.py`、`persistence.py` |

---

## 2. 数据：为什么需要"隐藏场景参数"

参数化样本 = 6 帧观测窗口 X `[N,6,6]` + 4 维隐藏物理参数 P `[N,4]`
（`v0 / accel_a / spring_omega / other_v2`）+ 未来 4 步真值 Y `[N,4,6]`，四类运动
uniform/accel/spring/collision 各若干轨迹，每条轨迹按滑窗展开多个样本。

关键设计：P 是**仅凭观测窗口无法唯一确定**的量。例如恒定加速度趋势、弹簧角频率、
碰撞中被撞质点的速度——看不到 P，同样的 6 帧窗口可对应不同未来。这正是检验
"场景条件是否真有用"的前提：若给不给 P 都一样，说明场景通路是摆设（v2.0 的问题）。

---

## 3. 核心实验结果（seed=42，52,191 参数，训练 56.8s）

### 3.1 单步精度与场景条件消融（M2 主证据）

| 指标 | 数值 |
|---|---|
| 单步 MSE 训练前 → 训练后 | 5.4710 → **0.0819**（降 66.8×） |
| conditioned MSE（给隐藏参数 P） | **0.0819** |
| unconditioned MSE（P 置零） | 0.8131 |
| **场景条件增益 condition_gain_x** | **9.93×** |
| `scene_gate` 训练前 → 训练后 | 0.000 → **−0.2899**（真正被训练激活，v2.0 恒为 0） |

分运动类型的条件/无条件 MSE（同一正式 checkpoint 复算）：

| 运动类型 | conditioned | unconditioned | 增益 | 解读 |
|---|---|---|---|---|
| uniform 匀速 | 0.0649 | 0.1793 | 2.76× | 初速隐藏，给参数更准 |
| accel 匀加速 | 0.0587 | 2.3507 | **40.03×** | 6 帧窗口估不出加速度趋势，P 最关键 |
| spring 弹簧 | 0.1913 | 0.2131 | 1.11× | 窗口内已能估角频率，隐藏参数边际小（诚实保留） |
| collision 碰撞 | 0.0242 | 0.1087 | 4.49× | 被撞质点速度在窗口外，需 P |

> spring 增益接近 1 不是失败而是合理现象：角频率信息已泄漏在观测窗口的振动里。
> 报告不掩盖这一类，整体 9.93× 增益由 accel/collision/uniform 共同支撑。

该能力由回归测试 `test_scene_conditioning_ablation_gain` 以 `gain>1.3` 锁死
（独立快速配置 d64/it8/n20/ep30 实测增益 1.86×，留足防 flaky 余量）。

### 3.2 多步自由 rollout（M1）

训练后让模型把自己的预测拼回窗口自由滚动 4 步，逐步 MSE：

```
step1 0.0819 → step2 0.1045 → step3 0.1566 → step4 0.2933
```

误差随推演步数单调累积（4 步约 3.6×），长时程不确定性可见、符合预期；teacher-forcing
训练保证了前两步精度，后续衰减被评估曲线量化，而非隐藏。

### 3.3 物理一致性诊断（M3）

一阶欧拉 `x(t+1)=x(t)+v(t)·dt`（用**前一帧**速度）的位置 RMS 残差：

| 范围 | overall | uniform | accel | spring | collision |
|---|---|---|---|---|---|
| 残差 | 0.189 | **0.122（最低）** | 0.225 | 0.245 | 0.151 |

匀速段最贴合一阶运动学（符合物理）；加速/弹簧段真值本身带 O(a·dt²) 偏离，残差更高，
属正确表现。

---

## 4. 一处关键纠错（根因 → 反证实验 → 修正）

物理一致性第一版用**预测时刻速度**积分：`x(t+1)=x(t)+v(t+1)·dt`，并作为全局损失
（`phys_weight=0.15`）。对照实验（同种子/数据，仅改 phys_weight）反证其错误：

| phys_weight | 条件单步 MSE | 运动学残差 |
|---|---|---|
| 0.0 | 0.1822 | 0.3492 |
| 0.15 | **0.2936（变差）** | **0.3878（变差）** |

根因：对加速运动 `v(t+1)=v(t)+a·dt`，正确位移是 `v(t)·dt+½a·dt²`，用 `v(t+1)`
积分必然系统性偏差；把它当全局硬损失，等于强迫模型违背真实加速轨迹，反而同时抬高
MSE 与残差。

修正：①残差公式改用**前一帧速度** `v(t)`（匀速严格成立）；②把一致性正确定位为
**分类型诊断指标 + 可选弱先验**，`phys_weight` 默认 0，不对混合运动施加全局硬约束。
修正后匀速段残差随训练显著下降（独立快速实验 2.50→0.275），且不损害 MSE。此死路
已记录，避免后续回退。

---

## 5. 测试 / 覆盖率 / 服务 / 性能

- **单元与契约测试：64 passed**（v2.0 的 55 项**原样全绿** + v2.1 新增 9 项）。
  新增覆盖：参数化数据集形状与多起点、一致性公式（匀速≈0/违反>0）、scene_encoder 与
  rollout 形状、多步训练与 v2.0 旧数据集兼容、评估报告结构、**场景条件消融增益**、
  带场景编码器 checkpoint 逐元素存载、多步 reason future_states、服务 `--checkpoint` 预加载。
- **覆盖率 95%**：persistence 100%、dynamics 99%、training 97%、evaluation 95%、
  ctm 96%、reasoning 94%、gpm 99%、server 84%。
- **端到端 HTTP 冒烟 11/11 PASS**（`python scripts/smoke_test.py`）：含 v2.1 多步
  /train 评估、reason horizon=4 返回 4 步 future_states，以及**单独起一个带
  `--checkpoint` 的进程，验证启动即 `predictor_trained=true`**。
- **性能回归守卫 `make guard` 通过**：ctm_small 7.60ms、gpm_small 1.89ms、
  e2e_small 8.08ms、upstream_ctm 7.80ms，峰值 RSS 285.7MB，均低于门槛、未较 v2.0 退化。

## 6. 向后兼容证据

1. v2.0 全部 55 测试未删除、未放宽，全部通过；版本号断言同步升至 2.1.0。
2. `PhysicsPredictor` 不传 `scene_param_dim` 时无 scene_encoder，rollout/单步行为同 v2.0；
   `CTMTrainer` 同时吃旧 `DynamicsDataset(X,y)` 与新参数化集（`_is_parametric` 分流）。
3. `load_predictor` 用 `.get("scene_param_dim")`，v2.0 旧 checkpoint 缺该字段也能载入。
4. `scene_gate` 仍零初始化，未训练时条件通路与无条件逐元素等价。
5. 服务不带 `--checkpoint` 时 `/health` 初始 `predictor_trained=false`，与 v2.0 一致。

## 7. 尚存缺口（不夸大）

- 数据仍为可解析合成运动；真实物理精度需接入实测/仿真数据再训。
- rollout 误差随步数累积，更长时域需 scheduled-sampling / 闭环再训练。
- 协同引擎 `reason` 的多步 rollout 走观测窗口外推（GPM 场景嵌入维度与预测器隐藏参数
  维度不同，未混用）；GPM 超网络与预测器的端到端联合微调留待后续。
- 沙箱无 docker daemon，镜像 Dockerfile/compose 已升级到 2.1.0 并内置预加载，
  但未在本机实际 build/run，仅以等价的本地进程 + `--checkpoint` 完成运行时验证。
