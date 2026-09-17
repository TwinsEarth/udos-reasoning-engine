# UDOS v7.0.2 验证报告（可复现）

环境：Linux CPU，Python 3.12，PyTorch CPU（`torch.set_num_threads(2)`），无 GPU/docker/sudo。
所有数字来自本机实测脚本，证据分级 **verified**（CPU 合同内）；不与 legacy 版本跨合同比较倍数。

> v7.0.2 为**补丁版本**：新增确定性可观测运动学通道、修两个服务/校准 bug。下表同时保留 v7.0.1 基线做同合同 A/B（相同种子、相同数据切分、相同训练配方，唯一 delta 是运动学通道与两处 bug 修复）。

## 复现命令

```bash
# 单元 + 契约测试（M1 统一图 / M2 训练 / M3 不确定性 / M4 API / v7.0.2 运动学）
python3 -m pytest tests7 -p no:warnings -q

# 档位选择 + 选中档训练，落盘 checkpoints7/worldmodel_v7.0.2.pt
python3 scripts7/train_v7.py

# 综合验证门禁（含运动学恢复门禁 + 相对 v7.0.1 改进门禁），写 reports7/v7_verification.json
python3 scripts7/verify_v7.py

# 真实 HTTP 冒烟（子进程拉起 /api/v7），写 reports7/service_smoke.json
python3 scripts7/service_smoke.py
```

## 1. 测试

`tests7` 共 **26 条全部通过**：M1 统一推理图 8、M2 训练 1、M3 不确定性 4、M4 API 3、**v7.0.2 运动学通道与服务修复 10**（`tests7/test_v702_kinematics.py`）。
legacy `udos/` 全量回归另计（见 CHANGELOG，1656 通过 / 2 跳过 / 0 失败）。

## 2. 模型档位（held-out 决定，非预设）

v7.0.2（含 352 个运动学编码器参数）：

| hidden | 可学习参数 | val 准则 | test oracle | test blind |
|---|---|---|---|---|
| 64  | 94,734 | 0.0392 | 0.0179 | 0.0251 |
| 128 | 270,862 | 0.0416 | 0.0175 | 0.0214 |
| **256（选中）** | **967,182** | **0.0348** | **0.0209** | **0.0248** |

选择规则（预注册）：验证准则最小、且不劣于最优 3% 的最小档。64（0.0392）超出 0.0348×1.03=0.0358，故选 256。选档只看 val，不看 test。

## 3. test rollout MSE（H=4，整体/分类别）与 v7.0.1 同合同 A/B

| 路径 | overall | uniform | accel | spring | collision |
|---|---|---|---|---|---|
| v7.0.2 oracle（显式参数） | 0.0209 | 0.0074 | 0.0618 | 0.0119 | 0.0026 |
| **v7.0.2 blind（仅窗口）** | **0.0248** | 0.0073 | **0.0693** | **0.0193** | 0.0033 |
| v7.0.1 blind（基线） | 0.0459 | 0.0077 | 0.1053 | 0.0637 | 0.0040 |
| blind 相对变化 | **−46%** | −5% | **−34%** | **−70%** | −17% |

改进集中在此前最差、且其隐藏量本就可从状态直接反演的 accel/spring；uniform/collision 本就接近 oracle，无显著回退。预注册改进门禁（blind overall≤0.0459、blind accel≤0.075、blind spring≤0.0637）**全部通过**。

## 4. 确定性可观测运动学恢复（不依赖训练，v7.0.2 新增）

`udos7/kinematics.py` 在 held-out test（每类 160 窗）上的实测（`kinematic_recovery`）：

| 量 | 指标 | 实测 | 门禁 |
|---|---|---|---|
| 三维加速度向量 a_lin | 对真值（X+Y 速度斜率）恢复 skill | **1.000**（MAE≈0） | ≥0.95 ✓ |
| 匀速窗加速度误报 | \|a_lin\| 均值 | **0.0000** | <1e-4 ✓ |
| 弹簧角频率 ω | 召回 / 有效 MAE / 非弹簧误报率 | **1.00 / 0.023 / 0** | 召回≥0.9、误报 0 ✓ |
| 碰撞速度跳变 | 跨碰撞窗检出 / 非碰撞误报率 | 0.74 / **0** | 误报 0 ✓（检出仅对跨帧窗有定义） |

这把 v7.0.1 “标量 accel_a skill=−0.53”的负结果正确归因为**表征口径错误**（方向 d 与带符号标量 a 不可分离），而非“加速度不可观测”：三维加速度向量可从速度斜率精确反演。

## 5. 预测区间覆盖率（split conformal，test 640 窗）

| 路径 | 名义 80% | 名义 90% | 名义 95% |
|---|---|---|---|
| oracle 经验覆盖 | 0.805 ✓ | 0.917 ✓ | 0.971 ✓ |
| blind 经验覆盖 | 0.793 ✓ | 0.909 ✓ | 0.965 ✓ |

容差 ±0.05。HTTP 冒烟实测平均半宽严格随 α 变化：α0.2→0.187、α0.1→0.388、α0.05→0.730。
v7.0.2 另修：请求 horizon 短于校准视界时的张量切片；服务校准器缓存键由 `use_explicit` 改为 `(use_explicit, horizon)`，不同视界不再串用带宽（均有测试）。

## 6. 参数蒙特卡洛扇形（blind，名义 80%）

与残差 conformal 带取包络后经验覆盖 **0.826 ✓**（门禁 0.80±0.10）。裸参数扇形仍只刻画参数不确定性，不单独作为总区间。

## 7. 延迟（CPU 2 线程，batch=1，200 次中位数，仅 v7 合同）

- `predict_next`：约 1.8 ms
- `rollout(H=4)`：约 4.2 ms

新增运动学通道为每窗一次小规模最小二乘/PCA，batch=1 延迟相对 v7.0.1（1.5/4.0 ms）在测量噪声内，无数量级变化。

## 8. HTTP 冒烟（reports7/service_smoke.json）

health=200、version=7.0.2、predict=200（形状 [1,4,6]）、interval 三档带宽严格递增、metrics=200 且 gating all_pass=true，`smoke_ok=true`。

## 9. 综合门禁

`reports7/v7_verification.json` → `gating.all_pass = true`（**15 项**：5 条运动学恢复 + 3 条相对 v7.0.1 改进 + 6 条 conformal 覆盖 + 1 条扇形覆盖）。

## 10. 明确未验证（不冒充）

- MuJoCo CPU 最小虚拟小鼠因果代理：**cpu-proxy 已跑通**（`proxies/mujoco_mouse/`）；非 DeepMind virtual rodent，无 RL/神经对齐；
- 0.5B/5B 真实小模型端到端、vLLM KV-offload、NEURON/CoreNEURON DHS、MuJoCo-MJX：需 GPU/HPC（约 ¥35,400/月档），本环境做不了，列为 M5 闸门；CPU 档 0.97M 参数数字不得外推为 GPU/大规模收益；
- 多 LLM 交叉打分：需供应商 key；
- 双云生产部署/域名/ICP/HTTPS：需实名与备案；
- 本文所有 CPU 数字仅在 v7 CPU 合同内成立。
