# UDOS v7.0.1 验证报告（可复现）

环境：Linux CPU，Python 3.12，PyTorch 2.14（CPU，`torch.set_num_threads(2)`），无 GPU/docker/sudo。
所有数字来自本机实测脚本，证据分级 **verified**（CPU 合同内）；不与 legacy 版本跨合同比较倍数。

## 复现命令

```bash
# 单元 + 契约测试（M1 统一图 / M2 训练 / M3 不确定性 / M4 API）
python3 -m pytest tests7 -p no:warnings -q

# 档位选择 + 选中档训练，落盘 checkpoints7/worldmodel_v7.0.1.pt
python3 scripts7/train_v7.py

# 综合验证门禁，写 reports7/v7_verification.json
python3 scripts7/verify_v7.py

# 真实 HTTP 冒烟（子进程拉起 /api/v7），写 reports7/service_smoke.json
python3 scripts7/service_smoke.py
```

## 1. 测试

`tests7` 共 16 条，全部通过：M1 统一推理图 8、M2 训练 1、M3 不确定性 4、M4 API 3。

## 2. 模型档位（held-out 决定，非预设）

| hidden | 可学习参数 | val 准则 | test oracle | test blind |
|---|---|---|---|---|
| 64  | 94,382 | 0.0653 | 0.0328 | 0.0569 |
| 128 | 270,510 | 0.0488 | 0.0278 | 0.0548 |
| **256（选中）** | **966,830** | **0.0466** | **0.0274** | **0.0459** |

选择规则：验证准则最小、且不劣于最优 3% 的最小档。128（0.0488）超出 0.0466×1.03=0.0480，故选 256。

## 3. test rollout MSE（H=4，整体/分类别）

| 路径 | overall | uniform | accel | spring | collision |
|---|---|---|---|---|---|
| oracle（显式参数） | 0.0274 | 0.0177 | 0.0587 | 0.0170 | 0.0030 |
| blind（仅窗口估计） | 0.0459 | 0.0077 | 0.0932 | 0.0639 | 0.0036 |

盲路径在 uniform/collision 上接近 oracle；accel/spring 因隐藏参数（a、ω）仅部分可辨识而退化，与第 5 节 skill 一致。

## 4. 预测区间覆盖率（split conformal，test 640 窗）

| 路径 | 名义 80% | 名义 90% | 名义 95% |
|---|---|---|---|
| oracle 经验覆盖 | 0.803 ✓ | 0.908 ✓ | 0.968 ✓ |
| blind 经验覆盖 | 0.775 ✓ | 0.894 ✓ | 0.954 ✓ |

容差 ±0.05。带宽严格随 α 变化（HTTP 冒烟实测平均半宽：α0.2→0.224、α0.1→0.461、α0.05→0.831）。
对照：legacy 旧 conformal 三档名义覆盖全为 0.8988（α 失效），v7 已根治。

## 5. 参数蒙特卡洛扇形（blind，名义 80%）

- 裸参数扇形（仅参数不确定性）经验覆盖约 0.27–0.74，**不足以**作为总不确定区间；
- 与残差 conformal 带取包络后经验覆盖 **0.828 ✓**（门禁 0.80±0.10）。

## 6. 隐藏参数辨识（test，有效槽）

| 参数 | MAE | 均值基线 MAE | skill | 可观测性头 |
|---|---|---|---|---|
| v0 | 0.664 | 0.968 | +0.31 | 0.76 |
| accel_a | 0.936 | 0.614 | **−0.53（不可辨识）** | 0.79 |
| spring_omega | 0.153 | 0.256 | +0.40 | 0.90 |
| other_v2 | 0.051 | 0.240 | +0.79 | 0.90 |

## 7. 延迟（CPU 2 线程，batch=1，200 次中位数，仅 v7 合同）

- `predict_next`：约 1.5 ms
- `rollout(H=4)`：约 4.0 ms

## 8. HTTP 冒烟（reports7/service_smoke.json）

health=200、predict=200（形状 [1,4,6]）、interval 三档带宽严格递增、metrics=200 且 gating all_pass=true，`smoke_ok=true`。

## 9. 综合门禁

`reports7/v7_verification.json` → `gating.all_pass = true`（7 项：6 条 conformal 覆盖 + 1 条扇形覆盖）。

## 10. 明确未验证（不冒充）

- 0.5B/5B 真实小模型端到端、vLLM KV-offload、NEURON/CoreNEURON DHS、MuJoCo-MJX：需 GPU/HPC，本环境做不了，列为 M5 闸门；
- 多 LLM 交叉打分：需供应商 key；
- 双云生产部署/域名/ICP/HTTPS：需实名与备案；
- 本文所有 CPU 数字仅在 v7 CPU 合同内成立，不得外推为 GPU/大规模收益。
