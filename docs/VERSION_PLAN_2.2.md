# UDOS 版本规划与设计：2.1.0 → 2.2.0 → 2.2.1

本文件是 2.2.x 的设计基线（design contract）：先锁问题、可检验假设、A/B 证据口径
与发布门禁，再写代码。所有"预期改善"在跑出结果前都只是假设，不得写成结论。

---

## 1. 出发点：v2.1.0 自陈缺口（证据来源 `VERIFICATION_v2.1.0.md` §7）

| 编号 | v2.1 已观察到的事实 | 性质 | 2.2 是否处理 |
|---|---|---|---|
| G1 | 自由 rollout 误差随步数累积：step1 0.082→step4 0.293（约 3.6×），报告点名需 scheduled-sampling/闭环再训练 | 可在沙箱量化改进 | **2.2.0 主攻 (F1)** |
| G2 | 训练缺早停/最佳状态选择，复现依赖手工；确定性只在入口 manual_seed | 工程治理 | 2.2.0 (F2) |
| G3 | 评估缺"置信度是否可信"的校准视角，rollout 只有逐步绝对误差 | 评估完备性 | 2.2.0 (F3) |
| G4 | 训练好的预测器只能随进程存在，缺在线评估/落盘的服务接口 | 模型管理 | 2.2.0 (F4) |
| G5 | 数据仍为可解析合成运动 | 数据边界，沙箱无真实数据源 | 不处理，继续诚实标注 |
| G6 | 无 docker daemon，镜像未实建 | 环境限制 | 不处理，沿用等价进程验证 |

## 2. 版本切分

- **2.2.0（minor）**：新增能力 F1–F4，全部向后兼容（新参数默认值等价旧行为）。
- **2.2.1（patch）**：2.2.0 实测暴露的边界/数值修复、文档与版本号对齐、QA 收口后发布。
  对外交付版本号为 **2.2.1**；CHANGELOG 同时记录 2.2.0（内部能力里程碑）与 2.2.1（补丁）。

## 3. F1 Scheduled Sampling（核心）设计与可检验假设

### 3.1 机制
v2.1 的多步损失是纯 teacher-forcing：第 h 步输入窗口的未来部分永远用真值
`Y[:, :h]`，模型训练时从未见过"自己上一步的预测"，导致自由 rollout 时 exposure bias、
误差滚雪球。Scheduled Sampling（Bengio et al. 2015 思路）在训练第 h>0 步，以概率
`p_ss(epoch)` 用模型自身上一步**已截断梯度**的预测替代真值拼回窗口：

```
p_ss(epoch) = ss_max * clip((epoch - ss_start) / max(ss_warmup,1), 0, 1)   # 线性爬坡
teacher_in  = where(bernoulli(p_ss) 逐样本, own_pred.detach(), Y[:, h-1])
```

- `TrainConfig` 新增 `ss_max=0.0, ss_start=0, ss_warmup=10`；**默认 ss_max=0 ⇒ 纯
 teacher-forcing，与 2.1 逐位元等价**（这是向后兼容锚点，有专门测试锁）。
- 只在参数化多步路径 `_parametric_loss` 生效；horizon=1 时无任何作用。
- 用 `pred.detach()` 拼输入，避免跨步非单调依赖；当前步对其输出仍正常反传。

### 3.2 可检验假设与 A/B 口径（同合同对照）
- **假设 H1**：在同种子/同数据/同 epoch/同超参、仅 `ss_max` 不同（0 vs 0.6）下，
 SS 模型自由 rollout 的**中后段（step3/step4）MSE 更低**，且不牺牲单步 MSE。
- 合同 fingerprint：seed=42、d64/it8、n_per_kind 固定、epochs 固定、同一 CPU、
 warmup 后评估；配对产出两条 `rollout_mse_curve`。
- 成功判据（写报告前用实测数字裁决）：`mean(roll[2:])` SS < TF（改善方向成立），
 且 `single_step_mse` 不恶化超过 +10%；否则记为"假设未被支持"，不强行上线为默认。
- 回归测试只锁**机制正确性**（ss_max=0 等价、概率随 epoch 单调不减、开启时确实走了
 自回归分支且 loss 可反传），**不把某次具体增益阈值写死进单测**（避免 flaky）；
 增益数字进验证报告与基准 JSON。

## 4. F2 训练治理
- `TrainConfig` 增 `patience: Optional[int]=None`、`min_delta=1e-4`、`early_stop=False`；
 默认 None/False ⇒ 不早停，跑满 epochs（兼容）。开启后按 eval_mse 记录 best_epoch/
 best_state_dict，连续 patience 轮无改善即停；`TrainHistory` 增 `best_epoch/best_eval`。
- 统一 `udos.training.set_seed(seed)`（torch CPU 线程确定性 + manual_seed），训练入口调用。
- 门禁：开启早停不允许突破 epochs 上限；无 eval_ds 时早停被安全忽略并在 history 标注。

## 5. F3 评估增强（`evaluation.py`，纯增量）
- `rollout_growth_x = roll[-1]/roll[0]`：误差累积率，单一数字刻画长时程稳定性。
- `confidence_stratification`：按最终 tick certainty 分 3 档（低/中/高），统计各档
 单步 MSE；**期望（假设 H2）高置信档误差更低**，输出 Spearman 式方向标志
 （高/低档误差比）。这是"置信度可不可信"的诊断，不是校准误差承诺。
- 全部为只读前向；旧返回键不动，只新增键（旧测试不断）。

## 6. F4 模型管理服务
- `POST /evaluate`：对当前挂载预测器在**新鲜采样**的参数化测试集上跑 evaluate_predictor，
 未训练（无预测器）返回 409 + 明确 message。
- `POST /save {name?}`：把当前预测器连同 last_train 指标落 `checkpoints/<name>.pt`，
 返回路径、字节数、版本、指标；未训练返回 409。
- 不引入新第三方依赖；写操作在推理锁内，文件名做白名单清洗防路径穿越。

## 7. 向后兼容锚点（逐条有测试）
1. ss_max=0 时 `_parametric_loss` 与 2.1 数值等价（同种子同批 loss allclose）。
2. 不启用早停时 history/训练行为与 2.1 一致；2.1 checkpoint 仍可 load。
3. evaluate_predictor 旧键全部保留；服务旧接口字段不删不改语义。
4. 版本号三处（`__init__/pyproject/server.health`）统一；旧版本字符串断言同步更新。
5. v2.0/v2.1 全部旧测试不删、不放宽。

## 8. 发布门禁（2.2.1 出包必须全部满足）
- 旧测试 + 新测试全绿；覆盖率不低于 v2.1（95%）量级；`make guard` 通过；冒烟全过。
- H1 有同合同 A/B 原始数字支撑并存基准 JSON；若 H1 不成立，SS 保持 opt-in 且报告如实说明。
- 从交付 zip 全新解压可独立跑通 build 复现脚本；版本三处一致为 2.2.1。

## 9. 失败与回滚
- SS 若损害单步或收益不稳：保持 ss_max 默认 0（opt-in），不作为默认，报告标注。
- 早停若引入不收敛：默认关闭即可完全回退到 2.1 行为。
- 任一新接口异常不得影响既有 /internalize /reason /reset（异常隔离 + 4xx/5xx 约定）。
