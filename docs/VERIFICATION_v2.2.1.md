# UDOS v2.2.1 验证报告（测试 · 实验 · 回归 · 发布判断）

- 版本：**2.2.1**（能力里程碑 2.2.0 + 补丁 2.2.1，对外发布号 2.2.1）
- 基线：v2.1.0（64 测试、覆盖率约 95%、场景条件增益 9.93×）
- 环境：Python 3.12.11、torch 2.14.0+cpu、2 线程、无 GPU、无 docker daemon（容器仍以
  等价进程方式验证，镜像构建沿用部署手册，不在本环境实建）
- 设计判据：`docs/VERSION_PLAN_2.2.md`；路线：`docs/ROADMAP.md`；变更：`CHANGELOG.md`

---

## 1. 版本范围与设计→实现映射

v2.1 报告 §7 自陈缺口 G1–G6。本轮处理可在沙箱闭环的 G1–G4；G5（真实数据）、G6
（docker daemon）属环境/数据边界，继续如实保留。

| 特性 | 缺口 | 实现位置 | 形态 |
|---|---|---|---|
| F1 Scheduled Sampling | G1 rollout 误差累积 | `training.py` 滚动窗口 + Bernoulli 自回归喂入、`TrainConfig.ss_*` | **opt-in，默认关闭** |
| F2 训练治理 | G2 缺早停/确定性 | `set_seed`、`patience/min_delta`、最佳权重恢复、`TrainHistory` | 默认关闭早停=同 2.1 |
| F3 评估增强 | G3 缺校准视角 | `evaluation.py` 新增 `rollout_growth_x`、`confidence_stratification` | 纯增量键 |
| F4 模型管理 | G4 不能在线评估/落盘 | `server.py` `POST /evaluate`、`POST /save`（409 + 路径白名单/不可逃逸校验） | 新接口 |
| 2.2.1 补丁 | 健壮性/一致性 | 空集与除零保护、版本号单一来源、冒烟补接口、文档对齐 | patch |

---

## 2. F1 Scheduled Sampling：完整 A/B 证据与诚实裁决

### 2.1 机制与兼容锚点
第 h 步预测后，以概率 `p_ss(epoch)=ss_max·clip((ep-ss_start)/ss_warmup,0,1)` 用模型
自身上一步预测（`.detach()`，不跨步反传）替代真值拼回滚动窗口。**`ss_max=0` 时滚动
窗口与 v2.1 的 `cat` 真值 teacher-forcing 逐步等价**，由
`test_ss_zero_numerically_equals_teacher_forcing` 以旧实现为 oracle 逐位锁定
（atol=1e-7）；`horizon=1` 时确定无操作；开启时 loss 可正常反传。

### 2.2 同合同 A/B（同种子/同数据/同初始 state_dict/同 epoch，仅 SS 不同）
全部对照由 `scripts/ablation_scheduled_sampling.py` 可复现，原始结果落
`benchmarks/results/ss_ablation_v2.2.0.json`。

**（a）正式复现，n=36/类、45 epoch、H=4、配置 ss0.30/start15/warmup25（191s）**

| 种子 | 方法 | single MSE | rollout 逐步 MSE | step3-4 均值 | 对 TF |
|---|---|---|---|---|---|
| 42 | teacher-forcing | 0.1335 | 0.134 / 0.168 / 0.221 / 0.300 | 0.2605 | — |
| 42 | SS | **0.0892** | **0.089 / 0.100 / 0.124 / 0.173** | **0.1485** | **改善 ≈43%** |
| 7 | teacher-forcing | 0.0939 | 0.094 / 0.120 / 0.162 / 0.236 | 0.1990 | — |
| 7 | SS | 0.1047 | 0.105 / 0.148 / 0.244 / 0.391 | 0.3175 | **恶化 ≈60%** |

**（b）激进配置失败模式**，ss_max=0.6/warmup10（n=28、40ep）：TF single 0.123 vs
SS 0.183，step3-4 均值恶化 32.9%——模型尚未学好即被自身误差污染（exposure bias 反噬）。

**（c）超参扫描**（n=28、40ep）：ss0.15/w20 仍偏差；ss0.30/w20 的 growth 更缓但
single 变差；ss0.30/start15/warmup25（先学好再缓慢暴露）单次最好。

**（d）更长时域 H=6 双种子**：seed42 TF 反而略优，seed7 SS 中段略好但 single 更差，
方向依旧不稳。

### 2.3 裁决：opt-in，默认关闭，不宣称确定增益
A/B 在不同种子与 horizon 下**方向反转**：SS 的收益依赖"数据规模/课程/初始条件"的良好
配合，在本沙箱小模型 + 合成数据下不构成稳健提升。据 `VERSION_PLAN` §3.2/§9 回滚条款：

- SS 作为**机制正确、可配置、带复现实验**的受控特性交付，服务与训练默认 `ss_max=0`；
- 不把任何一次有利种子的增益阈值写进单测（避免 flaky），单测只锁机制；
- 报告如实保留正反两面结果。更长时域的稳健收益需更大数据/自适应课程/闭环再训练。

---

## 3. F2 训练治理

- `set_seed(seed)` 统一确定性入口；`test_set_seed_deterministic_short_run` 锁两次同种子
  训练 loss 曲线逐位一致（<1e-9）。
- 早停：`patience/min_delta` 触发 `stopped_early`、记录 `best_epoch/best_eval` 并恢复最佳
  权重。`test_early_stop_triggers_and_records_best`（极大 min_delta 强制无改善→提前停、
  轮数<上限）；默认 `patience=None` 时 `test_early_stop_disabled_runs_full_by_default`
  锁"跑满 epochs、不记 best"，即与 v2.1 行为一致。
- 正式构建中早停在线（patience=12），60 轮内 best_epoch=57、未触发提前停止（末期仍有
  改善），说明阈值不冒进。

## 4. F3 评估增强与"置信未校准"发现

- 新增 `rollout_growth_x = roll[-1]/roll[0]`（累积率）与 `confidence_stratification`
  （按最终 tick certainty 等频三档，输出各档置信/MSE、高/低档误差比、是否单调递减）；
  旧返回键全部保留，由 `test_evaluation_new_keys_and_old_keys_preserved` 锁定，小样本/
  全零误差等退化情形由 `test_confidence_stratification_handles_tiny_and_zerodivision`
  锁定不崩。
- **发现（如实呈现）**：无论是 v2.1 预置件还是 v2.2.1 新件，置信分层都**非单调**——
  最高置信档误差并非最低（v2.2.1 件高/低档误差比 1.80）。这说明 CTM 的 certainty 主要
  度量内部同步收敛程度，**不是经过校准的预测置信度**。该指标作为诊断暴露了问题，而非
  粉饰；正式概率/区间预测与 ECE 校准列入后续路线。

## 5. F4 模型管理服务

| 行为 | 证据 |
|---|---|
| 未训练时 `/evaluate`、`/save` 返回 **409** | service 层 `ServiceNotReady` + 真实 HTTP `test_service_http_409_when_untrained`；冒烟"未训练409" |
| 训练后 `/evaluate` 返回新指标 | `test_service_evaluate_save_flow_and_409`、冒烟"在线评估+置信分层" |
| `/save` 落盘且可重载、字节数>0 | 同上；正式件 `predictor_v2.2.1.pt` reload 逐位一致 |
| 路径穿越防护 | 字符白名单清洗 + "最终绝对路径必须仍在 checkpoints 内"根本校验；`../escape`、`a/b` 被无害化，测试与冒烟双证 |
| 参数越界→400 | `test_service_invalid_args_400`（n_per_kind/horizon/ss_max/空名） |
| 单请求异常不崩进程 | do_POST 分层捕获 409/400/500，冒烟全程进程存活 |

## 6. v2.2.1 正式构建件（可复现）

`python3 scripts/build_v221_checkpoint.py`（默认 teacher-forcing + 早停，87.8s）：

| 指标 | 值 |
|---|---|
| 参数量 | 52,191（与 v2.1 同构，旧档可载） |
| 单步 MSE | 0.044571 |
| 相对未训练 / 朴素基线 | 改善 **103.92× / 4.04×** |
| 场景条件增益 condition_gain_x | **14.102×**（v2.1 为 9.93×，训练更充分） |
| 4 步自由 rollout | 0.0446 / 0.0537 / 0.0786 / 0.1110，累积率 **2.49×**（v2.1 同口径约 3.6×） |
| 置信分层 | 非单调，高/低档误差比 1.80（见 §4） |
| 落盘/重载 | `checkpoints/predictor_v2.2.1.pt`（228,171 字节），reload 前向一致 |
| 指标 JSON | `benchmarks/results/training_v2.2.1.json` |

## 7. 向后兼容证据（不删、不放宽旧契约）

- v2.1 的 **64 个旧测试原样保留且全部通过**；新增 13 个，**总计 77 passed**。
- `ss_max=0`、`patience=None` 默认组合与 v2.1 数值等价（有专门等价测试）。
- evaluate 只新增键、不改旧键；服务旧路由字段语义不变；v2.0/v2.1 checkpoint 仍可加载，
  新件亦可被旧 loader 读取（bundle 结构向后兼容）。
- 版本号三处统一：`udos/__init__.py`、`pyproject.toml`、`server.health`（改为引用
  `__version__` 单一来源）；镜像 tag / compose / Dockerfile / Makefile 同步 2.2.1。

## 8. QA 收口

- **单元/契约/集成**：`pytest tests/` → **77 passed**。
- **覆盖率**（pytest-cov）：核心模块 dynamics 99%、persistence 100%、training 95%、
  evaluation 94%、ctm 96%、gpm 99%、reasoning 94%、server 85%；核心约 **95%**
  （debug 面板为手动工具不计入自动化口径，与 v2.1 一致）。
- **性能守卫**：`make guard`（10 次重复）通过，median ctm-small 6.15ms、gpm 内化
  1.91ms、e2e 8.35ms、上游 CTM 7.79ms，峰值 RSS 285.5MB，未较 v2.1 退化。
- **端到端冒烟**：`scripts/smoke_test.py` 真实起 HTTP 服务，**15/15 通过**（v2.1 为 11，
  新增未训练409、在线评估、落盘、名称无害化），并验证 `--checkpoint predictor_v2.2.1.pt`
  启动即带训练态。

## 9. 发布判断与复现命令

**结论：go（可发布）**。交付物为可用、可复现、向后兼容的 2.2.1；其中 F2/F3/F4 为确定
落地能力，F1 为带诚实边界的 opt-in 特性。已知限制（合成数据、certainty 未校准、SS 收益
条件依赖、无真实容器实建）均已在 README/ROADMAP/本报告显式标注，不影响发布判定。

```bash
pip install -r requirements.txt
python3 -m pytest tests/ -q                     # 77 passed
python3 -m benchmarks.benchmark --repeats 10 --guard
python3 scripts/smoke_test.py                   # 15/15
python3 scripts/build_v221_checkpoint.py        # 产出正式 checkpoint + 指标 JSON
python3 scripts/ablation_scheduled_sampling.py  # 复现 F1 A/B (约 3 分钟, --quick 更快)
# 开启 opt-in SS: 训练接口 {"ss_max":0.3,"ss_start":15,"ss_warmup":25} 或 build --ss
```
