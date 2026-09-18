# UDOS Engine v7.0.3 架构（统一世界模型内核）

证据分级贯穿全文：**verified**（本机 CPU 固定 seed 实测，可复跑）｜**cpu-proto**（CPU 小规模代理，未等价 GPU/大规模）｜**unverified**（需 GPU/云/密钥等闸门，未做）。

> **v7.0.3（补丁）相对 v7.0.2 的根因修复**：v7.0.2 已能精确反演三维加速度，却只把它当通用上下文喂给 GRU，仍由 GRU 学着对恒定加速度做二次积分——探针显示 accel 误差几乎全是**随视界二次增长的位置误差**。v7.0.3 新增**解析运动学积分 + 独立学习门控的混合预测头**：`nxt = learned + g·(analytic−learned)`，`analytic` 用初始窗固定的 `a_lin` 做 `v+=a·dt, p+=v·dt+½a·dt²`，门控 `g=ca_conf·tanh(MLP(kin)/2)` 只吃确定性运动学特征（末层零初始化，未训练严格恒等）。盲路径 held-out 总误差 **0.0248→0.0104（−58%）**、accel **0.0693→0.0141（−80%）**，spring/collision 不退化。

> **v7.0.2（补丁）相对 v7.0.1 的根因修复**：新增**确定性可观测运动学通道**（`udos7/kinematics.py`），把“状态里本就可直接反演的量”（三维加速度向量、弹簧角频率、碰撞速度跳变）从“盲估计隐藏标量”改为“确定性计算 + 零初始化学习注入”；并修两个真实 bug（服务校准器缓存键漏 horizon、conformal 在请求短视界时的张量切片）。盲路径 held-out 总误差 **0.0459→0.0248（−46%）**，全部数字见 VERIFICATION.md。

## 1. 为什么重写（v5.5.5 → v7）

v5.5.5 的双引擎存在一条**名不副实的链路**：

- 主预测器 `PhysicsPredictor`（已训练）真实输出位置/速度；
- 场景内化链 `SceneReasoningEngine` 用 GPM 生成 LoRA 并注入演示基座 `TinyBaseModel`，但 `reason()` **从不调用被注入模型的 forward**；物理预测也没接收 GPM 场景嵌入。
- 即“场景内化为 LoRA”是一条死路，且与主预测器割裂成两套图。

v7 不再保留两套并行链路，而是合并为**单一推理图**，并删除未训练演示模型、LoRA、TinyBaseModel 等死代码（测试断言包内不得出现这些模块名）。

## 2. 单一推理图

```
观测窗口 X [B,W,6]
   │  obs_encoder（逐帧 Linear）
   ▼
统一场景通道 SceneChannel（在同一 ctx 空间求和）
   ├── 显式参数 explicit P（已知真值；NaN 槽=该槽缺失→回退估计）
   ├── 估计器 SceneEstimator：窗口 → p_hat（盲路径的隐藏参数估计）
   ├── 场景桥 SceneBridge（末层零初始化；v7 保留为可学习场景偏置，
   │                        不再是注入后从不调用的 LoRA）
   └── 确定性运动学通道 kin_encoder（v7.0.2；末层零初始化）：
         窗口 → kinematic_features（不依赖训练的可观测运动学，KIN_DIM=11，
         含 v7.0.3 的 ca_conf 恒定加速度一致性置信）
   ▼  ctx [B,W,scene_dim]（逐帧相加）
GRU 时序核（唯一循环核，2 层）
   ▼  取最后一帧 h
学习残差臂：learned = last_state + Δ(h)        （Δ 头零初始化）
解析积分臂（v7.0.3）：analytic = 恒定加速度积分（a_lin 固定自初始窗）
混合头：next = learned + g·(analytic−learned)
        g = ca_conf · tanh(MLP(kin)/2)，门控为只吃 kin 的独立小头（末层零初始化）
   ▼
next [B,6]；rollout 自回归滑动（场景 ctx、a_lin、ca_conf 由初始窗算一次并固定）→ traj [B,H,6]
   ▼
不确定性（不在确定性核里伪造概率）
   ├── split conformal 预测区间（按 α 真分水平，marginal + per-step）
   └── 参数蒙特卡洛扇形 ⊕ 残差 conformal 地板（包络）
```

关键不变量（均有测试守护，见 `tests7/test_m1_unified_graph.py`）：

1. **未训练即恒等预测**：残差头零初始化时 `predict(X) == X 最后一帧`；
2. **场景桥零初始化**：初始不扰动预测；
3. **唯一 GRU**：全模型仅一个 `nn.GRU` 实例；
4. **NaN 是“缺槽”哨兵，inf 才拒绝**：显式参数某槽为 NaN 时该槽回退估计器，整槽 inf 报错；
5. **LoRA/基座注入差为零的死路已删除**：场景影响必须真实流经前向。

## 3. 数据契约（`udos7/contracts.py`、`udos7/dynamics.py`）

- 状态 6 维：`[px,py,pz, vx,vy,vz]`；场景参数 4 维：`(v0, accel_a, spring_omega, other_v2)`。
- 四类动力学：`uniform / accel / spring / collision`。
- **真三维**：非碰撞类沿随机三维单位方向运动，px/py/pz 都非平凡；碰撞仍为 x 轴两体（y/z 平凡，诚实标注）。
- **轨迹级** train/val/test/calib 四分，独立种子 42/1337/2026/314；轨迹 id 全局偏移不重叠，窗口只在轨迹内切，杜绝窗口泄漏。
- 默认每类 64 训练轨迹（train 256 条），val/test/calib 各 128 条；每条轨迹切 5 个窗口（W=6,H=4,margin=4,T=14）。
- 有效参数槽 `ACTIVE_SLOTS`：uniform 仅 v0；accel v0+a；spring ω；collision v0(=v1)+other_v2。

## 4. 训练（`udos7/train.py`）

- 模型 `WorldModelCore`：逐帧观测编码 + 场景通道 + 2 层 GRU + 残差解码。
- 损失 = 单步 MSE + 0.7×rollout MSE
  + 0.3×**有监督参数辨识**（仅有效槽，把 p_hat 拉向真值）
  + 0.1×**可观测性头**（拟合 `exp(-|p_hat-P|/0.5)`，误差越小越接近 1）；
- AdamW + 余弦退火 + grad clip 5.0；scene dropout 0.5 训练盲路径。
- **档位由 held-out 决定**：在 64/128/256 hidden 三档中，选验证准则最小、且不劣于最优 3% 的最小档。v7.0.3 数据选中 hidden=256，**967,796 可学习参数（≈0.97M，其中运动学编码器 352 个、解析积分门控小头仅 582 个）**；三档盲路径 test MSE 分别为 0.0153/0.0117/0.0104，选档规则按 val 准则（非 test）选定 256。

## 4b. 解析运动学积分 + 学习门控混合头（v7.0.3 根因修复）

**根因证据（分步探针，verified）**：v7.0.2 的 accel 误差几乎全是位置误差且随视界二次增长——oracle accel 分步 MSE `[0.0088,0.042,0.114,0.245]`，位置 MSE 0.187 vs 速度 0.018。速度已准，但通用残差 GRU 难以对恒定加速度做精确二次积分。确定性天花板：用观测 `a_lin` 解析积分，uniform/accel MSE **精确为 0**，spring 1.98、collision 0.29（恒定加速度模型在这两类失配）。

**混合头**（`WorldModelCore._step`）：

- 学习臂 `learned = last + Δ(h)`（零初始化残差，沿用 v7.0.1/2）；
- 解析臂 `analytic`：`v'=v+a·dt，p'=p+v·dt+½a·dt²`，`a` 取初始窗 `a_lin`；
- 门控 `g = ca_conf · tanh(MLP(kin)/2)`，输出 `next = learned + g·(analytic−learned)`。

**三个被同合同 A/B 否决的错误方案**（详见 VERIFICATION.md，均有实测）：

1. `clamp(linear(h),0,1)` 零初始化：raw=0 恰处 clamp 边界、边界梯度为 0，门**冻死在 0**（两臂逐位相同）→ 改用 0 点可导的 `tanh`；
2. 门读共享 GRU 隐状态 h：门在 accel 饱和后改变共享表征梯度，**spring 盲路径 0.044→0.58 崩坏** → 门改为只吃确定性 `kin` 的独立小头，与共享路径解耦；
3. rollout 每步重测 ca_conf：spring 预测帧滑窗后 ca 翻转误开 → `a_lin/ca_conf` 与场景 ctx 一样由初始窗算一次并固定。

**ca_conf（恒定加速度一致性置信，kinematics 第 11 维）**：速度对时间线性拟合 R² × (1−ω_valid)(1−jump_valid)。实测 uniform/accel 中位与 p05 均=1.0、spring/collision=0；单用 R² 不够（spring .924、collision .860），必须乘弹簧/碰撞有效标志。门控末层零初始化 ⇒ 未训练 `tanh(0)=0` 即 g≡0，**严格恒等不变量保持**（atol 1e-6）。

**为何保留可学习门而非硬 g=ca_conf**：collision 中“窗内无跳变（ca=1）但视界内将碰撞”的窗，恒定速度解析解会穿过碰撞点，实测门在 collision 上仅学到 0.15 的谨慎信任（spring 严格 0）。

## 5. 隐藏参数可辨识性与确定性运动学通道（v7.0.2 根因修复）

### 5.1 学习型估计器的可辨识性（保留，诚实负结果）

除估计 MAE 外，报告 **identifiability_skill = 1 − MAE估计/MAE均值预测器**：≤0 表示从窗口反推不比直接猜均值强。v7.0.1 test 实测（verified）：

| 参数 | 估计 MAE | 均值基线 MAE | skill | 解读 |
|---|---|---|---|---|
| v0（初速度） | 0.66 | 0.97 | +0.31 | 部分可观 |
| accel_a（**带符号标量**加速度） | 0.94 | 0.61 | **−0.53** | **该标量口径不可辨识（见 5.2）** |
| spring_omega（ω） | 0.15 | 0.26 | +0.40 | 部分可观 |
| other_v2（被撞速度） | 0.05 | 0.24 | +0.79 | 滑窗含碰撞后帧时部分可观 |

低 MAE 不等于可辨识（参数集中在 0 附近时猜均值也能低误差），故必须看 skill。

### 5.2 根因：负结果是“表征口径”错误，不是“加速度不可观测”

v7.0.1 让估计器输出**沿未知随机三维方向 d 的带符号标量 a**，但 d 与 a 的符号不可分离：把 d 翻转、同时把 (v0,a) 反号，得到完全相同的轨迹。该标量天然弱可辨识，skill 为负是**口径选错**。

而状态向量本身含三维速度，**恒定加速度就是速度对时间（秒）的最小二乘斜率，三维加速度向量可从观测窗精确反演**，无需学习、无需猜测。v7.0.2 据此新增确定性通道（`udos7/kinematics.py`，v7.0.3 起 KIN_DIM=11）：

| 特征 | 计算 | 可观测性口径 | held-out 实测（verified） |
|---|---|---|---|
| 窗心速度 `v_c[0:3]` | 窗内速度均值 | 直接量 | — |
| 三维加速度向量 `a_lin[3:6]` | 速度对秒的最小二乘斜率（分母用 `Σ(t−t̄)²`） | 恒定加速度精确可反演 | accel 向量恢复 **skill=1.0、MAE≈0**；uniform 误报范数 0 |
| 弹簧角频率 `omega[6]`+`valid[7]` | 位置 PCA 主轴投影成标量 q，对内点做**带截距** `q̈=b1·q+b0`（b1=−ω²）；门限 `w2>0.45² & r_spring<0.25 & r_spring<r_const` | 短窗可反演 | 召回 **1.0**、有效样本 MAE≈0.023、非弹簧误报 **0** |
| 碰撞速度跳变 `jump[8]`+`valid[9]` | x 轴速度差 `max|Δvx|>3·median+0.35` | 仅跨碰撞帧的窗可检（其余窗本就近似匀速） | 跨碰撞窗检出 0.74、非碰撞误报 **0** |
| 恒定加速度置信 `ca_conf[10]`（v7.0.3） | 速度线性拟合 R² × (1−ω_valid)(1−jump_valid) | 恒定加速度模型成立度 | uniform/accel=1.0（p05 亦 1.0）、spring/collision=0 |

无效特征一律置 0 且对应 valid=0；函数对坏形状、NaN/inf 直接拒绝（有测试守护）。

### 5.3 如何进入推理图（零初始化，不破坏稳定起点）

`SceneChannel` 新增末层**零初始化**的 `kin_encoder: Linear(KIN_DIM, scene_dim)`：接入瞬间对预测零扰动（未训练仍严格恒等，见 `test_kin_encoder_zero_init_keeps_untrained_identity`），训练后才承载可观测运动学。它与显式参数、估计器、场景桥在**同一 ctx 求和**，不产生第二套链路。旧 v7.0.1 checkpoint 的 config 无 `use_kinematics`，加载时默认不构建该通道（state_dict 不错位，向后兼容，有测试守护）。v7.0.3 把 KIN_DIM 从 10 增到 11（新增 ca_conf），加载 v7.0.2 权重时对 `kin_encoder` 末列**零列填充**、门控小头缺失零初始化，行为与 v7.0.2 逐位一致（见 `test_v702_checkpoint_loads_with_zero_init_gate`）。

**口径纪律**：确定性通道只承载“状态里本就可直接反演”的量；真正的隐含量（碰撞发生时刻、被撞体参数、未来输入）仍交给估计器并保留可观测性标注，不把确定性与学习性混为一谈。

## 6. 不确定性（`udos7/uncertainty.py`）

确定性核只输出点预测，概率全部交给校准层：

- **split conformal**：calib 集按每个 α 单独取有限样本修正分位数，给 marginal 与 per-step 带宽。根治旧版“名义 80/90/95 经验覆盖全相同（0.8988）”的 α 失效。v7.0.2 修复“请求 horizon 短于校准视界时 pred(H) 与 Y(4) 维度不匹配”的切片 bug，并在 `fit` 显式拒绝 horizon>校准视界。
- **参数蒙特卡洛扇形**：对估计参数加 calib 残差自助，多次 rollout 得 p10/p50/p90；乘法 conformal 膨胀。
- **残差 conformal 地板（包络）**：裸参数扇形只刻画参数不确定性、覆盖不了自回归结构误差（未包络时仅约 0.27–0.74）。扇形与残差带取包络后，覆盖率不低于有边际保证的残差带。
- entropy 类 certainty 与校准概率严格分离，不把“网络自信”当正确概率。

## 7. HTTP API（`udos7/server.py`，前缀 `/api/v7`）

- `GET /api/v7/health`：版本/模型加载/证据分级；
- `POST /api/v7/predict`：`{window, explicit?, horizon?}` → 点轨迹；
- `POST /api/v7/interval`：`{window, alpha, explicit?, horizon?}` → 校准区间；
- `GET /api/v7/metrics`：最近一次 `v7_verification.json`。
标准库 `ThreadingHTTPServer`，零额外依赖；conformal 首次请求懒校准并缓存。v7.0.2 修复缓存键：由仅按 `use_explicit` 改为按 `(use_explicit, horizon)`，避免不同视界串用同一组 `q_per_step` 带宽（有测试守护）。

## 8. 证据分级与 M5 闸门（未做即标 unverified）

| 项 | 状态 |
|---|---|
| 统一推理图、三分/四分数据、真实训练、档位选择、参数辨识、conformal/扇形覆盖率、HTTP | **verified**（见 VERIFICATION.md） |
| MuJoCo CPU pip 最小虚拟小鼠因果模型（动作/体重干预因果改变轨迹） | **cpu-proxy**（`proxies/mujoco_mouse/`，非 DeepMind virtual rodent，无 RL/神经对齐） |
| vLLM KV-offload/压缩真实 TTFT/吞吐、NEURON/CoreNEURON 验 DHS、MuJoCo-MJX、0.5B/5B 端到端 | **unverified，需 GPU/HPC（约 ¥35,400/月档），本环境无 GPU/docker/sudo** |
| 多 LLM 交叉打分 | **unverified，需多供应商 key**；无 key 时只能确定性规则/CPU 小模型且不等同商用 LLM |
| PostgreSQL/并发压测/CI-CD/监控/灾备/双云部署/域名/ICP/HTTPS | schema 与脚本可本地验；**实名备案与双云真实部署需用户闸门（数周）** |
| AGI/ASI 倒计时网站 | 独立 **v6.2 线**，不在纯引擎 v7 范围内 |

## 9. 目录

```
udos7/        统一内核（contracts/dynamics/kinematics/scene/model/train/metrics/
              uncertainty/persistence/server）
tests7/       M1 统一图 / M2 训练 / M3 不确定性 / M4 API / v7.0.2 运动学 /
              v7.0.3 解析积分门控（共 33 测试）
scripts7/     train_v7（档位选择+训练）/ verify_v7（综合门禁）/ service_smoke
checkpoints7/worldmodel_v7.0.3.pt（hidden256, 967,796 参数，use_kinematics=True）
              worldmodel_v7.0.2.pt（旧档，加载时零列填充 KIN_DIM 10→11）
              worldmodel_v7.0.1.pt（旧档保留，加载时自动关闭运动学通道）
reports7/     收敛曲线、v7_verification.json、HTTP 冒烟、训练/验证日志
docs7/        本文档 + VERIFICATION.md + LEGACY_MIGRATION.md
udos/         v5.5.5 冻结 legacy（弃用对照，不再演进）
```
