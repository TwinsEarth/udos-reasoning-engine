# UDOS Engine v7.0.1 架构（统一世界模型内核）

证据分级贯穿全文：**verified**（本机 CPU 固定 seed 实测，可复跑）｜**cpu-proto**（CPU 小规模代理，未等价 GPU/大规模）｜**unverified**（需 GPU/云/密钥等闸门，未做）。

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
   └── 场景桥 SceneBridge（末层零初始化；v7 保留为可学习场景偏置，
                            不再是注入后从不调用的 LoRA）
   ▼  ctx [B,W,scene_dim]（逐帧相加）
GRU 时序核（唯一循环核，2 层）
   ▼  取最后一帧
残差解码器：next = last_state + Δ(h)   （Δ 头零初始化）
   ▼
next [B,6]；rollout 自回归滑动 → traj [B,H,6]
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
- **档位由 held-out 决定**：在 64/128/256 hidden 三档中，选验证准则最小、且不劣于最优 3% 的最小档。数据选中 hidden=256，**966,830 可学习参数（≈0.97M）**，落在事前诚实预估的 0.5–2M 带内（非预设、非吹嘘倍数）。

## 5. 隐藏参数可辨识性（诚实负结果）

除估计 MAE 外，报告 **identifiability_skill = 1 − MAE估计/MAE均值预测器**：≤0 表示从窗口反推不比直接猜均值强。test 实测（verified）：

| 参数 | 估计 MAE | 均值基线 MAE | skill | 解读 |
|---|---|---|---|---|
| v0（初速度） | 0.66 | 0.97 | +0.31 | 部分可观 |
| accel_a（加速度） | 0.94 | 0.61 | **−0.53** | **6 帧窗口不可辨识** |
| spring_omega（ω） | 0.15 | 0.26 | +0.40 | 部分可观 |
| other_v2（被撞速度） | 0.05 | 0.24 | +0.79 | 滑窗含碰撞后帧时部分可观 |

低 MAE 不等于可辨识（参数集中在 0 附近时猜均值也能低误差），故必须看 skill。

## 6. 不确定性（`udos7/uncertainty.py`）

确定性核只输出点预测，概率全部交给校准层：

- **split conformal**：calib 集按每个 α 单独取有限样本修正分位数，给 marginal 与 per-step 带宽。根治旧版“名义 80/90/95 经验覆盖全相同（0.8988）”的 α 失效。
- **参数蒙特卡洛扇形**：对估计参数加 calib 残差自助，多次 rollout 得 p10/p50/p90；乘法 conformal 膨胀。
- **残差 conformal 地板（包络）**：裸参数扇形只刻画参数不确定性、覆盖不了自回归结构误差（未包络时仅约 0.27–0.74）。扇形与残差带取包络后，覆盖率不低于有边际保证的残差带。
- entropy 类 certainty 与校准概率严格分离，不把“网络自信”当正确概率。

## 7. HTTP API（`udos7/server.py`，前缀 `/api/v7`）

- `GET /api/v7/health`：版本/模型加载/证据分级；
- `POST /api/v7/predict`：`{window, explicit?, horizon?}` → 点轨迹；
- `POST /api/v7/interval`：`{window, alpha, explicit?, horizon?}` → 校准区间；
- `GET /api/v7/metrics`：最近一次 `v7_verification.json`。
标准库 `ThreadingHTTPServer`，零额外依赖；conformal 首次请求懒校准并缓存。

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
udos7/        统一内核（contracts/dynamics/scene/model/train/metrics/
              uncertainty/persistence/server）
tests7/       M1 统一图 / M2 训练 / M3 不确定性 / M4 API（16 测试）
scripts7/     train_v7（档位选择+训练）/ verify_v7（综合门禁）/ service_smoke
checkpoints7/worldmodel_v7.0.1.pt（hidden256, 966,830 参数）
reports7/     收敛曲线、v7_verification.json、HTTP 冒烟、训练/验证日志
docs7/        本文档 + VERIFICATION.md + LEGACY_MIGRATION.md
udos/         v5.5.5 冻结 legacy（弃用对照，不再演进）
```
