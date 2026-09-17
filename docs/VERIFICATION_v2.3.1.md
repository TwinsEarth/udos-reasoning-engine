# UDOS 推演引擎 v2.3.1 验证报告（本沙箱实测）

主题：**可校准、长时程更稳健的可信推演**。在 v2.2.1（77 测试）之上新增 F1 保序置信校准、
F2 多时域损失、F3 split-conformal 经验预测区间、F4 服务/持久化；内部先 2.3.0(minor) 后 2.3.1(patch)。

## 1. 测试与回归
- **92 passed**：v2.2.1 的 77 项原样全绿 + v2.3 新增 15（PAVA/Spearman、保序单调/clip/退化、
  多时域权重与"默认 front 逐位复现 2.2.1"锚点、对称 conformal 半宽与保守上分位、校准不入
  state_dict、存载往返与旧档兼容、/calibrate /checkpoints /load 与 400/409）。
- 覆盖率：calibration 99%、persistence 100%、training 96%、evaluation 95%。

## 2. F1 置信保序校准（确定性后处理，本代稳健增益之一）
- 方法：校准集误差均值 scale 将逐样本误差映为经验精度 acc=exp(-err/scale)，自实现 PAVA 学
  单调非降"原始置信→经验精度"映射；零可学参数、不重训。
- 正式件独立测试集回归 ECE 0.397→0.069（降约 5.7×），校准集自身 0.364→0.076（4.8×），
  num_segments=4、ranking_informative=true。
- **诚实局限（不夸大）**：校准后置信-误差 Spearman 仍为正（0.13→0.13 量级），即 CTM 同步
  certainty 的逐样本排序方向仍是反的（越自信误差可越大，raw 高/低档误差比 1.87、非单调）；
  保序主要把"过度自信的数值刻度"压回经验精度，并**不把坏排序翻正**。欠训练时 PAVA 退化为
  常量（num_segments=1、ranking_informative=false），代码显式标注"只修数值水平、不制造排序"。

## 3. F2 多时域损失：从"拟设默认"到反证回退（完整证据链，负面结果如实保留）
- TrainConfig.step_weight_scheme ∈ front(默认)/uniform/back，三方案归一、H=1 恒为[1]。
- 口径A（无早停/45ep/3 种子，ablation_horizon_weight.py）：uniform 远期 tail 3/3 低于
  front（均值 −60.2%），最初据此拟改默认。
- **口径B（早停/正式 build 数据划分，sensitivity_step_weight_pipeline.py）反证**：front
  反超 uniform（uniform 单步 +39.6%、tail +34.0%），cross_consistent=false。
- 裁决：权重优劣对是否早停/数据划分口径敏感、不稳健（与 v2.2 SS 同型）。按"无同合同多口径
  证据不授权改默认"纪律，**默认回退 front=逐位复现 2.2.1（有专门等价测试锁定），uniform/back
  一律 opt-in**；正反结果全部留档（两份 JSON），不宣称 uniform 稳健增益。

## 4. F3 split-conformal 经验预测区间（方法学纠错过程）
- 初版用"有符号残差双侧分位"，实测覆盖率仅 0.78–0.86（自回归残差有偏/重尾 + 6 个尺度不同的
  物理维混合 + 独立集再切分致样本不足），系统性 undercover。
- 经三种构造同集对照定位：改为**标准回归 conformal——逐步、逐维对绝对残差 |y-ŷ| 取 0.9
  保守上分位作对称半宽 [ŷ-q,ŷ+q]**（小样本向上取整不偏窄；独立校准/测试集整份使用不再切分）。
- 改造后独立 iid 集各步覆盖率 0.894/0.894/0.895/0.897 ≈ 名义 90%，宽度随步非减；正式件数值
  见 benchmarks/results/training_v2.3.1.json。
- 局限：split-conformal 的覆盖保证依赖校准/测试交换性（近似同分布），不保证分布外覆盖；
  远期步更宽如实反映误差累积。

## 5. F4 服务与持久化（真实起 HTTP 验证）
- POST /calibrate（未训练 409、越界 400；独立集 rawECE→calECE 显著下降、informative=true）、
  GET /checkpoints（列 3 件）、POST /load（200、source_version=2.3.1、calibrated=true；
  `../evil.pt` 穿越 400）。
- 校准器+残差半宽随 checkpoint 存取（bundle 可选 calibration 键），v2.2.1/v2.1 旧档无该键时
  is_calibrated=false、正常载入（向后兼容有测试）；保存/重载前向逐位一致。

## 6. 正式件与复现
- checkpoints/predictor_v2.3.1.pt：52,191 参数（校准器外挂、不入 state_dict），默认 front
  训练；指标与校准/区间结果落 benchmarks/results/training_v2.3.1.json，reload_consistent=true。
- 一键复现：`make ckpt23`（训练+校准+评估+落件）、`make horizon-ablation`（口径A）、
  `python3 scripts/sensitivity_step_weight_pipeline.py`（口径B 反证）。
- 运行时：Python 3.12、torch 2.14 CPU、2 线程，无 GPU/无 docker 依赖。

## 7. 限制（不夸大）
合成参数化动力学，不等同真实物理精度；校准修数值刻度不修坏排序；conformal 仅同分布保证；
多时域权重收益口径敏感故不改默认；GPM 仍零反向传播、端到端联合微调留待后续。
