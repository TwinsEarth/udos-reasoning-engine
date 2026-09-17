# UDOS 推演引擎统一编排入口 (测试 / 验证 / 部署 三线协同)
PY ?= python3
.PHONY: install test cov bench perf-ab feature-bench feature-bench27 guard train demo4 demo5 ckpt ckpt22 ckpt23 ckpt24 ckpt25 ckpt252 ckpt260 ckpt262 ckpt270 ckpt273 ckpt280 ckpt290 ckpt300 ckpt303 ckpt310 ckpt320 ckpt330 ckpt333 ckpt340 ckpt345 ckpt350 ckpt360 ckpt370 ckpt380 ckpt386 ckpt390 ckpt399 eval5d ss-ablation horizon-ablation calib-ablation noise-ablation conformal-levels hybrid-ablation serve smoke docker-build docker-run clean wla-sparse-ab wla-flow-ab wla-xembodiment-ab wla-multitask-matrix ckpt419 ckpt420 self-train-ab ckpt429

install:
	$(PY) -m pip install torch --index-url https://download.pytorch.org/whl/cpu
	$(PY) -m pip install -r requirements.txt

## 测试线
test:
	$(PY) -m pytest tests/ -q

cov:
	$(PY) -m pytest tests/ -q --cov=udos --cov-report=term-missing

## v3.3.4 性能优化单 delta A/B 裁决 (C1 loop 复用 target_state + C2 max_shard 64)
perf-ab:
	$(PY) scratch/ab_c1_loop.py
	$(PY) scratch/ab_c2_shard.py

## 验证线
bench:
	$(PY) -m benchmarks.benchmark --repeats 20 --json benchmarks/results/baseline.json

## v2.6 新特性推理延迟基准 (hybrid/counterfactual/identify/adaptive/risk)
feature-bench:
	$(PY) scripts/benchmark_v26_features.py --repeats 15 --warmup 3

## v4.5.6 开源资源注册表性能 A/B (装配/probe/列表/转换热路径)
resource-bench:
	$(PY) scripts/bench_resource_v456.py

## v2.7 新特性推理延迟基准 (policy/online/active/lite/hierarchical)
feature-bench27:
	$(PY) scripts/benchmark_v27_features.py --repeats 15 --warmup 3

guard:
	$(PY) -m benchmarks.benchmark --repeats 10 --guard

## v2 学习闭环: 训练演示 (epochs 可覆盖: make train EPOCHS=45)
EPOCHS ?= 40
train:
	$(PY) -m demos.demo4_learn_to_predict $(EPOCHS)
demo4: train

## v2.1 多步/场景条件/一致性演示, 以及训练并落预置 checkpoint
demo5:
	$(PY) demos/demo5_multistep_reasoning.py
ckpt:
	$(PY) scripts/build_v21_checkpoint.py
## v2.2.1 预置 checkpoint (默认 teacher-forcing; 加 --ss 开 opt-in scheduled sampling)
ckpt22:
	$(PY) scripts/build_v221_checkpoint.py
## v2.2 F1 可复现 A/B (SS vs teacher-forcing)
ss-ablation:
	$(PY) scripts/ablation_scheduled_sampling.py
## v2.3.1 预置 checkpoint (含保序校准与经验区间; --quick 小规模)
ckpt23:
	$(PY) scripts/build_v231_checkpoint.py
## v2.4.0 预置 checkpoint (含 OOD 检测器; --quick 小规模)
ckpt24:
	$(PY) scripts/build_v240_checkpoint.py
## v2.5.0 预置 checkpoint (批量推理+缓存版; --quick 小规模)
ckpt25:
	$(PY) scripts/build_v250_checkpoint.py
## v2.5.2 正式发布件 checkpoint (2.5 线终点; --quick 小规模复现)
ckpt252:
	$(PY) scripts/build_v252_checkpoint.py
## v2.6.0 正式发布件 checkpoint (hybrid 模块默认关; --quick 小规模复现)
ckpt260:
	$(PY) scripts/build_v260_checkpoint.py
## v2.6.2 最终正式发布件 checkpoint (2.6 线终点; --quick 小规模复现)
ckpt262:
	$(PY) scripts/build_v262_checkpoint.py
## v2.7.0 正式发布件 checkpoint (MPC 动作优选外挂, 训练口径同 v2.6.2; --quick 小规模复现)
ckpt270:
	$(PY) scripts/build_v270_checkpoint.py
## v2.7.3 最终正式发布件 checkpoint (2.7 线终点; --quick 小规模复现)
ckpt273:
	$(PY) scripts/build_v273_checkpoint.py
## v2.8.0 正式训练件 checkpoint (PhysicalLoop 推理外挂, 训练口径同 v2.7.3; --quick 小规模复现)
ckpt280:
	$(PY) scripts/build_v280_checkpoint.py
## v2.9.0 正式训练件 checkpoint (ActionRetargeting 推理外挂, 训练口径同 v2.8.0; --quick 小规模复现)
ckpt290:
	$(PY) scripts/build_v290_checkpoint.py
## v3.0.0 正式训练件 checkpoint (FutureMultimodalHead 推理外挂, 训练口径同 v2.9.0; --quick 小规模复现)
ckpt300:
	$(PY) scripts/build_v300_checkpoint.py
## v3.0.0.dev6 四代正式件五维评测演化 (UDOS 内部基准, 非 PhysBrain)
eval5d:
	$(PY) scripts/eval5d_evolution.py
## v3.0.3 最终正式训练件 checkpoint (3.0 线终点; 含全部新特性离线 A/B; --quick 小规模复现)
ckpt303:
	$(PY) scripts/build_v303_checkpoint.py
## v3.1.0 正式训练件 checkpoint (3.1 线起点; ActionPiece 推理外挂, 同口径; --quick 小规模复现)
ckpt310:
	$(PY) scripts/build_v310_checkpoint.py
## v3.2.0 正式训练件 checkpoint (3.2 线起点; Ego360 启发数据增强 opt-in, 正式件仍用原始数据; 同口径; --quick 小规模复现)
ckpt320:
	$(PY) scripts/build_v320_checkpoint.py
## v3.3.0 正式训练件 checkpoint (3.3 线起点; 架构精炼 opt-in 默认关=旧架构; 同口径; --quick 小规模复现)
ckpt330:
	$(PY) scripts/build_v330_checkpoint.py
## v3.3.3 最终发布正式件 checkpoint (3.3 线终点; 全部新特性离线 A/B; backcompat 18 件; 同口径)
ckpt333:
	$(PY) scripts/build_v333_checkpoint.py
## v3.4.0 ICM 上下文记忆首训正式件 (同口径; ICM 离线 A/B; backcompat 19 件)
ckpt340:
	$(PY) scripts/build_v340_checkpoint.py
## v3.4.5 ICM 线终点正式件 (同口径; ICM 全特性离线 A/B; backcompat 20 件)
ckpt345:
	$(PY) scripts/build_v345_checkpoint.py
## v3.5.0 SFM 空间基础模型线首训正式件 (同口径; SFM 零外挂自检; backcompat 21 件)
ckpt350:
	$(PY) scripts/build_v350_checkpoint.py
## v3.6.0 PWM 物理世界模型线首训正式件 (同口径; WM 外挂零梯度自检; backcompat 22 件)
ckpt360:
	$(PY) scripts/build_v360_checkpoint.py
## v3.7.0 分层神经控制线首训正式件 (同口径; 大脑/小脑/脊髓外挂零梯度自检; backcompat 23 件)
ckpt370:
	$(PY) scripts/build_v370_checkpoint.py
## v3.8.0 全域调度线首训正式件 (同口径; 多体协同外挂零梯度自检; backcompat 24 件)
ckpt380:
	$(PY) scripts/build_v380_checkpoint.py
## v3.8.6 全域调度线终点正式件 (同口径最终重建; 全部新特性离线A/B; backcompat 25 件)
ckpt386:
	$(PY) scripts/build_v386_checkpoint.py
## v3.9.0 宇树 WLA 类比线首训正式件 (内化冻结 v3.8.6 主权重; ER头+动作三分组+外挂零梯度; backcompat 26 件)
ckpt390:
	$(PY) scripts/build_v390_checkpoint.py
## v3.9.9 宇树 WLA 类比线终点正式件 (全量回归; backcompat 27 件)
ckpt399:
	$(PY) scripts/build_v399_checkpoint.py
## v4.1.0 自规划自监督线起点正式件 (内化冻结 v3.9.9 主权重; 课程生成器+可解性自验证器外挂零梯度; backcompat 28 件)
ckpt410:
	$(PY) scripts/build_v410_checkpoint.py
## v4.1.0.dev2: 自生成课程 vs 固定课程 A/B (落 JSON)
curriculum-ab:
	$(PY) scripts/curriculum_ab_v410.py
## v4.1.0.dev5: 伪标签增益 A/B (有/无自监督损失门, 落 JSON)
pseudolabel-ab:
	$(PY) scripts/pseudolabel_gain_ab_v410.py
## v4.1.2: 自规划自监督线新路径性能基准
perf412:
	$(PY) benchmarks/perf_baseline_v412.py
## v4.1.9 自规划自监督线终点正式件 (内化冻结 v4.1.0 主权重; 全线外挂零梯度; backcompat 29 件)
ckpt419:
	$(PY) scripts/build_v419_checkpoint.py
## v4.2.0 完全自训练线起点正式件 (内化冻结 v4.1.0 主权重; 自生成三元组骨架零梯度; backcompat 30 件)
ckpt420:
	$(PY) scripts/build_v420_checkpoint.py
## v4.2.1 自训练 A/B (自产数据 vs 原始数据) + 收益递减判据
self-train-ab:
	$(PY) scripts/self_train_ab_v421.py
## v4.2.9 完全自训练线终点正式件 (内化冻结 v4.1.0 主权重; 全线外挂零梯度; backcompat 31 件)
ckpt429:
	$(PY) scripts/build_v429_checkpoint.py
## v4.3.0 完全自进化线起点正式件 (基础设施自优化搜索器; 内化冻结 v4.1.0 主权重; backcompat 32 件)
ckpt430:
	$(PY) scripts/build_v430_checkpoint.py
## v4.3 完全自进化线基准 (配置 A/B + 多代曲线 + 停止/纠正 + 长程闭环, 落 JSON)
self-evolution-bench:
	$(PY) scripts/self_evolution_bench_v43.py
## v4.3.9 完全自进化线终点正式件 (内化冻结 v4.1.0 主权重; 全线外挂零梯度; backcompat 33 件)
ckpt439:
	$(PY) scripts/build_v439_checkpoint.py
## v4.4.0.dev6: 四拓扑(star/chain/mesh) A/B 实测, 落 benchmarks/results/collab_ab_v44.json
collab-ab:
	$(PY) scripts/collab_ab_v44.py
## dev3: 稀疏 change-mask vs 3.6 PWM 稠密 rollout 精度/成本 A/B
wla-sparse-ab:
	$(PY) scripts/wla_sparse_vs_dense_ab.py
## dev6: flow-matching 少步解码器 vs 直接回归 A/B (被反证则 opt-in 留候选账本)
wla-flow-ab:
	$(PY) scripts/wla_flow_vs_regress_ab.py
## 3.9.1: 统一动作空间跨本体/跨末端迁移 A/B (复用 retargeting)
wla-xembodiment-ab:
	$(PY) scripts/wla_cross_embodiment_ab.py
## 3.9.2: 多任务统一头评测矩阵
wla-multitask-matrix:
	$(PY) scripts/wla_multitask_matrix.py
## v2.6.0 hybrid 混合物理修正 A/B 消融 (quick, 20 epochs)
hybrid-ablation:
	$(PY) scripts/ablation_hybrid.py
## v2.3 F2 可复现 A/B (多时域损失权重 front/uniform/back, 多种子)
horizon-ablation:
	$(PY) scripts/ablation_horizon_weight.py
## v2.4.6 校准方法 A/B (PAVA vs Temperature vs None, 多种子)
calib-ablation:
	$(PY) scripts/ablation_calibration.py
## v2.4.9 噪声鲁棒性 A/B (train_sigma x test_sigma, 多种子)
noise-ablation:
	$(PY) scripts/ablation_noise_robustness.py
## v2.4.11 多水平 conformal 区间覆盖率 (80/90/95, 多种子)
conformal-levels:
	$(PY) scripts/ablation_conformal_levels.py

## 部署线 (CHECKPOINT 可指向 checkpoints/predictor_v2.3.1.pt 开箱预加载)
CHECKPOINT ?=
serve:
	$(PY) -m udos.server --host 0.0.0.0 --port 8000 --preset small \
		$(if $(CHECKPOINT),--checkpoint $(CHECKPOINT),)

smoke:
	$(PY) scripts/smoke_test.py

## v4.5.0.dev4 隐式思考 精度-延迟-token Pareto A/B (落 benchmarks/results/pareto_v45.json)
latent-pareto:
	$(PY) scripts/latent_pareto_v45.py

docker-build:
	docker build -t udos-reasoning-engine:5.4.3 .

docker-run:
	docker run --rm -p 8000:8000 --name udos-engine udos-reasoning-engine:5.4.3

clean:
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache .coverage htmlcov
