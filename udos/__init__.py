"""
UDOS 推演引擎 (UDOS Reasoning Engine) — v5.5.5
==============================================
双引擎架构:
    GPM (Generative Physics engine)  —— 机制对齐 Sakana AI Doc-to-LoRA:
        单次前向把物理场景压缩为 LoRA 适配器并前向注入, 零反向传播。
    CTM (Continuous Thought Machine) —— 机制对齐 Sakana AI Continuous-Thought-Machines:
        内部时间轴 + 神经元级时序模型 + 神经同步表示, 在时间维度上展开因果推演。

v2.0.0 第二代能力:
    1. 物理学习闭环: dynamics 合成数据集 + training 训练器, 真实 loss 下降
    2. GPM->CTM 场景条件真耦合; 3. persistence 存/载; 4. 可解释下一状态 + /train

v2.1.0 迭代:
    1. 多步滚动推演: 参数化场景 + horizon 多步目标, teacher-forcing 训练 + 自由 rollout
    2. 场景条件联合训练: 隐藏物理参数经 scene_encoder 进训练回路, 消融实证条件增益
    3. 物理一致性: 一阶欧拉运动学残差诊断 (分运动类型)
    4. evaluation 评估体系 + 服务多步推演 + checkpoint 预加载

v2.2.x 迭代:
    1. Scheduled Sampling (opt-in, 默认 ss_max=0 与 2.1 等价; 收益经 A/B 证为条件依赖)
    2. 训练治理: 早停/最佳权重恢复、set_seed 统一确定性
    3. 评估增强: rollout 误差累积率、置信度分层校准诊断
    4. 模型管理服务: POST /evaluate、POST /save (未训练 409, 落盘路径白名单)

v2.3.x 迭代:
    1. 置信度保序回归校准 (calibration, 确定性后处理) + 回归 ECE/可靠性
    2. 多时域均衡损失 (front/uniform/back, 默认 front 等价 2.2.1)
    3. split-conformal 经验预测区间与覆盖率
    4. 服务 POST /calibrate、GET /checkpoints、POST /load

v2.4.x 迭代:
    1. OOD/漂移检测器 + StreamingDriftDetector (马氏距离 + 经验分位阈值)
    2. DeepEnsemble 深度集成 + split-conformal 多水平经验区间
    3. PredictionGuard 行级退化回退 (NaN/inf 回退 + 越界截断, opt-in)
    4. 在线 /detect-ood /predict 服务接口 + 逐步逐维置信可解释性

v2.5.x 迭代:
    1. BatchPredictor 批量推理引擎 (变长序列自动分组 + 大 batch 分片, 与逐笔逐位一致)
    2. InferenceCache LRU 推理缓存 (输入哈希+模型参数哈希, 默认关闭 opt-in)
    3. PhysicsPredictor.predict_batch 批量入口

v2.6.x 迭代:
    1. HybridPhysicsCorrector 混合物理修正 (一阶欧拉骨架+学习残差, opt-in 默认关)
    2. CounterfactualEngine 反事实推演 + SceneParameterIdentifier 场景参数辨识/Sobol
    3. AdaptiveStopper 自适应 iterations + RiskGrader/safety_boundary 风险传播

v2.7.x 迭代 (从预测到行动的闭环):
    1. policy.MPCActionSelector: MPC 式候选动作 rollout 优选 (奖励-lambda*风险-安全违例)
    2. online.OnlineAdapter: 流式漂移触发再校准闭环 (默认只重跑 PAVA, 不改权重)
    3. active_learning.UncertaintySampler: 集成方差+区间宽+OOD 合成信息增益选点
    4. lite: 幅值剪枝 / 动态 INT8 量化 / 蒸馏学生模型 (opt-in, 默认全量模型不变)
    5. hierarchical.HierarchicalRollout: 粗粒度跳步 + 细粒度修正的多尺度长时域 rollout
    6. experiment.ExperimentRegistry: 实验元数据注册 + 多种子 sweep (mean/std/best) 治理

上游开源代码库 (开源包不内置其源码; 可选适配器在显式启用时经 huggingface_hub 运行时拉取):
    - SakanaAI/continuous-thought-machines
    - SakanaAI/doc-to-lora

版本: v5.5.5
"""

import logging as _logging

logger = _logging.getLogger("udos")

from .pce_format import (
    PhysicalToken,
    PhysicsScene,
    PCEParser,
    PhysicsSceneEncoder,
    collate_tokens,
    DemonstrationPrompt,
    PCEPromptParser,
)
from .ctm_engine import CTMPhysicsEngine, CTMConfig
from .gpm_engine import (
    PhysicsHypernetwork,
    GPMConfig,
    LoRAInjector,
    LoRASet,
)
from .gpm_memory_bridge import GPMSceneBridge
from .scene_head import (
    SceneEstimationHead, differentiable_rollout, train_scene_head,
    save_scene_head, load_scene_head,
)
from .scene_fan import (
    ParamErrorModel, TrajectoryFan, monte_carlo_rollout,
    coverage_fraction, per_step_coverage, fit_conformal_inflation,
    calibrated_band,
)
from .dynamics_router import (
    DynamicsRoute, classify_dynamics, routed_scene_params,
    KindSpecificParamErrorModel, apply_kind_inflation,
    fit_kind_conformal_inflation,
)
from .dual_engine_observe import EngineObservation, observe_conditioning
from .reasoning import UDOSReasoningEngine, ReasoningResult
from .dynamics import (
    RAW_DIM,
    DynamicsDataset,
    build_dynamics_dataset,
    naive_baseline_mse,
    SCENE_PARAM_DIM,
    SCENE_PARAM_NAMES,
    ParametricDynamicsDataset,
    build_parametric_dataset,
    kinematic_residual,
)
from .training import (
    PhysicsPredictor,
    CTMTrainer,
    TrainConfig,
    TrainHistory,
)
from .evaluation import evaluate_predictor
from .calibration import (
    ConfidenceCalibrator,
    TemperatureScaling,
    fit_predictor_calibration,
    calibration_report,
    reliability,
    spearman,
)
from .persistence import (
    save_predictor,
    load_predictor,
    save_module,
    load_module_state,
    save_ensemble,
    load_ensemble,
    diff_snapshots,
    compare_checkpoints,
)
from .ood import DistributionDriftDetector, StreamingDriftDetector
from .ensemble import DeepEnsemble
from .guard import PredictionGuard
from .batch import BatchPredictor, predict_batch
from .cache import InferenceCache
from .hybrid import HybridPhysicsCorrector
from .counterfactual import CounterfactualEngine
from .identification import SceneParameterIdentifier, sobol_attribution
from .adaptive import AdaptiveStopper, adaptive_rollout
from .decision import RiskGrader, safety_boundary
from .policy import MPCActionSelector
from .online import OnlineAdapter
from .active_learning import UncertaintySampler
from .lite import MagnitudePruner, DynamicQuantizer, DistillationTrainer
from .hierarchical import HierarchicalRollout
from .experiment import ExperimentRegistry
from .icm import DemonstrationEpisode, DemonstrationMemory, ICMAggregator
from .icm_events import EventSegmenter, ThreeStreamAligner
from .icm_cross import CrossEmbodimentICM
from .icm_budget import ContextBudgetManager
from .spatial import (SpatialObject, SpatialScene, SpatialTransform,
                      OrthographicView, multiview_consistency_error)
from .scene_graph import SceneGraph
from .occupancy import OccupancyGrid, DistanceField
from .collision import CollisionDetector, NearestNeighbor
from .spatial_query import SpatialQueryEngine

from .neural_control import (
    ControlLayer,
    CortexPlanner,
    CerebellumTracker,
    SpinalReflex,
    HierarchicalController,
)

from .multi_agent import AgentCoordinator, AgentState, MultiAgentScene
from .wm_scheduler import WMScheduler
from .closed_loop import ClosedLoopOrchestrator
from .digital_twin import DigitalTwinScene

# v3.9.9 宇树 UnifoLM-WLA 机制类比线 (统一具身推理头 + 动作三分组 + WLA 外挂)
from .embodied import EmbodiedReasoningHead, ActionTriGroup
from .wla import (ChangeMask, ChangeMaskVQ, RVQActionTokenizer,
                  ActionStateTaskAlign, FlowMatchingDecoder)

# v5.4.3 自规划自监督线起点 (任务/课程自动生成器 + 可解性自验证器, 纯前向外挂)
from .curriculum import LessonSpec, CurriculumGenerator, SolvabilityVerifier
from .selfsup import PWMConsistencyPseudoLabeler, PhysicsMultiviewGate
from .selfplan import GoalDecomposer, StopCorrectController

# v5.4.3 完全自训练线 (RSI 一代闭环: 世界模型自生成三元组骨架, 纯前向外挂零梯度)
from .self_train import (SelfGeneratedTriplets, TransitionTripletGenerator,
                         SelfPlayExplorer, ExecutionValidator, ModelReviewer,
                         HumanAuditHook, DataQualityLedger, TeacherStudentLoop,
                         DataRefluxMixer, DegradationDetector, RollbackManager,
                         DiminishingReturnsCriterion, SelfTrainAB, LINE_STAGES)

# v5.4.3 完全自进化线 (基础设施自我优化缩微类比: 配置自优化搜索器, 纯前向外挂零梯度)
from .self_evolution import (ConfigSpec, ConfigEvaluator, ConfigSearcher,
                             SearchVerifySelectLoop, ConfigAB,
                             SelfEvolutionOrchestrator, MultiGenerationRunner,
                             GlobalStopCorrectCriterion, LongHorizonLoop,
                             SELF_EVO_LINE_STAGES)

# v5.4.3 多智能体协作&协同&协调线 (四拓扑/决策树/Bundle/治理三件套, 纯前向外挂零梯度)
from .transfer_bundle import TransferBundle
from .collab_governance import (OwnerLedger, TraceChain, StopGuard, ClaimLock)
from .collab_topology import Topology, TopologySelector, Selection
from .collab_agents import (ToolSpec, ToolResult, CapabilityRegistry,
                            build_default_registry)
from .collab_orchestrator import StarOrchestrator
from .collab_handoff import ChainHandoff
from .collab_swarm import MeshSwarm

# v5.4.3 开源资源注册表 (L0-L3 四层, 惰性装配; 别名避免与 collab_agents 冲突)
from .resource_registry import (ResourceRegistry, ResourceConnector,
                                ResourceSpec, ResourceUnavailable,
                                normalize_trajectory)
from .connectors import build_default_registry as build_default_resource_registry

__version__ = "5.5.5"

__all__ = [
    "PhysicalToken",
    "PhysicsScene",
    "PCEParser",
    "PhysicsSceneEncoder",
    "collate_tokens",
    "DemonstrationPrompt",
    "PCEPromptParser",
    "CTMPhysicsEngine",
    "CTMConfig",
    "PhysicsHypernetwork",
    "GPMConfig",
    "GPMSceneBridge",
    "SceneEstimationHead",
    "differentiable_rollout",
    "train_scene_head",
    "save_scene_head",
    "load_scene_head",
    "ParamErrorModel",
    "TrajectoryFan",
    "monte_carlo_rollout",
    "coverage_fraction",
    "per_step_coverage",
    "fit_conformal_inflation",
    "calibrated_band",
    "DynamicsRoute",
    "classify_dynamics",
    "routed_scene_params",
    "KindSpecificParamErrorModel",
    "apply_kind_inflation",
    "fit_kind_conformal_inflation",
    "EngineObservation",
    "observe_conditioning",
    "LoRAInjector",
    "LoRASet",
    "UDOSReasoningEngine",
    "ReasoningResult",
    # v2.0
    "RAW_DIM",
    "DynamicsDataset",
    "build_dynamics_dataset",
    "naive_baseline_mse",
    "PhysicsPredictor",
    "CTMTrainer",
    "TrainConfig",
    "TrainHistory",
    "save_predictor",
    "load_predictor",
    "save_module",
    "load_module_state",
    "save_ensemble",
    "load_ensemble",
    # v2.1
    "SCENE_PARAM_DIM",
    "SCENE_PARAM_NAMES",
    "ParametricDynamicsDataset",
    "build_parametric_dataset",
    "kinematic_residual",
    "evaluate_predictor",
    # v2.3
    "ConfidenceCalibrator",
    "TemperatureScaling",
    "fit_predictor_calibration",
    "calibration_report",
    "reliability",
    "spearman",
    # v2.4
    "DistributionDriftDetector",
    "StreamingDriftDetector",
    "DeepEnsemble",
    "PredictionGuard",
    # v2.5
    "BatchPredictor",
    "predict_batch",
    "InferenceCache",
    # v2.6
    "HybridPhysicsCorrector",
    "CounterfactualEngine",
    "SceneParameterIdentifier",
    "sobol_attribution",
    "AdaptiveStopper",
    "adaptive_rollout",
    "RiskGrader",
    "safety_boundary",
    "diff_snapshots",
    "compare_checkpoints",
    # v2.7
    "MPCActionSelector",
    "OnlineAdapter",
    "UncertaintySampler",
    "MagnitudePruner",
    "DynamicQuantizer",
    "DistillationTrainer",
    "HierarchicalRollout",
    "ExperimentRegistry",
    # v3.6.3 ICM 上下文记忆
    "DemonstrationEpisode",
    "DemonstrationMemory",
    "ICMAggregator",
    "EventSegmenter",
    "ThreeStreamAligner",
    "CrossEmbodimentICM",
    "ContextBudgetManager",
    # v3.6.3 SFM 空间基础模型 (纯推理外挂)
    "SpatialObject",
    "SpatialScene",
    "SpatialTransform",
    "OrthographicView",
    "multiview_consistency_error",
    "SceneGraph",
    "OccupancyGrid",
    "DistanceField",
    "CollisionDetector",
    "NearestNeighbor",
    "SpatialQueryEngine",
    # v5.4.3 分层神经控制 (大脑/小脑/脊髓, 纯推理外挂)
    "ControlLayer",
    "CortexPlanner",
    "CerebellumTracker",
    "SpinalReflex",
    "HierarchicalController",
    # v3.8.0 多体协同核心 (合成参数化代理, 纯推理外挂)
    "AgentCoordinator",
    "AgentState",
    "MultiAgentScene",
    # v5.4.3 世界模型调度器 (多体想象预算分配, 纯推理外挂)
    "WMScheduler",
    # v5.4.3 规划-执行-反馈全域闭环 (纯推理外挂)
    "ClosedLoopOrchestrator",
    # v5.4.3 合成数字孪生场景 (参数化生成/快照/回放)
    "DigitalTwinScene",
    # v3.9.9 宇树 UnifoLM-WLA 机制类比线 (纯推理外挂, 零梯度)
    "EmbodiedReasoningHead",
    "ActionTriGroup",
    "ChangeMask",
    "ChangeMaskVQ",
    "RVQActionTokenizer",
    "ActionStateTaskAlign",
    "FlowMatchingDecoder",
    # v5.4.3 自规划自监督线起点 (任务/课程生成器 + 可解性自验证器, 纯前向外挂)
    "LessonSpec",
    "CurriculumGenerator",
    "SolvabilityVerifier",
    # v5.4.3.dev3 自监督伪信号 (PWM rollout 一致性, 纯前向外挂)
    "PWMConsistencyPseudoLabeler",
    # v4.1.0.dev4 自监督伪信号门 (物理守恒+多视角一致, 纯前向外挂)
    "PhysicsMultiviewGate",
    # v4.1.0.dev6 自规划目标分解 (goal->子目标链, 复用 policy, 纯前向外挂)
    "GoalDecomposer",
    # v5.4.3 停止/自我纠正判据 (置信门控+收敛+OOD主动验证, 纯前向外挂)
    "StopCorrectController",
    # v5.4.3 完全自训练线 (RSI 一代闭环, 世界模型自生成三元组骨架, 纯前向外挂)
    "SelfGeneratedTriplets",
    "TransitionTripletGenerator",
    "SelfPlayExplorer",
    "ExecutionValidator",
    "ModelReviewer",
    "HumanAuditHook",
    "DataQualityLedger",
    "TeacherStudentLoop",
    "DataRefluxMixer",
    "DegradationDetector",
    "RollbackManager",
    "DiminishingReturnsCriterion",
    "SelfTrainAB",
    "LINE_STAGES",
    # v5.4.3 完全自进化线 (基础设施自我优化缩微类比, 纯前向外挂零梯度)
    "ConfigSpec",
    "ConfigEvaluator",
    "ConfigSearcher",
    "SearchVerifySelectLoop",
    "ConfigAB",
    "SelfEvolutionOrchestrator",
    "MultiGenerationRunner",
    "GlobalStopCorrectCriterion",
    "LongHorizonLoop",
    "SELF_EVO_LINE_STAGES",
    # v5.4.3 多智能体协作线
    "TransferBundle",
    "OwnerLedger",
    "TraceChain",
    "StopGuard",
    "ClaimLock",
    "Topology",
    "TopologySelector",
    "Selection",
    "ToolSpec",
    "ToolResult",
    "CapabilityRegistry",
    "build_default_registry",
    "StarOrchestrator",
    "ChainHandoff",
    "MeshSwarm",
]
