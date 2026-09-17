"""v5.0.2 情报分析服务内核。

公式口径与 agi-asi-countdown 网站 backend/app/core 对齐(只 Read 不改)。
纯逻辑/零梯度/零训练, 离线 CPU 可跑。
"""
from .coefficient import ewma_next, coefficient_today, FORMULA_VERSION
from .singularity_gate import singularity_gate, GATE_FIELDS
from .capability import DIMENSIONS, CapabilityVector
from .s_curve_eta import s_curve_eta
from .prediction_scoring import brier_single, score_predictions
from .asi_rsi import agi_to_asi_years, aggregate_rsi_score
from .model_aggregate import model_aggregate
from .risk_signals import RISK_SEED
from .seed import OFFLINE_SEED
from .api import IntelAPI

__all__ = [
    "ewma_next", "coefficient_today", "FORMULA_VERSION",
    "singularity_gate", "GATE_FIELDS",
    "DIMENSIONS", "CapabilityVector", "s_curve_eta",
    "brier_single", "score_predictions",
    "agi_to_asi_years", "aggregate_rsi_score",
    "model_aggregate", "RISK_SEED", "OFFLINE_SEED", "IntelAPI",
]
