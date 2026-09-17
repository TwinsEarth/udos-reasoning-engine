"""离线 seed: 与网站 backend/data/snapshots 同 schema 的确定性演示数据。

无需 GPU/API key 即可离线运行。
"""
from __future__ import annotations

OFFLINE_SEED = {
    "coefficient": [
        {"date": "2025-06-01", "raw_score": 42.0, "weighted_score": 37.8,
         "formula_version": "v1.0", "milestone_flag": False, "confidence": 0.9,
         "sources": ["seed"]},
        {"date": "2025-12-01", "raw_score": 55.0, "weighted_score": 45.5,
         "formula_version": "v1.0", "milestone_flag": False, "confidence": 0.9,
         "sources": ["seed"]},
        {"date": "2026-06-01", "raw_score": 68.0, "weighted_score": 61.2,
         "formula_version": "v1.0", "milestone_flag": True, "confidence": 0.95,
         "sources": ["seed"]},
    ],
    "capability": {
        "reasoning": 78, "coding": 82, "multimodal": 70, "agency": 55,
        "long_context": 80, "embodied": 40, "efficiency": 60,
    },
    "routes": [
        {"route": "LLM-scaling", "orgs": ["OpenAI", "DeepMind"],
         "people": [], "progress_pct": 62, "milestones": []},
        {"route": "embodied", "orgs": ["Figure"],
         "people": [], "progress_pct": 35, "milestones": []},
    ],
    "predictions": [
        {"author": "seed", "org": "internal", "claim": "AGI by 2028",
         "predicted_on": "2025-01-01", "due_date": "2028-12-31",
         "source_url": "internal://seed", "source_type": "seed",
         "status": "pending", "confidence": 0.6, "related_route": "LLM-scaling"},
    ],
    "rsi": {"compute": 55, "data": 60, "algorithms": 65},
    "models": [
        {"model_id": "m-lead", "label": "领袖共识", "perspective": "leader",
         "filter_rule": "top_lab", "median_eta_year": 2028.0,
         "series": [{"date": "2024-01", "eta_year": 2027.0}]},
        {"model_id": "m-res", "label": "研究者中位", "perspective": "researcher",
         "filter_rule": "peer_reviewed", "median_eta_year": 2031.0,
         "series": [{"date": "2024-01", "eta_year": 2030.0}]},
    ],
}
