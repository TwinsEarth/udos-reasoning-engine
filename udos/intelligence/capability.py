"""多维能力向量(口径同网站 agi_progress.py)。"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List

DIMENSIONS = ["reasoning", "coding", "multimodal", "agency",
              "long_context", "embodied", "efficiency"]
DEFAULT_THRESHOLDS = {"reasoning": 95, "coding": 95, "multimodal": 90,
                      "agency": 90, "long_context": 95, "embodied": 85,
                      "efficiency": 80}


@dataclass
class CapabilityVector:
    scores: Dict[str, float] = field(default_factory=dict)
    sources: Dict[str, List[str]] = field(default_factory=dict)
    confidence: Dict[str, float] = field(default_factory=dict)

    def dimensions_list(self):
        out = []
        for name in DIMENSIONS:
            out.append({
                "name": name,
                "score": self.scores.get(name, 0.0),
                "sources": self.sources.get(name, []),
                "confidence": self.confidence.get(name, 0.5),
            })
        return out
