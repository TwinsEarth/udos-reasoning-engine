"""树突区室(apical/basal), 距胞体距离 -> 被动衰减权重。analogy, not reproduction。"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Compartment:
    name: str
    kind: str            # apical / basal / soma
    distance: float      # 距胞体距离(任意单位)
    leak: float = 0.3    # 被动衰减系数假设值

    @property
    def attenuation(self) -> float:
        # 远端衰减更强: exp(-leak*distance)
        import math
        return round(math.exp(-self.leak * self.distance), 6)


@dataclass
class CompartmentTree:
    compartments: Dict[str, Compartment] = field(default_factory=dict)
    voltage_history: Dict[str, List[float]] = field(default_factory=dict)

    def add(self, c: Compartment):
        self.compartments[c.name] = c
        self.voltage_history[c.name] = []

    def passive_weight(self, name: str) -> float:
        return self.compartments[name].attenuation

    def record(self, name: str, v: float):
        self.voltage_history.setdefault(name, []).append(float(v))
