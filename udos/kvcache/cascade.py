"""KV Cascade: 辅助模型预筛 -> 主模型; 记录筛除比例与召回代理。"""
from __future__ import annotations


class CascadeRouter:
    def __init__(self, recall_floor: float = 0.95):
        self.recall_floor = recall_floor

    def route(self, difficulty: float) -> dict:
        """difficulty 0..1: 低分走 lite 辅助模型(省算力), 高分必须主模型。"""
        lite_ok = difficulty < 0.5
        return {"use_lite": lite_ok, "difficulty": round(difficulty, 3)}

    def evaluate(self, n: int, lite_filtered: int, recall: float) -> dict:
        """召回下限守门; 低于 recall_floor 判不合格。"""
        return {
            "total": n,
            "lite_filtered": lite_filtered,
            "filter_ratio": round(lite_filtered / n, 4) if n else 0.0,
            "recall": recall,
            "passed": recall >= self.recall_floor,
        }


def cascade_run(difficulties: list, threshold: float = 0.5) -> dict:
    """对 difficulty 序列(0..1)跑级联; 主模型输入缩小比例。"""
    if not difficulties:
        raise ValueError("需要 difficulty 序列")
    n = len(difficulties)
    lite = sum(1 for d in difficulties if d < threshold)
    return {
        "total": n,
        "lite_routed": lite,
        "main_routed": n - lite,
        "main_input_shrink": round(1.0 - (n - lite) / n, 4),
        "disclaimer": "analogy, not reproduction",
    }
