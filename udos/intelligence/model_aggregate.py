"""多模型聚合(The AGI Clock 思路): 同数据不同过滤/权重/AGI 定义产出差异化时间线。

领袖视角(leader)施加乐观偏差折减(eta 后移); 研究者(researcher)并列;
社区(community)取中位。bias_adjustment 显式记录折减量。
"""
from __future__ import annotations


def model_aggregate(models):
    """输入 [{model_id, label, perspective, median_eta_year, series}],
    返回折减后的 models 列表。leader 视角 eta 后移 +1.5 年(乐观折减)。"""
    out = []
    for m in models:
        eta = float(m.get("median_eta_year", 2030.0))
        persp = m.get("perspective", "researcher")
        adj = 0.0
        if persp == "leader":
            adj = 1.5          # 领袖公开ETA偏乐观, 折减后移
        elif persp == "community":
            adj = -0.5
        out.append({
            "model_id": m.get("model_id"),
            "label": m.get("label"),
            "filter_rule": m.get("filter_rule", "all"),
            "perspective": persp,
            "bias_adjustment": adj,
            "median_eta": round(eta + adj, 2),
            "series": m.get("series", []),
        })
    return out
