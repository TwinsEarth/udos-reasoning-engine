"""Payeur 四类树突信息处理 + 突触位置鲁棒性 + NGRAD 假设(外挂, analogy)。"""
from __future__ import annotations
import random


def payeur_demo(seed=7):
    rng = random.Random(seed)
    out = {}
    out["spatiotemporal_filter"] = round(sum(rng.random() for _ in range(4)) / 4, 4)
    out["info_selection"] = "NMDA 集群阈值选中" if rng.random() > 0.3 else "未选中"
    out["info_routing"] = "抑制 A 路 -> 走 B 路"
    out["multiplexing"] = {"ch": 2, "carrier": "平台电位"}
    out["note"] = "Payeur 2019 Curr Opin Neurobiol 58:78; analogy 最小演示"
    return out


def synaptic_position_robustness(seed=7):
    rng = random.Random(seed)
    dist_feedforward = []
    near_soma = []
    for _ in range(50):
        noise = rng.uniform(-0.1, 0.1)
        dist_feedforward.append(abs(0.5 + noise * 0.3))   # 远端: 被动衰减抗噪
        near_soma.append(abs(0.5 + noise))
    spread_far = round(max(dist_feedforward) - min(dist_feedforward), 4)
    spread_near = round(max(near_soma) - min(near_soma), 4)
    return {"far_dendrite_spread": spread_far,
            "near_soma_spread": spread_near,
            "direction": "远端前馈被动衰减->扰动后波动更小(更稳)",
            "note": "缩比合成, analogy; 不复现论文实验"}


def ngrad_hypothesis():
    return {"forward": "前向激活", "backward": "误差信号沿树突平台电位回传(假设)",
            "platform": "NMDA 平台电位承载",
            "beniaguev_ref": "Neuron 109(17):2727-2739,2021; 5-8层时序卷积DNN,R²>0.95",
            "xior_demo": "analogy, 非真实训练; 不入主 state_dict",
            "state": "analogy hypothesis, not trained"}
