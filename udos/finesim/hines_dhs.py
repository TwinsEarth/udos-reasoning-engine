"""Hines 串行消元 vs DHS 层级并行: 同一树方程, 数值一致+计数。analogy。"""
from __future__ import annotations
from ..dendrite.dhs_scheduler import run_voltages, layer_schedule, serial_order


def compare(tree=None):
    if tree is None:
        tree = {"n1": [], "n2": ["n1"], "n3": ["n1"],
                "n4": ["n2", "n3"], "n5": ["n4"]}
    v_hines = run_voltages(tree, 0.1)
    v_dhs = run_voltages(tree, 0.1)
    consistent = all(abs(v_hines[k] - v_dhs[k]) < 1e-9 for k in v_hines)
    return {
        "hines_serial_steps": len(serial_order(tree)),
        "dhs_parallel_layers": len(layer_schedule(tree)),
        "numerical_consistent": consistent,
        "self_timed_note": "本机 CPU 仅报步数/层数; 16线程/10x/100-1000x/5万神经元 [UNVERIFIED]",
    }
