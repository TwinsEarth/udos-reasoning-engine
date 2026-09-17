"""udos.finesim — v5.4.3 精细生物物理数值内核(CPU 参考/类比, 非 NEURON 运行)。"""
from .cable import cable_params, verify_lambda
from .hh import simulate, f_I_curve
from .nmda import nmda_conductance, temporal_suppression, falsifiable_control
from .payeur import payeur_demo, synaptic_position_robustness, ngrad_hypothesis
from .hines_dhs import compare

__all__ = ["cable_params", "verify_lambda", "simulate", "f_I_curve",
           "nmda_conductance", "temporal_suppression", "falsifiable_control",
           "payeur_demo", "synaptic_position_robustness", "ngrad_hypothesis",
           "compare"]
