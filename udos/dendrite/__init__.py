"""udos.dendrite — v5.3.0 类脑神经突区室微观编译器(CPU 合成 analogy 原型)。"""
from .compartment import Compartment, CompartmentTree
from .dendritic_compute import (coincidence_detect, nmda_amplify,
                                 temporal_inhibition, inhibition_curve)
from .multimodal_sync import align_channels
from .dhs_scheduler import layer_schedule, serial_order, run_voltages, benchmark

__all__ = [
    "Compartment", "CompartmentTree",
    "coincidence_detect", "nmda_amplify", "temporal_inhibition",
    "inhibition_curve", "align_channels",
    "layer_schedule", "serial_order", "run_voltages", "benchmark",
]
