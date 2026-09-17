"""GPU/CPU 设备后端抽象(同一 API; 本沙箱只验证 CPU 路径)。

带宽/延迟为显式假设参数, 标注"可替换实测"。GPU 路径无 torch.cuda 不启用。
"""
from __future__ import annotations
from dataclasses import dataclass, field


def gpu_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


@dataclass
class DeviceBackend:
    name: str
    bandwidth_gbs: float          # GB/s, 假设值可替换实测
    latency_us: float              # 每级访问延迟(微秒), 假设值
    tier: str = "ddr"             # hbm/ddr/ssd/remote
    verified_env: bool = True     # CPU 在无GPU沙箱已验证

    def read_cost_us(self, bytes_n: float) -> float:
        return self.latency_us + (bytes_n / (self.bandwidth_gbs * 1e9)) * 1e6


def cpu_backend() -> DeviceBackend:
    # 假设: DDR ~50 GB/s, 访问延迟 80ns
    return DeviceBackend(name="cpu-ddr", bandwidth_gbs=50.0,
                         latency_us=0.08, tier="ddr", verified_env=True)


def hbm_backend() -> DeviceBackend:
    return DeviceBackend(name="hbm", bandwidth_gbs=2000.0,
                         latency_us=0.01, tier="hbm",
                         verified_env=not gpu_available() is False or False)


def ssd_backend(tmpdir="/tmp") -> DeviceBackend:
    return DeviceBackend(name="ssd", bandwidth_gbs=3.0,
                         latency_us=200.0, tier="ssd", verified_env=True)


def remote_backend() -> DeviceBackend:
    return DeviceBackend(name="remote", bandwidth_gbs=0.5,
                         latency_us=5000.0, tier="remote", verified_env=False)
