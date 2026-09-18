"""共享上下文传输层（Shared Context Transport）。

- LocalTransport：进程内多协调器真实可用的发布/订阅与快照交换（verified）。
- CloudTransport：跨机“项目上下文云端共享”的适配器占位。真正的云端后端
  （对象存储/数据库/消息队列 + 凭据）属于部署闸门；未配置 endpoint/token 时
  任何 push/pull 都显式报 GateError，绝不静默假装已上云。
"""
from __future__ import annotations

import threading
from typing import Dict, Optional

from .automation import CapabilityGate
from .memory import SharedMemory


class Transport:
    def push(self, project: str, snap: dict) -> None:  # pragma: no cover - 接口
        raise NotImplementedError

    def pull(self, project: str) -> Optional[dict]:  # pragma: no cover - 接口
        raise NotImplementedError


class LocalTransport(Transport):
    """进程内、按 project 频道保存最新快照。线程安全，测试与单机多协调器可用。"""

    def __init__(self):
        self._lock = threading.RLock()
        self._store: Dict[str, dict] = {}
        self.pull_count = 0

    def push(self, project: str, snap: dict) -> None:
        with self._lock:
            self._store[project] = snap

    def pull(self, project: str) -> Optional[dict]:
        with self._lock:
            self.pull_count += 1
            s = self._store.get(project)
            return None if s is None else dict(s)

    def sync_into(self, project: str, mem: SharedMemory) -> int:
        """便利方法：把频道快照合并进某个 SharedMemory，返回并入事件数。"""
        snap = self.pull(project)
        return 0 if snap is None else mem.merge(snap)


class CloudTransport(Transport):
    """云端共享上下文适配器（需部署闸门）。

    配置项预期：endpoint（如对象存储/同步服务 URL）与 token。未配置即门禁。
    真实实现可在部署侧替换为 S3/OSS/PostgreSQL/消息总线，接口保持 push/pull。
    """

    def __init__(self, endpoint: str = "", token: str = "",
                 gate: Optional[CapabilityGate] = None):
        self.endpoint = endpoint
        self.token = token
        self.gate = gate or CapabilityGate()

    def _configured(self) -> bool:
        return bool(self.endpoint and self.token)

    def push(self, project: str, snap: dict) -> None:
        if not self._configured():
            self.gate.cloud_context_configured = False
            self.gate.require(__import__("udos7.agents.automation", fromlist=["AL"]).AL.AL4)
        # 配置后在此对接真实云端 HTTP/对象存储（部署侧实现，CPU 沙箱不验证）。
        raise NotImplementedError("云端 push 需在部署侧提供真实后端")

    def pull(self, project: str) -> Optional[dict]:
        if not self._configured():
            from .automation import AL
            self.gate.cloud_context_configured = False
            self.gate.require(AL.AL4)
        raise NotImplementedError("云端 pull 需在部署侧提供真实后端")
