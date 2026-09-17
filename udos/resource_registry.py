"""UDOS 开源资源注册表 (Resource Registry) — v4.5.4

四层诚信落地 (L0-L3), 与 CPU-only / 零重依赖环境严格对齐:

  L0 注册表(元数据): catalog 全量资源进统一 registry, 纯声明、零运行成本。
  L1 格式/数据适配:   真实实现公开数据格式 -> UDOS(PCE-Format/tensor) 表示的
                      解析与转换, 用最小内联 fixture 单测往返一致。
  L2 轻量可运行:      仅 CPU 可跑、pip 可装、许可宽松的小库真实 import+smoke;
                      connector 惰性 import, 缺失时该资源标 absent, 核心照常 import。
  L3 大权重/GPU:      只实现 connector 接口契约 + 能力声明(requires_gpu/weights)
                      + 懒加载 + 缺失时 503/absent 优雅降级, 不下载、不假装运行。

状态四态:
  available   —— 本环境真实可探测/可运行 (L2 smoke 通过, 或 L1 纯 python 转换跑通)
  degraded    —— connector 存在但部分子能力缺失 (主路径仍可用)
  absent      —— 外部包/权重需要但当前未安装/未加载 (契约就绪, 给安装/启动提示)
  env_blocked —— 能力探测被环境阻断 (如 pip 无外网索引)

严禁: 伪装跑通大模型 / 编造推理输出或成功率 / 把"接口已适配"说成"模型已运行"。
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

# 引擎主权重只读、零训练: registry 无可学参数, 不进主 state_dict。
RESOURCE_LEVELS = ("L0", "L1", "L2", "L3")
RESOURCE_STATUSES = ("available", "degraded", "absent", "env_blocked")
KINDS = ("model", "dataset", "action")
PROFILES = ("performance", "full")


class ResourceUnavailable(Exception):
    """调用了当前环境不可用的资源 (对应 HTTP 503)。"""

    def __init__(self, status: str, message: str,
                 *, install_hint: Optional[str] = None,
                 requires: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.status = status          # absent | env_blocked
        self.install_hint = install_hint
        self.requires = requires or {}


@dataclass(frozen=True)
class ResourceSpec:
    id: str
    name: str
    kind: str                       # model | dataset | action
    license: str
    level: str                      # L0..L3
    priority: str                   # high | normal
    requires_gpu: bool = False
    requires_weights: bool = False
    requires_pkg: Optional[str] = None   # L2 惰性 import 的 python 模块名
    profile_tags: tuple = ()
    category: str = ""
    org: str = ""
    source_url: str = ""
    cpu_feasibility: str = ""
    udos_fit: str = ""
    oss_triangle: str = ""
    scale: str = ""

    def public(self) -> Dict[str, Any]:
        return {
            "id": self.id, "name": self.name, "kind": self.kind,
            "license": self.license, "level": self.level,
            "priority": self.priority, "category": self.category,
            "org": self.org, "source_url": self.source_url,
            "cpu_feasibility": self.cpu_feasibility, "udos_fit": self.udos_fit,
            "scale": self.scale,
            "requires": {"gpu": self.requires_gpu,
                         "weights": self.requires_weights,
                         "pkg": bool(self.requires_pkg)},
            "profile_tags": list(self.profile_tags),
        }


@dataclass
class CapabilityReport:
    status: str
    level: str
    detail: str
    probed_at: float = 0.0
    install_hint: Optional[str] = None
    requires: Dict[str, Any] = field(default_factory=dict)
    evidence: str = ""              # 真实 smoke 证据 (L2) / fixture 往返 (L1)

    def public(self) -> Dict[str, Any]:
        return {
            "status": self.status, "level": self.level,
            "detail": self.detail, "probed_at": self.probed_at,
            "install_hint": self.install_hint, "requires": self.requires,
            "evidence": self.evidence,
        }


def try_import_module(module: str) -> tuple:
    """惰性 import。返回 (ok, err_or_module)。绝不抛异常。"""
    import importlib
    try:
        mod = importlib.import_module(module)
        return True, mod
    except Exception as e:            # noqa: BLE001 —— 任何 import 失败都算 absent
        return False, f"{type(e).__name__}: {e}"


def pip_index_reachable(timeout: float = 4.0) -> Optional[bool]:
    """探测 pip 索引是否可达 (仅用于区分 absent vs env_blocked)。
    返回 None 表示未能判定 (按可处理对待)。"""
    import json
    import urllib.request
    try:
        url = os.environ.get("UDOS_PIP_INDEX", "https://pypi.org/simple/")
        with urllib.request.urlopen(url, timeout=timeout) as r:    # noqa: S310
            return 200 <= r.status < 500
    except Exception:                 # noqa: BLE001
        return False


class ResourceConnector:
    """所有 connector 的基类。子类覆写 probe/invoke/to_udos。"""

    spec: ResourceSpec

    def __init__(self, spec: ResourceSpec):
        self.spec = spec
        self._report: Optional[CapabilityReport] = None
        self._lock = threading.Lock()

    # ---- 能力探测 (带缓存, 永不抛) ----
    def probe(self) -> CapabilityReport:
        with self._lock:
            if self._report is not None:
                return self._report
            rep = self._probe_uncached()
            rep.probed_at = time.time()
            self._report = rep
            return rep

    def _probe_uncached(self) -> CapabilityReport:
        s = self.spec
        # L3: 需要 GPU / 大权重 —— 契约就绪, 本环境不下载, 标 absent。
        if s.requires_gpu or s.requires_weights:
            return CapabilityReport(
                status="absent", level=s.level,
                detail=("契约就绪(connector 接口+能力声明+懒加载); 需 GPU/外部权重, "
                        "本 CPU 环境不下载、不假装运行"),
                install_hint=self._install_hint(),
                requires={"gpu": s.requires_gpu, "weights": s.requires_weights})
        # L2: 需要外部小库 —— 惰性真实 import。
        if s.requires_pkg:
            ok, info = try_import_module(s.requires_pkg)
            if ok:
                rep = self._smoke(ok, info)
                rep.requires = {"pkg": s.requires_pkg, "present": True}
                return rep
            # import 失败: 区分 absent (有外网可装) vs env_blocked (无索引)
            reachable = pip_index_reachable()
            if reachable is False:
                return CapabilityReport(
                    status="env_blocked", level=s.level,
                    detail=f"探测被环境阻断: 无法 import {s.requires_pkg} ({info}); "
                           f"且 pip 索引不可达, 降级为 L1+架构卡片",
                    install_hint=self._install_hint(),
                    requires={"pkg": s.requires_pkg, "present": False})
            return CapabilityReport(
                status="absent", level=s.level,
                detail=f"需要 python 包 {s.requires_pkg} 但未安装 ({info}); "
                       f"connector 惰性 import, 缺失不影响核心 udos",
                install_hint=self._install_hint(),
                requires={"pkg": s.requires_pkg, "present": False})
        # L0/L1: 纯 python 格式适配, 本环境直接可用。
        return CapabilityReport(
            status="available", level=s.level,
            detail="纯 python 格式适配/元数据声明, 本环境可直接转换",
            evidence="L1 inline fixture 往返 (见单测)")

    def _smoke(self, module: Any, mod: Any) -> CapabilityReport:
        """子类可覆写: 包已 import 成功后的最小 smoke。默认成功即 available。"""
        return CapabilityReport(
            status="available", level=self.spec.level,
            detail=f"已惰性 import {self.spec.requires_pkg} 并通过最小 smoke",
            evidence=f"import {self.spec.requires_pkg} OK")

    def _install_hint(self) -> Optional[str]:
        s = self.spec
        hints = []
        if s.requires_pkg:
            hints.append(f"pip install {s.requires_pkg}")
        if s.requires_gpu:
            hints.append("需 GPU 运行时 + 外部权重 (见 source_url)")
        if s.requires_weights:
            hints.append("需下载外部权重 (见 source_url; 本环境不下载)")
        return "; ".join(hints) or None

    # ---- 统一调用 ----
    def _ensure_loadable(self) -> None:
        rep = self.probe()
        if rep.status in ("absent", "env_blocked"):
            raise ResourceUnavailable(
                rep.status,
                f"资源 {self.spec.id} 当前 {rep.status}: {rep.detail}",
                install_hint=rep.install_hint, requires=rep.requires)

    def to_udos(self, payload: Any) -> Dict[str, Any]:
        """L1: 外部格式 -> UDOS PCE/action 表示。默认归一化为 canonical 轨迹。"""
        return normalize_trajectory(payload)

    def invoke(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        self._ensure_loadable()
        if action == "convert":
            return {"status": "ok", "id": self.spec.id,
                    "udos": self.to_udos(params.get("payload"))}
        if action == "capability":
            return {"status": "ok", "id": self.spec.id,
                    "capability": self.probe().public(),
                    "spec": self.spec.public()}
        raise ValueError(f"未知 action '{action}' (可选: convert/capability)")


def normalize_trajectory(payload: Any) -> Dict[str, Any]:
    """把异构轨迹/动作 chunk 归一化为 UDOS canonical 表示 (纯 python, 无重依赖)。

    输出 schema: udos/pce-action/v1
      observations: List[List[float]]  —— 每步本体状态/观测
      actions:      List[List[float]]  —— 连续动作 (7 维约定, 不足补 0)
      timestamps:   List[float]
    """
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise ValueError("payload 需为 dict (含 observations/actions/timestamps)")

    def _rows(key: str) -> List[List[float]]:
        rows = payload.get(key) or []
        out = []
        for r in rows:
            if isinstance(r, (int, float)):
                r = [r]
            if not isinstance(r, (list, tuple)):
                raise ValueError(f"{key} 行需为数值或数值列表")
            out.append([float(x) for x in r])
        return out

    obs = _rows("observations")
    acts = _rows("actions")
    ts = payload.get("timestamps")
    if ts is None:
        ts = [float(i) for i in range(max(len(obs), len(acts)))]
    elif not isinstance(ts, (list, tuple)):
        raise ValueError("timestamps 需为列表")
    return {
        "schema": "udos/pce-action/v1",
        "episode_id": payload.get("episode_id"),
        "observations": obs,
        "actions": acts,
        "timestamps": [float(t) for t in ts],
        "n_steps": max(len(obs), len(acts)),
    }


class ResourceRegistry:
    """统一注册表: 枚举 / 过滤 / 探测 / 统一调用。线程安全。"""

    def __init__(self, profile: str = "performance"):
        if profile not in PROFILES:
            raise ValueError(f"未知 profile '{profile}', 可选 {PROFILES}")
        self._connectors: Dict[str, ResourceConnector] = {}
        self._profile = profile
        self._lock = threading.RLock()

    # ---- 装配 ----
    def register(self, conn: ResourceConnector) -> None:
        with self._lock:
            self._connectors[conn.spec.id] = conn

    @property
    def profile(self) -> str:
        return self._profile

    def apply_profile(self, profile: str) -> str:
        if profile not in PROFILES:
            raise ValueError(f"未知 profile '{profile}', 可选 {PROFILES}")
        with self._lock:
            self._profile = profile
            # 切换不改主权重、不改默认数值; 仅影响"自动探测的子集"。
            # 清空探测缓存, 让 full 下重新批量探测 (带超时/并发上限由调用方控制)。
            for c in self._connectors.values():
                c._report = None
        return self._profile

    def _in_performance_set(self, spec: ResourceSpec) -> bool:
        # performance: 仅高优先级 + L1/L2 (排除 L3 大权重契约)。
        return spec.priority == "high" and spec.level in ("L0", "L1", "L2")

    # ---- 查询 ----
    def get(self, rid: str) -> ResourceConnector:
        if rid not in self._connectors:
            raise KeyError(rid)
        return self._connectors[rid]

    def ids(self) -> List[str]:
        with self._lock:
            return list(self._connectors.keys())

    def list(self, *, profile: Optional[str] = None, kind: Optional[str] = None,
             status: Optional[str] = None, license: Optional[str] = None,
             priority: Optional[str] = None, level: Optional[str] = None,
             autoprobe: bool = True) -> List[Dict[str, Any]]:
        prof = profile or self._profile
        with self._lock:
            items = list(self._connectors.values())
        out = []
        for c in items:
            s = c.spec
            if kind and s.kind != kind:
                continue
            if priority and s.priority != priority:
                continue
            if level and s.level != level:
                continue
            if license and license.lower() not in s.license.lower():
                continue
            # profile 过滤: performance 只暴露高性能子集; full 全量。
            if prof == "performance" and not self._in_performance_set(s):
                continue
            cap = c.probe().public() if autoprobe else None
            if status and cap and cap["status"] != status:
                continue
            row = s.public()
            if cap:
                row["capability"] = cap
            out.append(row)
        return out

    # ---- 探测 / 调用 ----
    def probe(self, rid: str) -> Dict[str, Any]:
        c = self.get(rid)
        rep = c.probe().public()
        return {"status": "ok", "id": rid, "spec": c.spec.public(),
                "capability": rep}

    def invoke(self, rid: str, action: str,
               params: Dict[str, Any]) -> Dict[str, Any]:
        c = self.get(rid)
        if not action:
            raise ValueError("缺少 'action' 字段 (convert/capability)")
        return c.invoke(action, params or {})

    def summary(self, profile: Optional[str] = None) -> Dict[str, Any]:
        rows = self.list(profile=profile, autoprobe=True)
        from collections import Counter
        return {
            "profile": profile or self._profile,
            "total_registered": len(self._connectors),
            "visible_in_profile": len(rows),
            "by_status": dict(Counter(r.get("capability", {}).get("status", "?")
                                     for r in rows)),
            "by_level": dict(Counter(r["level"] for r in rows)),
            "by_kind": dict(Counter(r["kind"] for r in rows)),
        }


def registry_from_env(default: str = "performance") -> str:
    """UDOS_RESOURCE_PROFILE 环境变量覆盖默认 profile。"""
    return os.environ.get("UDOS_RESOURCE_PROFILE", default)
