"""UDOS 受控联网（v5.0.1，合规模块 D）。

  - 总开关默认关（opt-in）；只访问配置白名单内的 URL。
  - 显式超时、可选代理、审计；**不做隐蔽连接、不绕防火墙、不主机发现**。
  - 出站失败不污染主推理路径。
"""
from __future__ import annotations

import os
import time
import urllib.request
from typing import Dict, List, Optional

DEFAULT_TIMEOUT = 5.0


class NetOps:
    def __init__(self, allowlist: Optional[List[str]] = None,
                 timeout: float = DEFAULT_TIMEOUT):
        self.enabled = os.environ.get("UDOS_NET", "off").lower() in (
            "on", "1", "true")
        self.allowlist = list(allowlist or [])
        self.timeout = timeout
        self._audit: List[Dict] = []

    def allow(self, url_prefix: str):
        self.allowlist.append(url_prefix)

    def _is_allowed(self, url: str) -> bool:
        return any(url.startswith(p) for p in self.allowlist)

    def probe(self, url: str) -> Dict:
        """白名单内 URL 的轻量健康探测。白名单外直接拒绝。"""
        rec = {"ts": time.time(), "url": url, "result": ""}
        if not self.enabled:
            rec["result"] = "net_disabled"
            self._audit.append(rec)
            return {**rec, "ok": False, "reason": "netopt opt-in off"}
        if not self._is_allowed(url):
            rec["result"] = "denied_not_allowlisted"
            self._audit.append(rec)
            return {**rec, "ok": False, "reason": "url 不在白名单"}
        try:
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                rec["result"] = "ok"
                self._audit.append(rec)
                return {**rec, "ok": True, "status": r.status}
        except Exception as e:        # noqa: BLE001
            rec["result"] = "error"
            self._audit.append(rec)
            return {**rec, "ok": False, "reason": str(e)[:120]}

    def audit(self) -> List[Dict]:
        return list(self._audit)
