"""UDOS WebAuthn / 平台认证器 MFA（v5.0.1，合规模块 C）。

合规边界：
  - 指纹/人脸等生物比对**只在设备 OS 内完成**；应用仅存公开凭证(credential id +
    公钥)，仅验证断言签名。**绝不采集/存储/上传原始生物模板**。
  - 本沙箱无浏览器/认证器，且未安装 fido2 库 -> 实现**接口契约 + 单测**，
    实际验证路径标 ENV_BLOCKED 优雅降级；**绝不伪造认证成功**。

为何不支持自建声纹/人脸库：自建生物模板库等于存储不可逆的生物特征，
泄露即终身暴露，且受生物特征相关法规严格约束；WebAuthn 把比对留在设备内，
应用只验断言，是合规且更安全的选择（见 docs/SECURITY.md）。
"""
from __future__ import annotations

import secrets
from typing import Any, Dict, Optional

ENV_BLOCKED = True   # 无 fido2/无认证器: 真实验签不可用, 仅契约


class WebAuthnService:
    """注册/断言接口契约。真实验签需 fido2 库 + 浏览器认证器。"""

    def __init__(self):
        self._credentials: Dict[str, Dict[str, Any]] = {}
        self._challenges: Dict[str, str] = {}

    def begin_registration(self, username: str) -> Dict[str, Any]:
        """发起注册: 返回 challenge (真实实现会再含 relying party 选项)。"""
        ch = secrets.token_urlsafe(32)
        self._challenges[username] = ch
        return {"status": "pending", "challenge": ch,
                "note": "需浏览器 WebAuthn 完成 attestation (ENV_BLOCKED)"}

    def verify_registration(self, username: str, attestation: Dict[str, Any]
                            ) -> bool:
        """真实验签需 fido2 库; 本环境 ENV_BLOCKED, 拒绝伪造成功。"""
        if ENV_BLOCKED:
            return False   # 不伪造成功
        raise NotImplementedError  # pragma: no cover

    def begin_login(self, username: str) -> Dict[str, Any]:
        ch = secrets.token_urlsafe(32)
        self._challenges[username] = ch
        return {"status": "pending", "challenge": ch}

    def verify_assertion(self, username: str, assertion: Dict[str, Any]
                         ) -> bool:
        if ENV_BLOCKED:
            return False   # 不伪造成功
        raise NotImplementedError  # pragma: no cover

    def registered_count(self) -> int:
        return len(self._credentials)
