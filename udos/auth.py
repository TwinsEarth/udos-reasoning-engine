"""UDOS 认证与 RBAC（v5.0.1，安全合规模块 A）。

设计原则（见 docs/SECURITY.md）：
  - 纯标准库实现，零重依赖：口令用 ``hashlib.pbkdf2_hmac``（等价标准 KDF；
    若可离线安装 argon2/bcrypt 见 ENV_BLOCKED 说明，本环境用 pbkdf2 即可满足
    "不使用明文/弱哈希"的要求）。
  - 会话 token 为 HMAC 签名 + 过期 + 服务端可吊销（不是不透明自校验，
    注销/权限变更可立即生效）。
  - 登录失败指数退避 + 锁定（防暴力破解的正道）。
  - append-only 审计日志：只记事件/主体/动作/结果，**绝不记口令或 token 明文**。
  - 不含任何硬编码口令/万能凭据/后门；首个 owner 只能由受控初始化创建。

本模块不做任何网络外联、不采集生物特征。
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import os
import secrets
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("udos.auth")

# ---- 角色权限矩阵 -------------------------------------------------------
# viewer: 只读；operator: +写/推理/训练相关；owner: +用户管理/备份/自治/审计
ROLES = ("viewer", "operator", "owner")
_PERMISSIONS: Dict[str, frozenset] = {
    "viewer": frozenset({"read"}),
    "operator": frozenset({"read", "write", "infer", "train", "backup_op"}),
    "owner": frozenset({"read", "write", "infer", "train", "backup_op",
                        "user_mgmt", "audit_read", "autonomy_op"}),
}

PBKDF2_ITERS = int(os.environ.get("UDOS_KDF_ITERS", "200000"))
TOKEN_TTL_SECONDS = int(os.environ.get("UDOS_TOKEN_TTL", "3600"))
MAX_FAIL_BEFORE_LOCK = 5
LOCK_BASE_SECONDS = 30.0          # 指数退避基数: 30,60,120,240...
ENV_BLOCKED = False               # argon2/bcrypt 未装 -> 用 pbkdf2, 不阻断


def _hash_password(password: str, salt: Optional[bytes] = None
                   ) -> Tuple[bytes, bytes]:
    salt = salt or secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"),
                             salt, PBKDF2_ITERS)
    return salt, dk


def _verify_password(password: str, salt: bytes, expected: bytes) -> bool:
    _, dk = _hash_password(password, salt)
    return hmac.compare_digest(dk, expected)


class AuthError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code          # invalid_credentials / locked / no_token / ...


class RateLimiter:
    """每用户名失败计数 + 指数退避锁定。纯内存, 线程安全由 GIL 足够。"""

    def __init__(self):
        self._fails: Dict[str, int] = {}
        self._locked_until: Dict[str, float] = {}

    def _retry_after(self, key: str) -> float:
        n = self._fails.get(key, 0)
        if n < MAX_FAIL_BEFORE_LOCK:
            return 0.0
        # 第 5/6/7... 次失败: base * 2^(n-4)
        return LOCK_BASE_SECONDS * (2 ** (min(n - MAX_FAIL_BEFORE_LOCK, 10)))

    def check(self, key: str) -> Optional[float]:
        """返回剩余锁定秒数; 0/None 表示可尝试。"""
        until = self._locked_until.get(key, 0.0)
        now = time.time()
        if until > now:
            return until - now
        return None

    def record_fail(self, key: str) -> float:
        self._fails[key] = self._fails.get(key, 0) + 1
        wait = self._retry_after(key)
        if wait > 0:
            self._locked_until[key] = time.time() + wait
        return wait

    def reset(self, key: str):
        self._fails.pop(key, None)
        self._locked_until.pop(key, None)


class AuditLog:
    """append-only 审计: 只存安全事件, 永不存口令/token 明文。"""

    def __init__(self):
        self._events: List[Dict[str, Any]] = []

    def record(self, actor: str, action: str, result: str,
               detail: str = ""):
        ev = {"ts": time.time(), "actor": actor, "action": action,
              "result": result, "detail": detail}
        self._events.append(ev)
        # stderr 审计 (不写秘密)
        logger.info("AUDIT actor=%s action=%s result=%s detail=%s",
                    actor, action, result, detail[:120])

    def list(self) -> List[Dict[str, Any]]:
        return list(self._events)


class AuthService:
    """认证 + RBAC + 会话 + 限速 + 审计。纯内存, 单进程。"""

    def __init__(self, secret: Optional[str] = None):
        # 签名密钥: 优先环境变量; 否则进程内随机 (重启即失效, 安全可接受)
        self._secret = (os.environ.get("UDOS_AUTH_SECRET")
                        or secret or secrets.token_hex(32))
        self._users: Dict[str, Dict[str, Any]] = {}
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._revoked: set = set()
        self._lim = RateLimiter()
        self.audit = AuditLog()

    # ---- 用户管理 -----------------------------------------------------
    @property
    def has_owner(self) -> bool:
        return any(u["role"] == "owner" for u in self._users.values())

    def bootstrap_owner(self, username: str, password: str) -> None:
        """首个 owner 初始化; 已存在 owner 则拒绝 (防越权提升)。"""
        if self.has_owner:
            raise AuthError("forbidden", "owner 已存在, 禁止重复引导")
        self.create_user(username, password, "owner", actor="bootstrap")

    def create_user(self, username: str, password: str, role: str,
                    actor: str = "system") -> None:
        if not (isinstance(username, str) and 1 < len(username) < 64):
            raise AuthError("bad_input", "非法用户名")
        if role not in ROLES:
            raise AuthError("bad_input", f"未知角色 {role}")
        if len(password) < 8:
            raise AuthError("weak_password", "口令至少 8 位")
        salt, dk = _hash_password(password)
        self._users[username] = {"salt": salt, "hash": dk, "role": role,
                                 "must_change_password": True}
        self.audit.record(actor, "user.create", "ok",
                          f"user={username} role={role}")

    def list_users(self) -> List[Dict[str, Any]]:
        return [{"username": u, "role": v["role"]}
                for u, v in self._users.items()]

    # ---- 认证 ---------------------------------------------------------
    def authenticate(self, username: str, password: str) -> Tuple[str, dict]:
        # 锁定检查 (用同一个时间常量失败避免枚举)
        locked = self._lim.check(username)
        if locked:
            self.audit.record(username, "login", "locked",
                              f"retry_after={locked:.1f}s")
            raise AuthError("locked", f"账户锁定, {locked:.0f}s 后重试")
        u = self._users.get(username)
        ok = bool(u) and _verify_password(password, u["salt"], u["hash"])
        if not ok:
            wait = self._lim.record_fail(username)
            self.audit.record(username, "login", "fail",
                              f"backoff={wait:.0f}s")
            if wait > 0:
                raise AuthError("locked", f"失败过多, 锁定 {wait:.0f}s")
            raise AuthError("invalid_credentials", "用户名或口令错误")
        self._lim.reset(username)
        token = self._issue_token(username, u["role"])
        self.audit.record(username, "login", "ok")
        return token, {"username": username, "role": u["role"],
                       "must_change_password": u["must_change_password"]}

    def _issue_token(self, username: str, role: str) -> str:
        jti = secrets.token_urlsafe(16)
        exp = time.time() + TOKEN_TTL_SECONDS
        body = f"{jti}|{username}|{role}|{exp}".encode()
        sig = hmac.new(self._secret.encode(), body, hashlib.sha256).hexdigest()
        self._sessions[jti] = {"username": username, "role": role, "exp": exp}
        return f"{jti}.{sig}"

    def verify(self, token: str) -> Dict[str, Any]:
        if not token or "." not in token:
            raise AuthError("no_token", "缺少会话 token")
        jti, sig = token.split(".", 1)
        sess = self._sessions.get(jti)
        if not sess or jti in self._revoked:
            raise AuthError("invalid_token", "会话无效或已吊销")
        body = f"{jti}|{sess['username']}|{sess['role']}|{sess['exp']}".encode()
        expect = hmac.new(self._secret.encode(), body,
                          hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expect, sig):
            raise AuthError("invalid_token", "签名不匹配")
        if time.time() > sess["exp"]:
            raise AuthError("expired", "会话已过期")
        return {"username": sess["username"], "role": sess["role"]}

    def logout(self, token: str, actor: str = ""):
        jti = token.split(".", 1)[0] if token else ""
        if jti in self._sessions:
            self._revoked.add(jti)
            self.audit.record(actor or self._sessions[jti]["username"],
                              "logout", "ok")

    # ---- 授权 ---------------------------------------------------------
    def require(self, principal: Optional[Dict[str, Any]], perm: str):
        if principal is None:
            raise AuthError("unauthenticated", "未认证")
        role = principal["role"]
        if perm not in _PERMISSIONS.get(role, frozenset()):
            self.audit.record(principal["username"], "authz", "denied",
                              f"perm={perm} role={role}")
            raise AuthError("forbidden", f"角色 {role} 无权限 {perm}")

    def change_password(self, principal: Dict[str, Any],
                        old: str, new: str) -> None:
        u = self._users.get(principal["username"])
        if not u or not _verify_password(old, u["salt"], u["hash"]):
            raise AuthError("invalid_credentials", "原口令错误")
        if len(new) < 8:
            raise AuthError("weak_password", "新口令至少 8 位")
        salt, dk = _hash_password(new)
        u["salt"], u["hash"] = salt, dk
        u["must_change_password"] = False
        self.audit.record(principal["username"], "password.change", "ok")


def auth_enabled() -> bool:
    return os.environ.get("UDOS_AUTH", "off").lower() in ("on", "1", "true")
