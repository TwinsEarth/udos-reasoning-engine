"""UDOS 授权中间件（v5.0.1，合规模块 B）。

在 UDOS_AUTH=on 时为现有 HTTP 加认证/授权：
  - 未认证 -> 401；权限不足 -> 403（与既有 400/404/409/500 一致）；
  - /health、/metrics 默认保留（本地演示/健康探针不应被认证挡住）；
  - UDOS_AUTH=off 时完全放行，旧行为/旧测试语义零变化。

本模块只做判定，不存储口令，不外联。
"""
from __future__ import annotations

from typing import Dict, Optional

from .auth import AuthError, AuthService, auth_enabled

# 公开路由 (无需认证): 健康/指标/引导/登录本身
PUBLIC_GET = {"/health", "/metrics", "/demo"}
PUBLIC_POST = {"/auth/login", "/auth/bootstrap", "/auth/change-password",
               "/auth/logout"}

# 写/管理路由需要的最小权限
_WRITE_PERM = "write"
_ROUTES_NEED = {
    # operator 可写/推理/训练类
    "/predict": "infer", "/reason": "infer", "/reason/latent": "infer",
    "/reset": "write", "/internalize": "infer",
    "/train": "train", "/evaluate": "infer", "/save": "backup_op",
    "/load": "backup_op", "/rollback": "autonomy_op",
    "/auth/users": "user_mgmt", "/auth/audit": "audit_read",
    "/backup/run": "backup_op", "/backup/verify": "backup_op",
    "/autonomy/start": "autonomy_op", "/autonomy/stop": "autonomy_op",
    "/autonomy/kill": "autonomy_op",
}


class AuthzMiddleware:
    """把 auth 服务挂进 HTTP 层; off 时是透明直通。"""

    def __init__(self, auth: Optional[AuthService] = None):
        self.auth = auth

    @property
    def enabled(self) -> bool:
        return auth_enabled() and self.auth is not None

    def principal_from_headers(self, headers) -> Optional[Dict]:
        """从 Authorization: Bearer <token> 解析主体验证。off 时 None。"""
        if not self.enabled:
            return None
        h = headers.get("Authorization", "") or headers.get("authorization", "")
        if not h.startswith("Bearer "):
            # 无 token: 返回 None, 由 authorize() 对非公开路由判 401;
            # 公共路由 (/auth/login 等) 不需 token, 不在这里崩。
            return None
        token = h[len("Bearer "):].strip()
        return self.auth.verify(token)

    def authorize(self, method: str, path: str, principal) -> None:
        """off 直通; on 时按路由要求的权限判定。"""
        if not self.enabled:
            return
        route = path.split("?", 1)[0].rstrip("/")
        if method == "GET" and route in PUBLIC_GET:
            return
        if method == "POST" and route in PUBLIC_POST:
            return
        # GET 只读默认需要 read (即任何登录用户)
        if method == "GET":
            perm = "read"
        else:
            perm = _ROUTES_NEED.get(route, _WRITE_PERM)
        if principal is None:
            raise AuthError("unauthenticated", "未认证")
        self.auth.require(principal, perm)
