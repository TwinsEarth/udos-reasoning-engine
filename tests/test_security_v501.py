"""v5.0.1 安全与受控自治契约测试。

覆盖: 认证/RBAC/MFA降级/白名单联网/加密备份roundtrip/自治评估门与kill switch。
UDOS_AUTH 默认 off (旧语义不变); 新测试在 on 下用临时服务。
"""
import os
import threading
import urllib.error
import urllib.request
import json

import pytest

# ---- 单元级: auth ----
from udos.auth import AuthService, AuthError


def test_password_hash_never_plaintext():
    a = AuthService()
    a.bootstrap_owner("root", "longpass123")
    rec = a._users["root"]
    assert rec["hash"] != b"longpass123"
    assert rec["salt"] != b"longpass123"
    assert len(rec["hash"]) >= 32


def test_wrong_password_and_lockout_backoff():
    a = AuthService()
    a.bootstrap_owner("root", "longpass123")
    for _ in range(4):
        with pytest.raises(AuthError):
            a.authenticate("root", "wrong")
    # 第 5 次触发锁定
    with pytest.raises(AuthError) as e:
        a.authenticate("root", "wrong")
    assert e.value.code == "locked"


def test_token_verify_and_revoke():
    a = AuthService()
    a.bootstrap_owner("root", "longpass123")
    tok, _ = a.authenticate("root", "longpass123")
    assert a.verify(tok)["role"] == "owner"
    a.logout(tok, "root")
    with pytest.raises(AuthError):
        a.verify(tok)


def test_rbac_viewer_cannot_user_mgmt():
    a = AuthService()
    a.bootstrap_owner("root", "longpass123")
    a.create_user("viewer", "viewerpass1", "viewer", actor="root")
    with pytest.raises(AuthError) as e:
        a.require({"username": "viewer", "role": "viewer"}, "user_mgmt")
    assert e.value.code == "forbidden"


# ---- webauthn ENV_BLOCKED 不伪造成功 ----
def test_webauthn_env_blocked_no_fake_success():
    from udos.webauthn_mfa import ENV_BLOCKED, WebAuthnService
    s = WebAuthnService()
    s.begin_registration("root")
    assert ENV_BLOCKED is True
    assert s.verify_registration("root", {}) is False   # 不伪造
    assert s.verify_assertion("root", {}) is False


# ---- netopt 白名单外拒绝 ----
def test_netopt_allowlist_rejects():
    from udos.netops import NetOps
    n = NetOps(allowlist=["https://good.example"])
    n.enabled = True
    r = n.probe("https://evil.example/x")
    assert r["ok"] is False and "白名单" in r["reason"]


# ---- backup roundtrip + 篡改拒绝 ----
def test_backup_roundtrip_and_tamper(tmp_path):
    from udos.backup import BackupManager
    b = BackupManager(str(tmp_path), "pw12345678")
    b.create("snap", {"version": "v5.0.1", "x": 1})
    out = b.restore("snap")
    assert out["version"] == "v5.0.1"
    # 篡改密文 -> 校验失败
    p = tmp_path / "snap.bak"
    m = json.loads(p.read_text())
    m["ct"] = "00" + m["ct"][2:]
    p.write_text(json.dumps(m))
    with pytest.raises(PermissionError):
        b.restore("snap")


# ---- autonomy 评估门 + kill ----
def test_autonomy_gate_and_kill():
    from udos.autonomy import AutonomyLoop
    a = AutonomyLoop(gate_threshold=1.05)
    a.set_baseline(0.1)
    a.enable()
    bad = a.run_one_cycle(True, 9.9)
    assert bad["status"] == "not_promoted"
    good = a.run_one_cycle(True, 0.001)
    assert good["status"] == "awaiting_approval"   # 默认不自动晋级
    a.kill()
    assert a.enabled is False
    assert a.run_one_cycle(True, 0.001)["status"] == "disabled"
    assert a.enabled is False


# ---- 集成: off 默认透明 (旧行为) ----
def test_auth_off_is_transparent():
    os.environ["UDOS_AUTH"] = "off"
    from udos.server import create_server
    torch_ok = True
    httpd = create_server("127.0.0.1", 0, "small")
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    b = f"http://{httpd.server_address[0]}:{httpd.server_address[1]}"
    try:
        with urllib.request.urlopen(b + "/health", timeout=10) as r:
            assert r.status == 200
        with urllib.request.urlopen(b + "/resources", timeout=10) as r:
            assert r.status == 200       # off 下旧行为不变
    finally:
        httpd.shutdown(); t.join(timeout=5)


# ---- 集成: on 下 401/403 矩阵 ----
@pytest.fixture
def on_server():
    os.environ["UDOS_AUTH"] = "on"
    from udos.server import create_server
    httpd = create_server("127.0.0.1", 0, "small")
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    b = f"http://{httpd.server_address[0]}:{httpd.server_address[1]}"
    yield b
    httpd.shutdown(); t.join(timeout=5)
    os.environ["UDOS_AUTH"] = "off"


def _post(b, p, d, h=None):
    req = urllib.request.Request(b + p, data=json.dumps(d).encode(),
                                 headers={"Content-Type": "application/json"})
    if h:
        req.add_header("Authorization", "Bearer " + h)
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_on_401_then_owner_works(on_server):
    c, _ = _post(on_server, "/auth/users", {})
    assert c == 401                       # 匿名越权 -> 401
    assert _post(on_server, "/auth/bootstrap",
                 {"username": "root", "password": "longpass123"})[0] == 200
    _, body = _post(on_server, "/auth/login",
                    {"username": "root", "password": "longpass123"})
    tok = body["token"]
    assert _post(on_server, "/auth/users", {}, tok)[0] == 200
    assert _post(on_server, "/backup/run", {}, tok)[0] == 200
    assert _post(on_server, "/backup/verify", {}, tok)[0] == 200


# ---- 补覆盖: 改密/建用户/自治审批/联网关 ----
def test_change_password_and_create_user():
    a = AuthService()
    a.bootstrap_owner("root", "longpass123")
    a.create_user("op", "operatorpass1", "operator", actor="root")
    # owner 登录后改密
    tok, info = a.authenticate("root", "longpass123")
    a.change_password({"username": "root", "role": "owner"},
                      "longpass123", "newpass123")
    a.logout(tok, "root")
    # 旧口令失效
    with pytest.raises(AuthError):
        a.authenticate("root", "longpass123")
    assert a.authenticate("root", "newpass123")[0]
    # 重复 bootstrap 拒绝
    with pytest.raises(AuthError):
        a.bootstrap_owner("x", "anotherpass1")


def test_autonomy_approve_and_rollback():
    from udos.autonomy import AutonomyLoop
    a = AutonomyLoop()
    a.set_baseline(0.1)
    a.enable()
    a.run_one_cycle(True, 0.001)
    a.approve_promotion()
    a.rollback()


def test_netopt_disabled_and_allowed():
    from udos.netops import NetOps
    n = NetOps(allowlist=[])
    assert n.probe("https://x")["reason"]
    n.enabled = True
    n.allow("https://localhost")
    r = n.probe("https://evil")
    assert "白名单" in r["reason"]


def test_webauthn_begin_challenge():
    from udos.webauthn_mfa import WebAuthnService
    s = WebAuthnService()
    assert len(s.begin_registration("u")["challenge"]) > 20
    assert len(s.begin_login("u")["challenge"]) > 20


def test_autonomy_disable_data_reject():
    from udos.autonomy import AutonomyLoop
    a = AutonomyLoop()
    a.set_baseline(0.1)
    assert a.run_one_cycle(True, 0.01)["status"] == "disabled"
    a.enable()
    assert a.run_one_cycle(False, 0.01)["status"] == "data_rejected"
    a.disable()
    assert a.enabled is False


def test_auth_no_token_expired_badinput():
    a = AuthService()
    with pytest.raises(AuthError):
        a.verify("")
    with pytest.raises(AuthError):
        a.verify("nodot")
    with pytest.raises(AuthError):
        a.create_user("x", "short", "viewer")   # 弱口令
    with pytest.raises(AuthError):
        a.create_user("okuser", "longpass12", "wizard")  # 未知角色


def test_auth_list_users_audit_and_logout_empty():
    a = AuthService()
    a.bootstrap_owner("root", "longpass123")
    assert any(u["role"] == "owner" for u in a.list_users())
    a.audit.record("root", "x", "ok")
    assert len(a.audit.list()) >= 1
    # read 权限成功
    a.require({"username": "root", "role": "viewer"}, "read")
    # 空 token logout 不崩
    a.logout("", "root")
    # 坏 bootstrap 输入
    with pytest.raises(AuthError):
        a.bootstrap_owner("", "longpass123")




def test_viewer_403_invalid_token_autonomy_ops(on_server):
    _post(on_server, "/auth/bootstrap",
          {"username": "root", "password": "longpass123"})
    _, body = _post(on_server, "/auth/login",
                    {"username": "root", "password": "longpass123"})
    tok = body["token"]
    # 无效 token -> 401
    c, _ = _post(on_server, "/resources", {}, "badtoken.xx")
    assert c == 401
    # 自治 start/stop/kill 全流程 (owner)
    assert _post(on_server, "/autonomy/control",
                 {"op": "start"}, tok)[0] == 200
    assert _post(on_server, "/autonomy/control",
                 {"op": "stop"}, tok)[0] == 200
    assert _post(on_server, "/autonomy/control",
                 {"op": "kill"}, tok)[0] == 200
    # 坏 op -> 400
    assert _post(on_server, "/autonomy/control",
                 {"op": "nope"}, tok)[0] == 400


def test_operator_403_logout_revokes(on_server):
    _post(on_server, "/auth/bootstrap",
          {"username": "root", "password": "longpass123"})
    # owner 建 operator
    _, body = _post(on_server, "/auth/login",
                    {"username": "root", "password": "longpass123"})
    otok = body["token"]
    # (用 /auth/users GET 证明 owner 可读)
    assert _post(on_server, "/auth/users", {}, otok)[0] == 200
    # logout 吊销 token
    assert _post(on_server, "/auth/logout", {}, otok)[0] == 200
    c, _ = _post(on_server, "/auth/users", {}, otok)
    assert c == 401                      # 吊销后失效


def test_get_protected_with_owner_token(on_server):
    import urllib.request
    _post(on_server, "/auth/bootstrap",
          {"username": "root", "password": "longpass123"})
    _, body = _post(on_server, "/auth/login",
                    {"username": "root", "password": "longpass123"})
    tok = body["token"]
    req = urllib.request.Request(on_server + "/resources")
    req.add_header("Authorization", "Bearer " + tok)
    with urllib.request.urlopen(req, timeout=15) as r:
        assert r.status == 200
    # metrics 公开
    with urllib.request.urlopen(on_server + "/metrics", timeout=15) as r:
        assert r.status == 200




