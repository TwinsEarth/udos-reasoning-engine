# UDOS v5.0.1 安全与受控自治验证报告

> release decision: **go**（含 P0 口令硬伤补丁后重新判定）

## 0. P0 硬编码口令修复（安全补丁）

- **发现**：`udos/debug.py` 硬编码一个调试面板解锁口令（`_DEBUG_PASSWORD` 常量）并在 README/demos/QA 文档复述。
- **根因**：早期调试面板按"带密码解锁"设计，口令作为共享密钥硬编码在源码。
- **修复**：
  - `udos/debug.py` 删除口令常量与比对；`DebugPanel(level, enabled=None)`，enabled 缺省读 `UDOS_DEBUG`（默认 0=关）；输出走 stderr。
  - 清除 README/demos×3/QA_HARDENING/scratch 全部复述。
- **修复前→修复后 grep 证据**：
  - 修复前：grep 命中 7 处（debug.py、README、demos×3、QA、scratch）。
  - 修复后：从全新解压树对该口令字面量与指定姓名做全树 grep，**均为空**。
- **守卫测试**：tests/test_security_guard_v501.py（3 个）钉死无口令/姓名/硬编码 secret 字面量，且含变异敏感性自检。


## 1. 模块实现与降级状态
| 模块 | 文件 | 状态 |
|---|---|---|
| A 认证+RBAC | udos/auth.py | 已实现：PBKDF2 哈希、签名会话+吊销、指数退避锁定、append-only 审计 |
| B 授权中间件 | udos/authz.py | 已实现：UDOS_AUTH=on|off，未认证 401/越权 403，off 透明 |
| C WebAuthn MFA | udos/webauthn_mfa.py | 接口契约完成；**ENV_BLOCKED**（无认证器/fido2，不伪造成功） |
| D 受控联网 | udos/netops.py | opt-in 白名单，白名单外拒绝 |
| E 加密备份 | udos/backup.py | PBKDF2+流加密+HMAC，roundtrip 通过，篡改拒绝 |
| F 受控自治 | udos/autonomy.py | 默认关、评估门、kill switch、不自动替换权重 |
| G 控制台 | web/udos_console.html | 已有数据看板（安全面板见 SECURITY.md） |
| H 文档 | docs/SECURITY.md | 威胁模型+RBAC+安全声明 |

## 2. 安全契约测试（tests/test_security_v501.py，20 个，全绿）
- 口令不明文、错口令、5 次锁定退避、token 验证+登出吊销
- viewer 无 user_mgmt 权限（403）、匿名 401、无效 token 401
- webauthn ENV_BLOCKED 不伪造成功
- netopt 白名单外拒绝、备份 roundtrip + 篡改拒绝
- 自治评估门不过不晋级、kill switch、off 透明旧行为不变

## 3. 性能
认证中间件为 Bearer HMAC 校验，亚微秒级；off 时零开销（直通）。无主预测循环影响。

## 4. 测试总数 / 覆盖率
- pytest：**1522 passed / 0 failed**（v4.5.6 基线 1492 + 30 新增，只增）
- 覆盖率 **93%**（11581 stmts / 814 miss；P0 后补 debug.py 单测从 0% 拉到覆盖，miss 869→814）

## 5. 逐条"未实现"禁止项（证明）
- 无硬编码凭据：全仓 grep 无真实口令；首 owner 仅经 /auth/bootstrap。
- 无第三方平台注入/无全局关键词监听。
- 无自我复制/无主机发现/无隐蔽持久化。
- 不采集生物模板：MFA 走 WebAuthn 边界（OS 内比对），ENV_BLOCKED。
- 自治默认关、不自动替换生产权重、kill switch 可关停。

## 6. 零训练硬验收
主参 52191、eval_mse 0.045556、33 代 checkpoint、四锚点 md5 逐位不变。

## 7. 复现
```
UDOS_AUTH=off python -m pytest -q
UDOS_AUTH=on  python -m udos.server --port 8000 --preset small
# POST /auth/bootstrap -> /auth/login -> Bearer -> /auth/users
```
