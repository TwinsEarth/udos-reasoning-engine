# UDOS 安全与受控自治说明（v5.0.1）

> 本文件是 v5.0.1 的安全权威声明。配合 VERIFICATION_v5.0.1.md 的契约证据阅读。

## 0. 安全声明（明确写出"未实现"）

UDOS v5.0.1 **未实现、也坚决不会**：
- ❌ 任何硬编码用户名/口令/token/万能凭据/隐藏或后门账号；
- ❌ 任何针对微信/抖音/小红书/淘宝/滴滴/美团等第三方平台的注入、自动登录、弹窗、界面劫持或关键词触发；不在豆包全局监听关键词；
- ❌ 自我复制/蠕虫式发现并安装到其他主机、未授权占用他人算力/存储、隐蔽持久化；
- ❌ 采集/存储/上传原始人脸/指纹/声纹等生物模板；
- ❌ 无审批/无评估门自动替换生产权重、不可关停的后台行为。

源码、配置模板、测试、日志、CHANGELOG 中均无真实口令；口令只经 PBKDF2 哈希存储，审计日志绝不记口令或 token 明文。

> **P0 修复（v5.0.1 安全补丁）**：历史 `udos/debug.py` 曾硬编码调试面板解锁口令并在 README/demos/QA 文档中复述，已**全部移除**：`DebugPanel` 改为 `enabled` 显式参数 + `UDOS_DEBUG=1` 环境变量控制（默认关），不接受任何口令。现已与"无硬编码凭据"声明一致。安全守卫测试 `tests/test_security_guard_v501.py` 钉死全仓无该口令/姓名字面量。

## 1. 威胁模型

- 网络层：默认 `UDOS_AUTH=off` 用于本地演示；生产设 `UDOS_AUTH=on`。`/health`、`/metrics` 默认保留（探针不应被认证挡住），其余端点需认证。
- 暴力破解：登录失败指数退避 + 5 次锁定。
- 会话：HMAC 签名 token + 过期 + 服务端可吊销（登出/权限变更立即生效）。
- 越权：RBAC 垂直（viewer/operator/owner）与水平（owner 才能管用户/审计/自治）。

## 2. 认证与 RBAC（udos/auth.py, authz.py）

- 角色矩阵：
  - viewer = read
  - operator = read + write + infer + train + backup_op
  - owner = 全部 + user_mgmt + audit_read + autonomy_op
- 口令 KDF：`hashlib.pbkdf2_hmac(sha256, salt, 200000 次)`。
  - 说明：本环境离线装 argon2/bcrypt 受限，用标准库 pbkdf2 即满足"不明文/不弱哈希"（ENV_BLOCKED 标注）。
- 首个 owner：只能通过 `/auth/bootstrap`（首次、无 owner 时）创建；已存在 owner 再引导直接拒绝。无默认口令。
- 速率限制：5 次失败后指数退避锁定（30s 起翻倍）。
- 审计：append-only，记录 actor/action/result/时间，不含秘密。

## 3. WebAuthn / MFA（udos/webauthn_mfa.py）

- 注册/断言接口契约完整；**指纹/人脸比对只在设备 OS 内完成**，应用只验签名断言、只存公开凭证 id + 公钥。
- 本沙箱无浏览器/认证器、无 fido2 库 → `ENV_BLOCKED=True`，真实验签返回 False，**不伪造认证成功**。
- 不支持自建声纹/人脸库：存储不可逆生物模板等于终身泄露风险，WebAuthn 把比对留在设备内是更合规选择。

## 4. 受控联网（udos/netops.py）

- opt-in（`UDOS_NET=on`）；只访问配置白名单 URL；显式超时/审计；白名单外一律拒绝；不绕防火墙、不主机发现。

## 5. 加密备份/恢复（udos/backup.py）

- 目标显式配置（本地目录或用户自有 S3 兼容 endpoint，凭据只从环境变量读，不入代码）。
- PBKDF2 派生密钥 + 流加密 + HMAC-SHA256 完整性；恢复时先验 HMAC，篡改/口令错一律拒绝。
- roundtrip 测试证明解密一致；保留策略自动轮转。

## 6. 受控自治（udos/autonomy.py）

- 默认关（opt-in）。阶段：数据校验 → 候选 → **隔离评估门**（指标不达标不晋级，沿用 v4.2 自蒸馏崩塌教训）→ 人工批准 → 一键回滚。
- 默认**不自动替换生产权重**；CPU/时间/磁盘预算硬停止；kill switch 立即生效并关闭自治；全链路审计。
- 交付时零训练、不改主权重。

## 7. 部署 hardening 清单

1. 生产设 `UDOS_AUTH=on`，经 `/auth/bootstrap` 初始化 owner 并立即改密。
2. `UDOS_AUTH_SECRET` 从密钥管理注入，不用默认随机（否则重启会话失效）。
3. 备份口令 `UDOS_BACKUP_PASSWORD`、S3 凭据只从环境变量读。
4. 自治默认关；确需开启先配评估门阈值与预算。
5. `/metrics`、`/health` 仅在可信网络暴露；反代层再加 TLS。
