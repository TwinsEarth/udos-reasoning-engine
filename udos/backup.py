"""UDOS 自有目标加密备份/恢复（v5.0.1，合规模块 E）。

  - 目标必须显式配置（本地目录）；S3 兼容 endpoint 凭据只从环境变量读，不入代码。
  - 静态加密：口令经 PBKDF2 派生密钥，流密码异或加密 + HMAC-SHA256 完整性。
  - 恢复 roundtrip 单测证明解密/校验一致。
  - 不做自主主机发现；凭据绝不硬编码。
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import time
from pathlib import Path
from typing import Any, Dict, Optional

_ITER = 120_000


def _derive(password: str, salt: bytes):
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, _ITER, dklen=64)
    return dk[:32], dk[32:]      # enc key, mac key


def _crypt_stream(key: bytes, data: bytes) -> bytes:
    # 以密钥为种子的 SHA256-CTR 流 (演示级; 生产应换 AES-GCM)
    out = bytearray()
    counter = 0
    while len(out) < len(data):
        block = hashlib.sha256(key + counter.to_bytes(8, "big")).digest()
        out.extend(block)
        counter += 1
    return bytes(b ^ k for b, k in zip(data, out[:len(data)]))


class BackupManager:
    def __init__(self, target_dir: str, password: str,
                 keep: int = 5):
        self.target = Path(target_dir)
        self.target.mkdir(parents=True, exist_ok=True)
        self._password = password
        self.keep = keep

    def _manifest(self, name: str, salt: bytes, ct: bytes, mac: bytes):
        return {"name": name, "v": 1, "salt": salt.hex(),
                "ct": ct.hex(), "mac": mac,
                "ts": time.time()}

    def create(self, name: str, payload: Dict[str, Any]) -> str:
        """加密写入一个备份。返回文件名。"""
        raw = json.dumps(payload, sort_keys=True).encode()
        salt = secrets.token_bytes(16)
        enc_key, mac_key = _derive(self._password, salt)
        ct = _crypt_stream(enc_key, raw)
        mac = hmac.new(mac_key, ct, hashlib.sha256).hexdigest()
        fn = self.target / f"{name}.bak"
        fn.write_text(json.dumps(self._manifest(name, salt, ct, mac)),
                      encoding="utf-8")
        self._rotate()
        return fn.name

    def restore(self, name: str) -> Dict[str, Any]:
        """校验 HMAC + 解密; 篡改/口令错一律拒绝。"""
        fn = self.target / f"{name}.bak"
        m = json.loads(fn.read_text(encoding="utf-8"))
        salt = bytes.fromhex(m["salt"]); ct = bytes.fromhex(m["ct"])
        enc_key, mac_key = _derive(self._password, salt)
        expect = hmac.new(mac_key, ct, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expect, m["mac"]):
            raise PermissionError("备份完整性校验失败 (篡改或口令错)")
        raw = _crypt_stream(enc_key, ct)
        return json.loads(raw.decode())

    def _rotate(self):
        files = sorted(self.target.glob("*.bak"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        for p in files[self.keep:]:
            p.unlink()

    def list(self):
        return sorted(p.name for p in self.target.glob("*.bak"))
