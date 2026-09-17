"""
UDOS 统一日志配置 (v3.3.4 hardening)
=====================================
全引擎共享一套 logging 基础设施, 严格保证:
  1. 所有 handler 只输出到 stderr, 绝不写 stdout / HTTP 响应体;
  2. 默认级别 WARNING, 保证 pytest 运行干净 (INFO/DEBUG 不刷屏);
  3. 环境变量 UDOS_LOG_LEVEL 可覆盖级别 (DEBUG/INFO/WARNING/ERROR);
  4. 服务入口 main() 显式 configure_logging(level=INFO)。

设计要点 (与 pytest caplog 兼容):
  - handler 挂在 root logger 上, 用 _UdosOnlyFilter 只放行 ``udos.*`` 记录,
    这样 udo 记录既进 stderr handler, 又能沿传播链到达 root 上 pytest 的
    LogCaptureHandler, caplog 才能断言到; 同时不污染第三方库日志。
  - 级别控制作用在包 logger ``udos`` 上 (而非全局 root), 不改变其它库的行为。
  - 重复调用幂等: 先移除打了标记的旧 handler, 再挂新 handler。
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional, TextIO

# 名称 -> logging 级别常量
_LEVEL_MAP = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "WARN": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

# 挂在 handler 上的标记属性名, 用于幂等识别/移除
_TAG = "_udos_logging_handler"

# 统一格式: 时间(ISO) 级别  logger名  消息
_FMT = "%(asctime)s %(levelname)s %(name)s %(message)s"
_DATEFMT = "%Y-%m-%dT%H:%M:%S"


class _UdosOnlyFilter(logging.Filter):
    """只放行 udo 命名空间 (udos 及其子 logger) 的记录。"""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401
        return record.name == "udos" or record.name.startswith("udos.")


def _resolve_level(level) -> int:
    """把 None / 字符串 / int 统一解析为 logging 级别常量。"""
    if level is None:
        env = os.environ.get("UDOS_LOG_LEVEL", "").strip().upper()
        return _LEVEL_MAP.get(env, logging.WARNING)
    if isinstance(level, str):
        return _LEVEL_MAP.get(level.strip().upper(), logging.WARNING)
    return int(level)


def configure_logging(level=None, stream: Optional[TextIO] = None) -> logging.Handler:
    """配置 UDOS 统一日志。

    Args:
        level: None -> 读 UDOS_LOG_LEVEL, 缺省 WARNING; 也可传
            "DEBUG"/"INFO"/"WARNING"/"ERROR" 或 logging 常量。
        stream: 输出流, 默认 sys.stderr。**绝不默认 stdout**。

    Returns:
        新建的 stderr handler (便于测试校验 stream)。
    """
    resolved = _resolve_level(level)
    out_stream = stream if stream is not None else sys.stderr

    root = logging.getLogger()
    pkg = logging.getLogger("udos")

    # 幂等: 移除此前由本函数挂上的 handler (root 上 + 包 logger 上)
    for h in list(root.handlers):
        if getattr(h, _TAG, False):
            root.removeHandler(h)
    for h in list(pkg.handlers):
        if getattr(h, _TAG, False):
            pkg.removeHandler(h)

    formatter = logging.Formatter(_FMT, datefmt=_DATEFMT)
    handler = logging.StreamHandler(stream=out_stream)
    handler.setFormatter(formatter)
    handler.addFilter(_UdosOnlyFilter())
    setattr(handler, _TAG, True)
    root.addHandler(handler)

    # 级别作用在包 logger 上, 不抬全局 root 级别 (避免影响第三方库)
    pkg.setLevel(resolved)
    return handler


def get_udos_handler() -> Optional[logging.Handler]:
    """返回当前挂在 root 上的 udo handler (测试/自检用), 没有则 None。"""
    for h in logging.getLogger().handlers:
        if getattr(h, _TAG, False):
            return h
    return None
