"""结构化日志。

规范见 docs/01-开发规范.md 第七条：

- 使用 structlog 输出 JSON 行；本地开发可切人类可读渲染器。
- 禁止 ``print()``（CLI 面向用户的输出除外）。
- 日志中**严禁**出现 token、账号、密码；账号必须脱敏。
"""

from __future__ import annotations

import logging
import re
import sys
from typing import Any, Final, TextIO

import structlog

__all__ = ["get_logger", "mask_account", "mask_secret", "setup_logging"]

_SENSITIVE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "api_key",
        "apikey",
        "token",
        "secret",
        "password",
        "passwd",
        "authorization",
        "cookie",
        "anthropic_api_key",
        "tushare_token",
        "jin10_api_key",
        "jin10_api_secret",
        "intel_ingest_token",
        "webhook_url",
        "account_id",
        "fund_account",
        "xtquant_account_id",
    }
)

_SECRET_PATTERN: Final = re.compile(
    r"(sk-[A-Za-z0-9\-_]{8,}|[A-Fa-f0-9]{32,})",
)

_KEEP_TAIL: Final = 4
"""脱敏后保留的尾部字符数，用于人工核对是不是同一个密钥/账号。"""


def mask_secret(value: str) -> str:
    """脱敏密钥，只保留尾部 4 位。

    Args:
        value: 原始密钥。

    Returns:
        形如 ``••••1234`` 的脱敏串；空值返回 ``<empty>``。
    """
    if not value:
        return "<empty>"
    if len(value) <= _KEEP_TAIL:
        return "•" * len(value)
    return f"••••{value[-_KEEP_TAIL:]}"


def mask_account(value: str) -> str:
    """脱敏资金账号，只保留尾部 4 位。

    Args:
        value: 原始账号。

    Returns:
        形如 ``****1234`` 的脱敏账号。
    """
    if not value:
        return "<empty>"
    if len(value) <= _KEEP_TAIL:
        return "*" * len(value)
    return f"****{value[-_KEEP_TAIL:]}"


def _redact_processor(
    _logger: Any,  # noqa: ANN401 - structlog 处理器签名由框架规定
    _method: str,
    event_dict: structlog.types.EventDict,
) -> structlog.types.EventDict:
    """脱敏处理器：按 key 名与内容模式两路拦截敏感信息。

    Args:
        _logger: structlog 内部参数，未使用。
        _method: structlog 内部参数，未使用。
        event_dict: 待输出的事件字典。

    Returns:
        脱敏后的事件字典。
    """
    for key, value in list(event_dict.items()):
        lowered = key.lower()
        if lowered in _SENSITIVE_KEYS:
            event_dict[key] = mask_secret(str(value))
        elif isinstance(value, str) and _SECRET_PATTERN.search(value):
            event_dict[key] = _SECRET_PATTERN.sub(lambda m: mask_secret(m.group()), value)
    return event_dict


def setup_logging(
    *,
    level: str = "INFO",
    fmt: str = "json",
    stream: TextIO | None = None,
) -> None:
    """初始化日志系统。

    应在进程启动最早期调用一次（CLI 入口与 Web 入口各调用一次即可）。

    Args:
        level: 日志级别名，如 ``INFO`` / ``DEBUG``。
        fmt: ``json`` 输出结构化 JSON 行；``console`` 输出人类可读彩色文本。
        stream: 输出流，默认 stderr。
    """
    processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=False),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        _redact_processor,
    ]
    if fmt == "console":
        processors.append(structlog.dev.ConsoleRenderer(colors=True))
    else:
        processors.append(structlog.processors.JSONRenderer(ensure_ascii=False))

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelNamesMapping()[level.upper()]
        ),
        logger_factory=structlog.PrintLoggerFactory(file=stream or sys.stderr),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """取一个带模块名的 logger。

    Args:
        name: 模块名，通常传 ``__name__``。

    Returns:
        绑定了 ``module`` 字段的 logger。
    """
    logger: structlog.stdlib.BoundLogger = structlog.get_logger(name).bind(module=name)
    return logger
