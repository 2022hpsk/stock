"""日志脱敏测试。

红线 R7：日志中严禁出现 token、账号、密码。这里覆盖按 key 名与按内容模式两条拦截路径。
"""

from __future__ import annotations

import io
import json

import pytest

from quantstock.infra.logging import (
    _redact_processor,
    get_logger,
    mask_account,
    mask_secret,
    setup_logging,
)


class TestMasking:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("", "<empty>"),
            ("ab", "••"),
            ("abcd", "••••"),
            ("sk-1234567890", "••••7890"),
        ],
    )
    def test_mask_secret(self, raw: str, expected: str) -> None:
        assert mask_secret(raw) == expected

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("", "<empty>"), ("12", "**"), ("123456789", "****6789")],
    )
    def test_mask_account(self, raw: str, expected: str) -> None:
        assert mask_account(raw) == expected

    def test_mask_secret__keeps_tail_for_human_verification(self) -> None:
        """保留尾 4 位是为了让人能核对"是不是同一个密钥"，而不泄漏密钥本身。"""
        assert mask_secret("sk-super-secret-1234").endswith("1234")


class TestRedactProcessor:
    def test_sensitive_key__masked(self) -> None:
        event = {"event": "call", "api_key": "sk-abcdefgh1234"}
        result = _redact_processor(None, "info", event)
        assert result["api_key"] == "••••1234"

    def test_account_id__masked(self) -> None:
        result = _redact_processor(None, "info", {"account_id": "880012345678"})
        assert result["account_id"] == "••••5678"

    def test_secret_in_free_text__masked_by_pattern(self) -> None:
        """即使 key 名无害，值里的密钥模式也要拦截。"""
        event = {"event": "failed", "detail": "auth error with sk-abcdefgh1234 token"}
        result = _redact_processor(None, "info", event)
        assert "sk-abcdefgh1234" not in result["detail"]
        assert "••••1234" in result["detail"]

    def test_ordinary_field__untouched(self) -> None:
        event = {"event": "bar_loaded", "symbol": "600519.SH", "rows": 2400}
        assert _redact_processor(None, "info", event) == event


class TestSetupLogging:
    def test_json_output__is_parseable_and_redacted(self) -> None:
        stream = io.StringIO()
        setup_logging(level="INFO", fmt="json", stream=stream)
        get_logger("test").info("data_updated", symbol="600519.SH", token="sk-abcdefgh1234")

        record = json.loads(stream.getvalue().strip())
        assert record["event"] == "data_updated"
        assert record["symbol"] == "600519.SH"
        assert record["module"] == "test"
        assert record["token"] == "••••1234"
        assert "sk-abcdefgh1234" not in stream.getvalue()

    def test_level_filtering(self) -> None:
        stream = io.StringIO()
        setup_logging(level="WARNING", fmt="json", stream=stream)
        get_logger("test").info("should_not_appear")
        get_logger("test").warning("should_appear")

        output = stream.getvalue()
        assert "should_not_appear" not in output
        assert "should_appear" in output

    def test_console_format__renders(self) -> None:
        stream = io.StringIO()
        setup_logging(level="INFO", fmt="console", stream=stream)
        get_logger("test").info("hello")
        assert "hello" in stream.getvalue()
