"""Anthropic 供应商适配器测试。

**永不调用真实 API**：SDK 的构造收在 ``_ensure_client()`` 里，测试直接把
假客户端塞进 ``_client``，整条翻译路径照常走。

这一层只负责"翻译"，所以测试关注的也只有翻译的正确性与失败时的**降级形态**：
调用失败必须变成 ``LLMError``，上层 ``LLMClient`` 才能把它降级成
"本次不使用 LLM"，继续走纯量化路径（红线 LR2：关闭 LLM 后系统须完整可用）。
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import pytest

from quantstock.infra.errors import LLMError
from quantstock.llm.protocols import ChatMessage, CompletionRequest, LLMProvider
from quantstock.llm.providers.anthropic_provider import AnthropicProvider

REQUEST = CompletionRequest(
    model="claude-sonnet-4-5",
    system="你是一个只做文本归纳的助手",
    messages=(ChatMessage(role="user", content="总结这条公告"),),
    max_tokens=512,
    temperature=0.0,
)


class _FakeMessages:
    """假的 ``client.messages``。"""

    def __init__(self, result: Any) -> None:
        """初始化。

        Args:
            result: 返回值或要抛出的异常。
        """
        self._result = result
        self.kwargs: dict[str, Any] = {}

    def create(self, **kwargs: Any) -> Any:
        """创建消息。

        Args:
            kwargs: SDK 参数。

        Returns:
            响应对象。

        Raises:
            Exception: 配置的异常。
        """
        self.kwargs = kwargs
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


def _provider(result: Any) -> tuple[AnthropicProvider, _FakeMessages]:
    """构造注入了假客户端的供应商。

    Args:
        result: 假客户端的返回值或异常。

    Returns:
        (供应商, 假 messages 对象)。
    """
    provider = AnthropicProvider("sk-test-key")
    messages = _FakeMessages(result)
    provider._client = SimpleNamespace(messages=messages)
    return provider, messages


def _message(
    blocks: list[Any],
    *,
    model: str = "claude-sonnet-4-5",
    usage: Any = None,
) -> SimpleNamespace:
    """构造一个 SDK 响应对象。

    Args:
        blocks: 内容块。
        model: 实际使用的模型。
        usage: 用量对象。

    Returns:
        响应对象。
    """
    return SimpleNamespace(content=blocks, model=model, usage=usage)


def _text_block(text: str) -> SimpleNamespace:
    """构造文本块。

    Args:
        text: 文本。

    Returns:
        内容块。
    """
    return SimpleNamespace(type="text", text=text)


class TestConstruction:
    """构造期校验。"""

    def test_satisfies_provider_protocol(self) -> None:
        """满足供应商协议，可被 ``LLMClient`` 直接使用。"""
        assert isinstance(AnthropicProvider("sk-test"), LLMProvider)

    @pytest.mark.parametrize("key", ["", "   "])
    def test_empty_key_rejected(self, key: str) -> None:
        """空 Key 当场报错。

        留到调用时才失败会让"忘了配环境变量"表现成一次莫名其妙的请求失败。
        """
        with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
            AnthropicProvider(key)

    def test_name(self) -> None:
        """供应商标识用于缓存键与用量记账。"""
        assert AnthropicProvider("sk-test").name == "anthropic"

    def test_missing_sdk_raises_llm_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """未安装 SDK 时给出可操作的错误信息。"""
        monkeypatch.setitem(sys.modules, "anthropic", None)
        provider = AnthropicProvider("sk-test")

        with pytest.raises(LLMError, match="anthropic SDK"):
            provider._ensure_client()


class TestComplete:
    """请求与响应翻译。"""

    def test_request_is_translated(self) -> None:
        """请求字段逐项翻译成 SDK 参数。"""
        provider, messages = _provider(_message([_text_block("好的")]))

        provider.complete(REQUEST)

        assert messages.kwargs["model"] == "claude-sonnet-4-5"
        assert messages.kwargs["max_tokens"] == 512
        assert messages.kwargs["temperature"] == 0.0
        assert messages.kwargs["system"] == "你是一个只做文本归纳的助手"
        assert messages.kwargs["messages"] == [{"role": "user", "content": "总结这条公告"}]

    def test_text_blocks_are_joined(self) -> None:
        """多个文本块按顺序拼接。"""
        provider, _ = _provider(_message([_text_block("第一段"), _text_block("第二段")]))

        assert provider.complete(REQUEST).text == "第一段\n第二段"

    def test_non_text_blocks_are_skipped(self) -> None:
        """非文本块（如工具调用）跳过而不是让整次调用失败。"""
        provider, _ = _provider(
            _message(
                [
                    SimpleNamespace(type="tool_use", name="calc", input={}),
                    _text_block("结论"),
                ]
            )
        )

        assert provider.complete(REQUEST).text == "结论"

    def test_empty_content_yields_empty_text(self) -> None:
        """空内容返回空串，交由上层的反幻觉校验判定。"""
        provider, _ = _provider(_message([]))

        assert provider.complete(REQUEST).text == ""

    def test_usage_is_reported(self) -> None:
        """用量透传给上层做预算记账（红线 LR7）。"""
        provider, _ = _provider(
            _message(
                [_text_block("x")],
                usage=SimpleNamespace(input_tokens=1200, output_tokens=340),
            )
        )

        response = provider.complete(REQUEST)

        assert response.input_tokens == 1200
        assert response.output_tokens == 340

    def test_missing_usage_degrades_to_zero(self) -> None:
        """拿不到用量时记 0，而不是让整次调用作废。

        token 计数只影响费用估算精度，为它丢掉一次已经付过费的结果不划算。
        """
        provider, _ = _provider(_message([_text_block("x")], usage=None))

        response = provider.complete(REQUEST)

        assert response.input_tokens == 0
        assert response.output_tokens == 0

    def test_actual_model_is_reported(self) -> None:
        """记录服务端实际使用的模型。

        请求里写的可能是别名，缓存与审计必须落实际模型（红线 R6 可复现）。
        """
        provider, _ = _provider(_message([_text_block("x")], model="claude-sonnet-4-5-20250929"))

        assert provider.complete(REQUEST).model == "claude-sonnet-4-5-20250929"

    def test_sdk_exception_is_wrapped(self) -> None:
        """SDK 异常包成 ``LLMError``，上层才能降级成"本次不使用 LLM"。"""
        provider, _ = _provider(TimeoutError("read timeout"))

        with pytest.raises(LLMError) as excinfo:
            provider.complete(REQUEST)

        assert "TimeoutError" in str(excinfo.value)
        assert "read timeout" in str(excinfo.value)

    def test_error_does_not_leak_api_key(self) -> None:
        """错误信息里不得出现 API Key（红线 R7）。"""
        provider, _ = _provider(RuntimeError("auth failed"))

        with pytest.raises(LLMError) as excinfo:
            provider.complete(REQUEST)

        assert "sk-test-key" not in str(excinfo.value)
