"""Anthropic 供应商适配器。

只做一件事：把 ``CompletionRequest`` 翻译成 Anthropic SDK 调用，
再把结果翻译回 ``CompletionResponse``。

**缓存、预算、限流、重试、反幻觉校验全部不在这里**——它们在 ``LLMClient``
里，对所有供应商一视同仁。换供应商时只需再实现一遍这个协议，
那些保证一条都不会丢。

密钥只从环境变量读（红线 R7）。SDK 与 ``httpx`` 一样是延迟导入的：
没配 key 的用户不该因为装了个包就受影响，没装包的用户也不该起不来。
"""

from __future__ import annotations

from typing import Any

from quantstock.infra.errors import LLMError
from quantstock.infra.logging import get_logger
from quantstock.llm.protocols import CompletionRequest, CompletionResponse

__all__ = ["AnthropicProvider"]

_log = get_logger(__name__)

DEFAULT_TIMEOUT_SEC = 60.0


class AnthropicProvider:
    """Anthropic Messages API 适配器。"""

    def __init__(
        self,
        api_key: str,
        *,
        timeout_sec: float = DEFAULT_TIMEOUT_SEC,
        max_retries: int = 2,
    ) -> None:
        """初始化。

        Args:
            api_key: API Key。**只应来自环境变量**，绝不写进配置文件（红线 R7）。
            timeout_sec: 单次调用超时。
            max_retries: SDK 层重试次数。上层 ``LLMClient`` 不重试——
                重试逻辑放两处会让实际请求数变成两者的乘积，预算会失控。

        Raises:
            LLMError: API Key 为空。
        """
        if not api_key.strip():
            msg = "Anthropic API Key 为空。请设置环境变量 ANTHROPIC_API_KEY"
            raise LLMError(msg, provider="anthropic")
        self._api_key = api_key
        self._timeout = timeout_sec
        self._max_retries = max_retries
        self._client: Any = None

    @property
    def name(self) -> str:
        """供应商标识。"""
        return "anthropic"

    def _ensure_client(self) -> Any:  # noqa: ANN401 - 第三方 SDK 无类型存根
        """延迟构造 SDK 客户端。

        Returns:
            Anthropic 客户端。

        Raises:
            LLMError: 未安装 SDK。
        """
        if self._client is not None:
            return self._client
        try:
            import anthropic  # noqa: PLC0415 - 刻意延迟：没配 LLM 的用户不受影响
        except ImportError as exc:
            msg = "未安装 anthropic SDK，请执行：uv pip install anthropic"
            raise LLMError(msg, provider="anthropic") from exc

        self._client = anthropic.Anthropic(
            api_key=self._api_key, timeout=self._timeout, max_retries=self._max_retries
        )
        return self._client

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        """发起一次补全。

        Args:
            request: 补全请求。

        Returns:
            补全响应。

        Raises:
            LLMError: 调用失败或返回结构异常。
        """
        client = self._ensure_client()
        try:
            message = client.messages.create(
                model=request.model,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                system=request.system,
                messages=[{"role": m.role, "content": m.content} for m in request.messages],
            )
        except Exception as exc:
            # 抛 LLMError 而不是原样上抛：LLMClient 会把它降级成"本次不使用 LLM"，
            # 上层照常走纯量化路径
            msg = "Anthropic 调用失败"
            raise LLMError(msg, provider="anthropic", error=f"{type(exc).__name__}: {exc}") from exc

        return CompletionResponse(
            text=_extract_text(message),
            model=str(getattr(message, "model", request.model)),
            input_tokens=_usage(message, "input_tokens"),
            output_tokens=_usage(message, "output_tokens"),
        )


def _extract_text(message: Any) -> str:  # noqa: ANN401 - SDK 返回对象
    """从响应里抽出文本。

    Messages API 返回的是内容块数组（可能含工具调用等非文本块），
    这里只取文本块并拼接。

    Args:
        message: SDK 响应对象。

    Returns:
        拼接后的文本。
    """
    blocks = getattr(message, "content", None) or []
    parts = [
        str(block.text)
        for block in blocks
        if getattr(block, "type", None) == "text" and hasattr(block, "text")
    ]
    return "\n".join(parts).strip()


def _usage(message: Any, field: str) -> int:  # noqa: ANN401 - SDK 返回对象
    """取用量字段。

    取不到时返回 0 而不是抛错——token 计数拿不到只影响费用估算的精度，
    不该让整次调用作废。

    Args:
        message: SDK 响应对象。
        field: 字段名。

    Returns:
        token 数。
    """
    usage = getattr(message, "usage", None)
    value = getattr(usage, field, 0) if usage is not None else 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
