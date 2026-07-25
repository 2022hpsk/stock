"""LLM 供应商契约。

规范见 docs/10-大模型集成规格.md 第十节。

抽成 Protocol 有两个目的：换供应商时只改一个实现；测试用打桩替换，
**CI 永不打真实 API**。后者尤其重要——一个会在 CI 里花钱的测试套件
迟早会被人关掉。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

__all__ = ["ChatMessage", "CompletionRequest", "CompletionResponse", "LLMProvider"]


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """一条对话消息。"""

    role: str
    content: str


@dataclass(frozen=True, slots=True)
class CompletionRequest:
    """一次补全请求。"""

    model: str
    system: str
    messages: tuple[ChatMessage, ...]
    temperature: float = 0.0
    max_tokens: int = 2048
    top_p: float = 1.0

    def payload(self) -> dict[str, object]:
        """转成可入缓存的字典。

        Returns:
            请求负载。
        """
        return {
            "model": self.model,
            "system": self.system,
            "messages": [{"role": m.role, "content": m.content} for m in self.messages],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
        }


@dataclass(frozen=True, slots=True)
class CompletionResponse:
    """一次补全响应。"""

    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0

    def payload(self) -> dict[str, object]:
        """转成可入缓存的字典。

        Returns:
            响应负载。
        """
        return {
            "text": self.text,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_ms": self.latency_ms,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> CompletionResponse:
        """从缓存恢复。

        Args:
            payload: 缓存中的响应负载。

        Returns:
            响应对象。
        """
        return cls(
            text=str(payload.get("text", "")),
            model=str(payload.get("model", "")),
            input_tokens=_as_int(payload.get("input_tokens")),
            output_tokens=_as_int(payload.get("output_tokens")),
            latency_ms=_as_int(payload.get("latency_ms")),
        )


def _as_int(value: object) -> int:
    """把缓存里的任意值转成 int。

    缓存是外部输入（可能被手工编辑过），转不动时按 0 而不是抛错——
    token 计数出错不该让整个回放中断。

    Args:
        value: 待转换的值。

    Returns:
        整数；无法转换时 0。
    """
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int | float):
        return int(value)
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value)
    return 0


@runtime_checkable
class LLMProvider(Protocol):
    """大模型供应商。"""

    @property
    def name(self) -> str:
        """供应商标识。"""
        ...

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        """发起一次补全。

        Args:
            request: 补全请求。

        Returns:
            补全响应。

        Raises:
            LLMError: 调用失败。
        """
        ...
