"""大模型供应商适配器。

目前只有 Anthropic。换供应商时只需再实现一遍 ``LLMProvider`` 协议，
上层的缓存、预算、限流、反幻觉校验全部不变。
"""

from quantstock.llm.providers.anthropic_provider import AnthropicProvider

__all__ = ["AnthropicProvider"]
