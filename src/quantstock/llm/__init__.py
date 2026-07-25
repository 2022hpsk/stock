"""大模型集成：客户端、缓存回放、结构化校验、三类任务（情报理解/研判归纳/解释生成）。

核心边界：**LLM 做「非结构化文本 → 结构化特征」，量化做「结构化特征 → 决策」**。
这条分界线同时解决了训练集泄漏——文本理解任务的输入就是当天的原文材料，
模型的"未来知识"帮不上忙。

LLM 只有 ``conviction_adjustment`` 一个数值出口，且被 α（默认 0.15、硬上限 0.20）
严格限幅。模型失效、幻觉、超预算的最坏结果都只是这个数字变成 0——
系统退化为纯量化，照常出建议。
"""

from quantstock.llm.anonymize import Anonymizer, strip_absolute_dates
from quantstock.llm.budget import BudgetGuard, estimate_cost
from quantstock.llm.cache import CacheEntry, CacheStats, LLMCache, compute_cache_key
from quantstock.llm.client import LLMClient, LLMMode, TaskCall, TaskResult
from quantstock.llm.influence import (
    HARD_INFLUENCE_CAP,
    InfluenceResult,
    apply_conviction,
    apply_exposure,
)
from quantstock.llm.protocols import (
    ChatMessage,
    CompletionRequest,
    CompletionResponse,
    LLMProvider,
)
from quantstock.llm.schemas import (
    ExplanationOutput,
    Factor,
    IntelClassification,
    MarketJudgement,
    PositionJudgement,
)
from quantstock.llm.tasks import (
    ExplainTask,
    IntelClassifyTask,
    JudgeOutcome,
    MarketJudgeTask,
    PositionJudgeTask,
)
from quantstock.llm.validate import parse_output, validate_evidence_refs

__all__ = [
    "HARD_INFLUENCE_CAP",
    "Anonymizer",
    "BudgetGuard",
    "CacheEntry",
    "CacheStats",
    "ChatMessage",
    "CompletionRequest",
    "CompletionResponse",
    "ExplainTask",
    "ExplanationOutput",
    "Factor",
    "InfluenceResult",
    "IntelClassification",
    "IntelClassifyTask",
    "JudgeOutcome",
    "LLMCache",
    "LLMClient",
    "LLMMode",
    "LLMProvider",
    "MarketJudgeTask",
    "MarketJudgement",
    "PositionJudgeTask",
    "PositionJudgement",
    "TaskCall",
    "TaskResult",
    "apply_conviction",
    "apply_exposure",
    "compute_cache_key",
    "estimate_cost",
    "parse_output",
    "strip_absolute_dates",
    "validate_evidence_refs",
]
